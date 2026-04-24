"""
Unit tests for ml_engine.agent.memory.MemoryStore.

Async-only after Phase 7 (sync API removed). Uses the async fakeredis
fixture ``redis_async`` from conftest (shares FakeServer with ``redis_sync``
when cross-client parity is needed, though MemoryStore itself has no
sync API anymore).
"""

from __future__ import annotations

import pytest

from ml_engine.agent.memory import MemoryStore, MEMORY_TYPES


@pytest.fixture
def store(redis_async):
    return MemoryStore(redis_async=redis_async)


# ---------------------------------------------------------------------------
# write() / read()
# ---------------------------------------------------------------------------

class TestWriteRead:
    @pytest.mark.asyncio
    async def test_write_and_read_by_key(self, store):
        await store.write("feedback", "key1", {"verdict": "pass", "mAP50": 0.7})
        records = await store.read("feedback", "key1")
        assert len(records) == 1
        assert records[0]["content"]["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_read_all_returns_all_records(self, store):
        await store.write("feedback", "k1", {"verdict": "pass"})
        await store.write("feedback", "k2", {"verdict": "retry"})
        records = await store.read("feedback")
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_read_nonexistent_key_returns_empty(self, store):
        records = await store.read("feedback", "does-not-exist")
        assert records == []

    @pytest.mark.asyncio
    async def test_write_overwrites_existing_key(self, store):
        await store.write("feedback", "dup", {"value": 1})
        await store.write("feedback", "dup", {"value": 2})
        records = await store.read("feedback", "dup")
        assert len(records) == 1
        assert records[0]["content"]["value"] == 2

    @pytest.mark.parametrize("mem_type", sorted(MEMORY_TYPES))
    @pytest.mark.asyncio
    async def test_write_all_valid_types(self, store, mem_type):
        await store.write(mem_type, "test-key", {"data": "x"})
        records = await store.read(mem_type, "test-key")
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_write_invalid_type_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory type"):
            await store.write("invalid_type", "k", {})

    @pytest.mark.asyncio
    async def test_read_invalid_type_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory type"):
            await store.read("invalid_type")

    @pytest.mark.asyncio
    async def test_record_has_type_and_key_fields(self, store):
        await store.write("project", "proj-key", {"info": "stuff"})
        record = (await store.read("project", "proj-key"))[0]
        assert record["type"] == "project"
        assert record["key"] == "proj-key"

    @pytest.mark.asyncio
    async def test_record_has_updated_at(self, store):
        await store.write("user", "u1", {})
        record = (await store.read("user", "u1"))[0]
        assert "updated_at" in record

    @pytest.mark.asyncio
    async def test_content_is_parsed_dict(self, store):
        await store.write("reference", "ref1", {"url": "http://example.com", "desc": "API"})
        record = (await store.read("reference", "ref1"))[0]
        assert isinstance(record["content"], dict)
        assert record["content"]["url"] == "http://example.com"

    @pytest.mark.asyncio
    async def test_empty_content_dict(self, store):
        await store.write("feedback", "empty-content", {})
        record = (await store.read("feedback", "empty-content"))[0]
        assert record["content"] == {}

    @pytest.mark.asyncio
    async def test_nested_content_survives_roundtrip(self, store):
        nested = {"metrics": {"mAP50": 0.72, "nested": {"deep": True}}}
        await store.write("feedback", "nested", nested)
        record = (await store.read("feedback", "nested"))[0]
        assert record["content"]["metrics"]["nested"]["deep"] is True


# ---------------------------------------------------------------------------
# to_llm_context()
# ---------------------------------------------------------------------------

class TestToLlmContext:
    @pytest.mark.asyncio
    async def test_returns_no_memory_when_empty(self, store):
        result = await store.to_llm_context()
        assert result == "(no memory)"

    @pytest.mark.asyncio
    async def test_includes_memory_type_header(self, store):
        await store.write("feedback", "k1", {"verdict": "pass"})
        result = await store.to_llm_context(types=["feedback"])
        assert "## Memory: feedback\n  [k1] {\"verdict\": \"pass\"}" in result

    @pytest.mark.asyncio
    async def test_includes_key_in_output(self, store):
        await store.write("feedback", "my-key", {"verdict": "pass"})
        result = await store.to_llm_context(types=["feedback"])
        assert "my-key" in result

    @pytest.mark.asyncio
    async def test_filters_to_requested_types(self, store):
        await store.write("feedback", "k1", {"data": 1})
        await store.write("project", "k2", {"data": 2})
        result = await store.to_llm_context(types=["feedback"])
        assert "## Memory: feedback" in result
        assert "## Memory: project" not in result

    @pytest.mark.asyncio
    async def test_multiple_records_all_appear(self, store):
        await store.write("feedback", "run1", {"verdict": "pass"})
        await store.write("feedback", "run2", {"verdict": "retry"})
        result = await store.to_llm_context(types=["feedback"])
        assert "run1" in result
        assert "run2" in result


# ---------------------------------------------------------------------------
# _parse_record() -- corrupt data resilience
# ---------------------------------------------------------------------------

class TestParseRecord:
    @pytest.mark.asyncio
    async def test_corrupt_content_json_returns_raw(self, redis_async, store):
        """Simulate Redis returning garbled content bytes."""
        await redis_async.hset("mem:feedback:corrupt", mapping={
            "type": "feedback",
            "key": "corrupt",
            "content": "NOT JSON {{{",
            "updated_at": "2026-01-01",
        })
        records = await store.read("feedback", "corrupt")
        assert len(records) == 1
        assert "raw" in records[0]["content"]
