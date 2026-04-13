"""
Unit tests for ml_engine.agent.memory.MemoryStore.

Uses fakeredis.
"""

from __future__ import annotations

import pytest
import fakeredis

from ml_engine.agent.memory import MemoryStore, MEMORY_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def redis():
    r"""Returns a fresh fakeredis instance for each test.
    Note that fakeredis returns bytes by default, which
    simulates real Redis behavior but is less convenient
    for testing. We decode responses to get strings instead."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis):
    return MemoryStore(redis)


# ---------------------------------------------------------------------------
# write() / read()
# ---------------------------------------------------------------------------

class TestWriteRead:
    r"""Tests for basic write() and read() functionality of MemoryStore."""
    def test_write_and_read_by_key(self, store):
        r"""Test that writing a record and then reading it by key returns the correct content."""
        store.write("feedback", "key1", {"verdict": "pass", "mAP50": 0.7})
        records = store.read("feedback", "key1")
        assert len(records) == 1
        assert records[0]["content"]["verdict"] == "pass"

    def test_read_all_returns_all_records(self, store):
        r"""Test that read() with key=None returns all records of that type."""
        store.write("feedback", "k1", {"verdict": "pass"})
        store.write("feedback", "k2", {"verdict": "retry"})
        records = store.read("feedback")
        assert len(records) == 2

    def test_read_nonexistent_key_returns_empty(self, store):
        r"""Test that reading a non-existent key returns an empty list."""
        records = store.read("feedback", "does-not-exist")
        assert records == []

    def test_write_overwrites_existing_key(self, store):
        r"""Test that writing to an existing key overwrites the previous content."""
        store.write("feedback", "dup", {"value": 1})
        store.write("feedback", "dup", {"value": 2})
        records = store.read("feedback", "dup")
        assert len(records) == 1
        assert records[0]["content"]["value"] == 2

    @pytest.mark.parametrize("mem_type", MEMORY_TYPES)
    def test_write_all_valid_types(self, store, mem_type):
        r"""Test that writing records of all valid memory types works without error."""
        store.write(mem_type, "test-key", {"data": "x"})
        records = store.read(mem_type, "test-key")
        assert len(records) == 1

    def test_write_invalid_type_raises(self, store):
        r"""Test that writing with an invalid memory type raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            store.write("invalid_type", "k", {})

    def test_read_invalid_type_raises(self, store):
        r"""Test that reading with an invalid memory type raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            store.read("invalid_type")

    def test_record_has_type_and_key_fields(self, store):
        r"""Test that the stored record includes 'type' and 'key' fields in the output."""
        store.write("project", "proj-key", {"info": "stuff"})
        record = store.read("project", "proj-key")[0]
        assert record["type"] == "project"
        assert record["key"] == "proj-key"

    def test_record_has_updated_at(self, store):
        r"""Test that the stored record includes an 'updated_at' timestamp field."""
        store.write("user", "u1", {})
        record = store.read("user", "u1")[0]
        assert "updated_at" in record

    def test_content_is_parsed_dict(self, store):
        r"""Test that the 'content' field is stored as a JSON string but returned as a parsed dict."""
        store.write("reference", "ref1", {"url": "http://example.com", "desc": "API"})
        record = store.read("reference", "ref1")[0]
        assert isinstance(record["content"], dict)
        assert record["content"]["url"] == "http://example.com"

    def test_empty_content_dict(self, store):
        r"""Test that writing an empty dict as content is handled correctly."""
        store.write("feedback", "empty-content", {})
        record = store.read("feedback", "empty-content")[0]
        assert record["content"] == {}

    def test_nested_content_survives_roundtrip(self, store):
        r"""Test that nested dicts in content are correctly stored and retrieved."""
        nested = {"metrics": {"mAP50": 0.72, "nested": {"deep": True}}}
        store.write("feedback", "nested", nested)
        record = store.read("feedback", "nested")[0]
        assert record["content"]["metrics"]["nested"]["deep"] is True


# ---------------------------------------------------------------------------
# to_llm_context()
# ---------------------------------------------------------------------------

class TestToLlmContext:
    r"""Tests for the to_llm_context() method of MemoryStore, which formats memory for LLM input."""
    def test_returns_no_memory_when_empty(self, store):
        r"""Test that when there are no records, to_llm_context() returns a placeholder string."""
        result = store.to_llm_context()
        assert result == "(no memory)"

    def test_includes_memory_type_header(self, store):
        r"""Test that the output includes a header for the memory type."""
        store.write("feedback", "k1", {"verdict": "pass"})
        result = store.to_llm_context(types=["feedback"])
        assert "## Memory: feedback\n  [k1] {\"verdict\": \"pass\"}" in result

    def test_includes_key_in_output(self, store):
        r"""Test that the output includes the key of the memory record."""
        store.write("feedback", "my-key", {"verdict": "pass"})
        result = store.to_llm_context(types=["feedback"])
        assert "my-key" in result

    def test_filters_to_requested_types(self, store):
        r"""Test that only the requested memory types are included in the output."""
        store.write("feedback", "k1", {"data": 1})
        store.write("project", "k2", {"data": 2})
        result = store.to_llm_context(types=["feedback"])
        assert "## Memory: feedback" in result
        assert "## Memory: project" not in result

    def test_multiple_records_all_appear(self, store):
        r"""Test that if multiple records of the same type exist, they all appear in the output."""
        store.write("feedback", "run1", {"verdict": "pass"})
        store.write("feedback", "run2", {"verdict": "retry"})
        result = store.to_llm_context(types=["feedback"])
        assert "run1" in result
        assert "run2" in result


# ---------------------------------------------------------------------------
# _parse_record() -- corrupt data resilience
# ---------------------------------------------------------------------------

class TestParseRecord:
    def test_corrupt_content_json_returns_raw(self, redis, store):
        """Simulate Redis returning garbled content bytes."""
        redis.hset("mem:feedback:corrupt", mapping={
            b"type": b"feedback",
            b"key": b"corrupt",
            b"content": b"NOT JSON {{{",
            b"updated_at": b"2026-01-01",
        })
        records = store.read("feedback", "corrupt")
        assert len(records) == 1
        assert "raw" in records[0]["content"]
