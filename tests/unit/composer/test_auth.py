"""Unit tests for composer/auth.py.

Covers:
- create_user / verify_pin: round-trip, wrong pin, unknown user
- is_first_run: true when no users, false when users exist
- list_users / delete_user: basic operations
- require_engineer: raises 403 for supervisor role
"""

import json
import sys
from pathlib import Path

import pytest

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def auth_file(tmp_path):
    return tmp_path / "auth.json"


@pytest.fixture
def auth_module(auth_file, monkeypatch):
    import composer.auth as auth

    monkeypatch.setattr(auth, "AUTH_FILE", auth_file)
    return auth


class TestCreateAndVerify:
    def test_create_and_verify_round_trip(self, auth_module):
        auth = auth_module
        auth.create_user("alice", "1234", "engineer")
        user = auth.verify_pin("alice", "1234")
        assert user is not None
        assert user["role"] == "engineer"
        assert user["username"] == "alice"

    def test_wrong_pin_returns_none(self, auth_module):
        auth = auth_module
        auth.create_user("bob", "9999", "supervisor")
        assert auth.verify_pin("bob", "0000") is None

    def test_unknown_user_returns_none(self, auth_module):
        auth = auth_module
        assert auth.verify_pin("nobody", "1234") is None

    def test_create_duplicate_username_replaces_old(self, auth_module):
        auth = auth_module
        auth.create_user("alice", "1111", "supervisor")
        auth.create_user("alice", "2222", "engineer")  # replace
        assert auth.verify_pin("alice", "1111") is None
        user = auth.verify_pin("alice", "2222")
        assert user["role"] == "engineer"

    def test_create_user_rejects_invalid_role(self, auth_module):
        auth = auth_module
        with pytest.raises(ValueError, match="Unknown role"):
            auth.create_user("x", "1234", "admin")

    def test_pin_hash_not_stored_in_plaintext(self, auth_module, auth_file):
        auth = auth_module
        auth.create_user("carol", "secret", "supervisor")
        data = json.loads(auth_file.read_text())
        stored = data["users"][0]["pin_hash"]
        assert "secret" not in stored
        assert stored.startswith("$2b$")  # bcrypt prefix


class TestFirstRun:
    def test_first_run_true_when_no_auth_file(self, auth_module, auth_file):
        auth = auth_module
        assert not auth_file.exists()
        assert auth.is_first_run() is True

    def test_first_run_true_when_empty_users(self, auth_module, auth_file):
        auth = auth_module
        auth_file.write_text(json.dumps({"users": []}))
        assert auth.is_first_run() is True

    def test_first_run_false_when_user_exists(self, auth_module):
        auth = auth_module
        auth.create_user("admin", "0000", "engineer")
        assert auth.is_first_run() is False


class TestListAndDelete:
    def test_list_users_excludes_hash(self, auth_module):
        auth = auth_module
        auth.create_user("a", "1111", "supervisor")
        auth.create_user("b", "2222", "engineer")
        users = auth.list_users()
        assert len(users) == 2
        for u in users:
            assert "pin_hash" not in u
            assert "username" in u
            assert "role" in u

    def test_delete_existing_user(self, auth_module):
        auth = auth_module
        auth.create_user("del_me", "1234", "supervisor")
        result = auth.delete_user("del_me")
        assert result is True
        assert auth.verify_pin("del_me", "1234") is None

    def test_delete_nonexistent_user_returns_false(self, auth_module):
        auth = auth_module
        assert auth.delete_user("nobody") is False


class TestRequireEngineer:
    def test_engineer_passes(self, auth_module):
        auth = auth_module
        user = {"username": "eng", "role": "engineer"}
        result = auth.require_engineer(user=user)
        assert result == user

    def test_supervisor_raises_403(self, auth_module):
        auth = auth_module
        from fastapi import HTTPException

        user = {"username": "sup", "role": "supervisor"}
        with pytest.raises(HTTPException) as exc:
            auth.require_engineer(user=user)
        assert exc.value.status_code == 403
