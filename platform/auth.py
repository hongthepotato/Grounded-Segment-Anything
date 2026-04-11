"""PIN-based auth for the platform.

Two roles: supervisor and engineer.
PINs are stored bcrypt-hashed in auth.json.
No sessions — PIN is verified per-request via Authorization header or form field.
Single-workstation assumption: no network auth.

auth.json format:
{
  "users": [
    {"username": "supervisor1", "role": "supervisor", "pin_hash": "<bcrypt>"},
    {"username": "engineer1",   "role": "engineer",   "pin_hash": "<bcrypt>"}
  ]
}
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger(__name__)

AUTH_FILE = Path(__file__).parent.parent / "auth.json"

_security = HTTPBasic(auto_error=False)


def _load_auth() -> dict:
    if AUTH_FILE.exists():
        with open(AUTH_FILE) as f:
            return json.load(f)
    return {"users": []}


def _save_auth(data: dict) -> None:
    with open(AUTH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_first_run() -> bool:
    """True if auth.json doesn't exist or has no users."""
    data = _load_auth()
    return len(data.get("users", [])) == 0


def create_user(username: str, pin: str, role: str) -> None:
    """Hash the PIN and store the user. Role must be 'supervisor' or 'engineer'."""
    if role not in ("supervisor", "engineer"):
        raise ValueError(f"Unknown role: {role}")
    pin_hash = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
    data = _load_auth()
    # Prevent duplicate usernames
    data["users"] = [u for u in data["users"] if u["username"] != username]
    data["users"].append({"username": username, "role": role, "pin_hash": pin_hash})
    _save_auth(data)
    logger.info(f"Created user '{username}' with role '{role}'")


def verify_pin(username: str, pin: str) -> Optional[dict]:
    """Return the user dict if PIN is correct, else None."""
    data = _load_auth()
    for user in data.get("users", []):
        if user["username"] == username:
            if bcrypt.checkpw(pin.encode(), user["pin_hash"].encode()):
                return user
    return None


def _get_current_user(credentials: Optional[HTTPBasicCredentials] = Depends(_security)) -> dict:
    """FastAPI dependency: validate HTTP Basic credentials against auth.json."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user = verify_pin(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or PIN",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


def require_auth(user: dict = Depends(_get_current_user)) -> dict:
    """Require any authenticated user (supervisor or engineer)."""
    return user


def require_engineer(user: dict = Depends(_get_current_user)) -> dict:
    """Require engineer role."""
    if user.get("role") != "engineer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Engineer role required")
    return user


def list_users() -> list:
    data = _load_auth()
    return [{"username": u["username"], "role": u["role"]} for u in data.get("users", [])]


def delete_user(username: str) -> bool:
    data = _load_auth()
    before = len(data["users"])
    data["users"] = [u for u in data["users"] if u["username"] != username]
    if len(data["users"]) < before:
        _save_auth(data)
        return True
    return False
