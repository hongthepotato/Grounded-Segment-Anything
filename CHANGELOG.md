# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] — 2026-04-28

### Fixed

- **Coordinator durability on restart** — `resume_orphaned_coordinators()` now runs at FastAPI startup (`api/app.py` lifespan). Any pipeline run that was in a non-terminal state when the previous process died is automatically re-launched without needing a manual re-approve call.
- **Idempotent `POST /api/agent/approve`** — First call (`state=created`) transitions the state machine and publishes `contract_approved`. Subsequent calls on the same run skip the duplicate transition and event publish so the Coordinator can pick up from its Redis Stream PEL cursor. Returns 409 for terminal runs.
- **`store_approved_contract` ordering** — The approved contract is now persisted in Redis _before_ the `created → planning` state transition. Previously, a crash between the two writes would leave the run in `planning` with no stored contract, permanently blocking orphan recovery.
- **`_on_done` callback `CancelledError` crash** — `asyncio.Task.exception()` raises `CancelledError` on cancelled tasks. The callback now checks `t.cancelled()` first; cancelled tasks are logged and cleaned up without attempting to read the exception.
- **Agent Redis client cleanup on shutdown** — `close_async_redis_client()` is now called in the FastAPI lifespan shutdown block alongside `close_async_job_managers()`. Prevents event-loop contamination across successive `TestClient` uses in integration tests.
- **Falsy-dict treated as `None`** — `get_approved_contract` and `get_proposed_contract` used `result if result else None`, which would return `None` for a stored empty dict `{}` (falsy). Changed to `result if result is not None else None`.
- **Silent `JSONDecodeError`** — Malformed JSON in the `approved_contract` or `proposed_contract` Redis fields now emits a `logger.warning` before returning `None`.

### Added

- `StateMachine.store_approved_contract` / `get_approved_contract` — persist the approved contract in the run's Redis HASH so startup recovery can reconstruct the Coordinator after a container restart.
- `StateMachine.scan_non_terminal_run_ids` — SCAN-based classmethod that returns run IDs not in a terminal state; used by orphan recovery.
- `StateMachine.exists` — lightweight existence check without raising on missing keys.
- TODOs #21–24 filed: `pending_contract_approval` exit endpoint missing, `failed_retrying` retry dispatch missing, worker task self-termination stub, multi-instance Coordinator collision risk.

### Security

- TODO comment added to `POST /api/agent/approve` and `POST /api/agent/gate/{rid}/{action}` noting that these endpoints have no authentication and should have an API-key guard before public exposure.

## [0.1.0] — initial release
