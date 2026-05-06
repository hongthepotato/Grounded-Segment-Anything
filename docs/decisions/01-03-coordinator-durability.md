# 01 + 03 — Coordinator durability: orphan resume + idempotent approve

Shipped 2026-04-27 on branch `agentic` (commit `dc86cbd`).
Companion to [02-coordinator-crash-failed-unrecoverable.md](02-coordinator-crash-failed-unrecoverable.md).

---

## What shipped

### StateMachine additions (`ml_engine/agent/state_machine.py`)

**`store_approved_contract(contract)`** / **`get_approved_contract()`**
Persist the approved `PipelineContract` dict inside the existing Redis HASH
(`run:{run_id}:state` field `approved_contract`). The Coordinator holds the
contract only in RAM (`self._contract`); after any container restart that
in-memory value is gone. Startup orphan recovery calls `get_approved_contract`
to reconstruct the dict and pass it to `_start_coordinator`.

**`scan_non_terminal_run_ids(redis_async)` (classmethod)**
`SCAN`s all `run:*:state` keys, strips the prefix/suffix to extract `run_id`,
reads the `state` field, and filters to runs not in `TERMINAL_STATES`. O(N)
over active-run count; safe at expected scale (<100 active runs). Used by
`resume_orphaned_coordinators`.

### `resume_orphaned_coordinators()` (`api/routes/agent.py`)

Called once from the FastAPI lifespan startup hook (after stale-worker cleanup
in `api/app.py`). Algorithm:

1. Call `scan_non_terminal_run_ids` to get candidates.
2. Skip runs whose `asyncio.Task` is already alive in `_coordinator_tasks`.
3. Load `get_approved_contract` for each remaining run.
4. Skip runs with no stored contract — these are in `created` or
   `pending_contract_approval` and haven't been approved yet; no Coordinator
   is expected.
5. Call `_start_coordinator(run_id, contract_dict)` for the rest.

`_start_coordinator` is already idempotent (no-ops if task is running), so
calling `resume_orphaned_coordinators` twice is safe.

### Idempotent `approve_plan` (`api/routes/agent.py`)

Old behaviour: unconditionally called `sm.transition("planning")`, which raised
`ValueError` if the run was already past `created`. That 400 leaked an internal
state-machine detail to the caller.

New behaviour:

| Current state | Action |
|---|---|
| `created` | Transition `created → planning`, publish `contract_approved`, persist contract, launch Coordinator |
| Non-terminal, past `created` | Skip transition + event (Coordinator resumes from stream PEL); update persisted contract; launch/no-op Coordinator |
| Terminal | Return 409 |

Skipping the event re-publish on re-approve is correct: the consumer group's
last-delivered-id has already advanced past `contract_approved` (or the message
is in the PEL awaiting ACK). Re-publishing would cause the event to be
processed twice.

---

## Stream consumer resume semantics (why re-publish is safe to skip)

The stream consumer uses `XREADGROUP` with `id=">"` which means "give me only
new messages after the group's last-delivered-id." That cursor is:

- Stored by Redis internally in the consumer group record.
- Durable via AOF (`appendonly yes` in `docker-compose.yml` + named volume
  `redis-data`).
- Preserved across restarts because `ensure_consumer_group` catches
  `BUSYGROUP` silently (an existing group with its existing cursor is left
  intact).
- Bootstrapped from `"0"` (beginning of stream) only when the group is first
  created — not on every Coordinator launch.

The PEL (`_drain_pel` using `XPENDING` + `XCLAIM`) reclaims any messages that
were delivered to the old consumer but never ACKed. So a resumed Coordinator
replays exactly the unfinished work and then continues with new messages.

**`LoopState.last_event_id`** is a separate field stored in
`agent:{run_id}:loop_state`. It tracks the last event the `AgentLoop` has
*processed* (for higher-level resume logic), not the Redis stream cursor. The
two are independent; `last_event_id` does not affect what `XREADGROUP` returns.

---

## What was NOT changed

**Leader election.** `resume_orphaned_coordinators` will be called by every
FastAPI replica if the service is scaled to >1. Each replica will attempt to
launch its own Coordinator task for the same run. The current single-instance
deployment avoids this; multi-replica needs a distributed lock (e.g. Redis
`SET NX`) or an external coordinator election mechanism. Filed as a note in
TODO #1's original entry (now Completed).

**Job liveness detection.** If a subprocess (teacher training, auto-labeling)
dies silently during pure computation without publishing a `job_failed` event,
the Coordinator blocks indefinitely waiting for that event. This is a separate
concern from container-restart durability. The Coordinator resumes correctly
after restart but still has no way to detect a silent subprocess death. Needs
its own watchdog / heartbeat mechanism.

**Crash type refinement (TODO #20).** `_on_done` currently marks all Coordinator
crashes as `failed_unrecoverable`. Transient errors (OOM, Redis blip) could
instead transition to `failed_retrying` and be retried automatically. Blocked
on designing retry backoff + a way to distinguish transient from permanent
failures.

---

## Tests added (`tests/unit/api/test_agent_coordinator.py`)

| Class | Coverage |
|---|---|
| `TestIdempotentApprove` | First approve, re-approve (already-planning), 409 on terminal, 404 on missing run |
| `TestResumeOrphanedCoordinators` | No runs, skips terminal, skips pre-approve, resumes non-terminal, skips already-running tasks, multi-run mixed |
| `TestApprovedContract` | Round-trip through store/get; get returns None before store; empty dict treated as None |
| `TestScanNonTerminalRunIds` | Empty Redis, single non-terminal, terminal filtered, multiple runs mixed, missing state key skipped |

All 36 new tests pass. Total test count for the file: 365.
