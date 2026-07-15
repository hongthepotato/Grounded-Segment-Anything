# 2. Coordinator crash → `failed_unrecoverable`

**Status:** Complete 2026-04-27.

## What shipped

When a Coordinator asyncio task raises an unhandled exception, the run now
transitions to `failed_unrecoverable` with the exception message stored in
Redis so `GET /api/agent/status` returns a clear failure state instead of a
stale state with `coordinator_active: false` and no error context.

**Three files changed:**

- `ml_engine/agent/state_machine.py` — TRANSITIONS expanded: `failed_unrecoverable`
  was previously unreachable from `pending_contract_approval`, `label_review_gate`,
  `training_eval_gate`, `distill_eval_gate`, `pending_approval`. Added to all five.
  The original intent ("failure transitions from any non-terminal state") was stated
  in the module docstring but only partially implemented.

- `api/routes/agent.py` — `_on_done` callback (registered on the Coordinator task)
  now schedules a `_mark_failed()` coroutine via `asyncio.create_task` when
  `t.exception()` is not None. The coroutine calls
  `StateMachine.transition("failed_unrecoverable", error_message=str(exc))`.
  Errors from the transition itself (e.g., run already terminal via a concurrent
  path) are caught and logged rather than propagated — the event loop has no
  caller to surface them to at that point.

- `tests/unit/api/test_agent_coordinator.py` — 5 new tests covering: crash →
  `failed_unrecoverable`, error message stored verbatim, clean exit unaffected,
  mid-pipeline crash (state != `planning`), and the already-terminal edge case.

## asyncio sync-callback constraint

`_on_done` is a *sync* callback added via `task.add_done_callback(...)`. It runs
on the event loop thread but cannot `await`. The solution is `asyncio.create_task`
— schedules the coroutine without blocking the callback. This works because
`_on_done` is always called from within the running event loop (asyncio calls
done callbacks from `Task.__step`), so `create_task` can find the running loop.

## Known limitation: all crashes → unrecoverable

The current implementation classifies every Coordinator exception as
`failed_unrecoverable` regardless of exception type. This is intentional as a
first pass — simple and observable. The follow-on (TODO #20) is to distinguish
transient failures (import errors, Redis connection reset, network timeout) and
send those to `failed_retrying` instead, enabling automatic recovery. That
requires auto-resume infra (TODO #1) to exist first so the retry actually kicks
off a new Coordinator.

## Stream_consumer resume audit (parallel to this work)

A concurrent audit of `ml_engine/agent/stream_consumer.py` confirmed the
pre-blocker for TODO #1 (auto-resume): the consumer group is initialized with
`id="0"` and XREADGROUP reads with `">"` (new messages only), but the group
cursor starts at stream-beginning. A Coordinator re-launch would replay *all*
historical events, causing duplicate LLM calls and job submissions. Fix needed
before TODO #1 ships: store the last-ACKed stream ID in Redis and resume from
it, or use a versioned consumer group name per Coordinator lifecycle.
