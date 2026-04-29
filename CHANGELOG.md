# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] — 2026-04-29

### Fixed

- **Coordinator crash classification** — Coordinator task crashes are now classified as transient or permanent before deciding whether to retry. Transient infrastructure failures (`ConnectionError`, `TimeoutError`, `InterruptedError`, `redis.exceptions.ConnectionError/TimeoutError`) in states that allow it (`auto_labeling`, `teacher_training`, `student_distillation`) are routed to `failed_retrying` and the Coordinator is re-launched automatically. Permanent errors (logic bugs, bad input, etc.) and transient errors in non-retryable states go straight to `failed_unrecoverable`. Respects `budget.max_retries` (default 2).
- **`TRANSIENT_EXCEPTION_TYPES` scope narrowed** — The constant previously included bare `OSError`, which would have misclassified `FileNotFoundError`, `PermissionError`, and `ChildProcessError` as retryable infrastructure failures. Now uses specific `OSError` subclasses only: `ConnectionError`, `TimeoutError`, `InterruptedError`.
- **Coordinator stuck on `failed_retrying` transition failure** — If `sm.transition("failed_retrying")` itself fails (e.g., concurrent state change), the crash handler now falls through to `failed_unrecoverable` rather than returning silently and leaving the run stranded in a non-terminal state with no active Coordinator task.

### Added

- `_handle_coordinator_crash()` — module-level async function in `api/routes/agent.py` that encapsulates the crash-routing logic. Replaces the nested `_mark_failed()` inner function, eliminating a three-layer nesting and making the retry logic independently testable.
- `_is_transient_exception()` — module-level helper that checks an exception against `core.constants.TRANSIENT_EXCEPTION_TYPES` and, via lazy import, `redis.exceptions.ConnectionError` / `TimeoutError`.
- `TRANSIENT_EXCEPTION_TYPES` tuple in `core/constants.py` — single source of truth for which exception families count as transient infrastructure failures, importable by workers and the Coordinator in addition to the API layer.

### Tests

- `TestCrashClassification` added to `tests/unit/api/test_agent_coordinator.py` (13 tests): transient-in-retryable-state → `failed_retrying` + coordinator re-launch (3 states covered); transient-in-non-retryable-state → `failed_unrecoverable`; permanent error → `failed_unrecoverable`; retries-exhausted → `failed_unrecoverable`; error-message persisted on `failed_retrying`; `retry_count` incremented; `_is_transient_exception` for builtin types (Connection, Timeout, Interrupted, BrokenPipe, ConnectionReset); non-transient types including `OSError`, `FileNotFoundError`, `PermissionError`; second-crash-from-`failed_retrying` terminates at `failed_unrecoverable`; missing Redis key returns silently.

## [0.1.4] — 2026-04-28

### Fixed

- **`pending_contract_approval` gate endpoint** — `POST /api/agent/gate/{run_id}/approve` now handles both human gate states. Previously the endpoint only accepted runs in `pending_approval`; runs paused at `pending_contract_approval` (the start-of-pipeline contract review) had no way to advance and would stall permanently. Approve transitions to `auto_labeling` (event: `contract_approved`); reject transitions to `cancelled` (event: `contract_rejected`).
- **`pending_approval` reject state bug** — rejecting at the end-of-pipeline gate previously attempted to transition to `escalated`, which is not a valid target from `pending_approval` per the state machine. Fixed to `cancelled` (consistent with human-initiated cancellation). Discovered and caught by the new gate tests.

### Tests

- `TestHumanGate` added to `tests/unit/api/test_agent_coordinator.py` (13 tests): covers both gate states (approve/reject), invalid action (400), unknown run (404), non-gate states (409), event type assertions for `contract_approved` / `contract_rejected`, and `ValueError` propagation from `sm.transition()` → 400.

## [0.1.3] — 2026-04-28

### Fixed

- **`_keep_higher_p` error message when both dicts missing `p`** — previously only reported one of the two missing dicts; now reports all missing dicts in a single `ValueError`. Symmetric test cases added for `new`-missing and both-missing paths.

## [0.1.2] — 2026-04-28

### Fixed

- **`upscale_masks` 5D crash** — `SAMHQLoRA.upscale_masks()` now handles 5D `[B,N,K,H,W]` multimask tensors by reshaping to 4D before interpolation and restoring shape after. Previously raised `RuntimeError: expected 4D input` when called with multimask output.
- **`SegmentationLoss` ignored `iou_predictions`** — The IoU quality regression head was never trained. Added MSE loss between `iou_predictions [B,N]` and actual mask IoU computed under `torch.no_grad()`. Weight controlled by `loss_weights["iou_quality"]` (default `1.0`); callers who pass custom `loss_weights` without this key get weight `0.0` (no change to their training).
- **`box_prompts` N=0 opaque crash** — `SAMHQLoRA.forward()` now raises `ValueError("at least 1 object")` immediately when `box_prompts.shape[1] == 0`, instead of propagating an opaque `RuntimeError` from `torch.cat` deep in the stack.
- **`point_prompts` N=0 guard** — Symmetric guard added: raises `ValueError` when `point_prompts[0].shape[1] == 0`.
- **`pred_masks` dimensionality guard in `SegmentationLoss`** — Now raises `ValueError` on entry if `pred_masks.ndim != 4`, preventing 5D multimask tensors from silently computing wrong loss via `view(b*n, -1)` treating `K*H*W` as the pixel dimension.
- **`iou_predictions` shape guard in `SegmentationLoss`** — Raises `ValueError` with clear message if `iou_predictions` is not `[B, N]` (e.g. `[B, N, K]` multimask), instead of crashing silently or misaligning indices.

### Tests

- Moved `tests/test_sam_lora.py` → `tests/unit/ml_engine/test_sam_lora.py` to match project layout convention.
- Expanded from 18 to 24 tests: added `test_5d_non_square_target`, `test_iou_predictions_multimask_shape_raises`, `test_iou_quality_weight_zero_excludes_quality_from_total`, `test_forward_empty_box_list` (narrowed to match new `ValueError`).

## [0.1.1] — 2026-04-28

### Fixed

- **Coordinator durability on restart** — `resume_orphaned_coordinators()` now runs at FastAPI startup (`api/app.py` lifespan). Any pipeline run that was in a non-terminal state when the previous process died is automatically re-launched without needing a manual re-approve call.
- **Idempotent `POST /api/agent/approve`** — First call (`state=created`) transitions the state machine and publishes `contract_approved`. Subsequent calls on the same run skip the duplicate transition and event publish so the Coordinator can pick up from its Redis Stream PEL cursor. Returns 409 for terminal runs.
- **`store_approved_contract` ordering** — The approved contract is now persisted in Redis _before_ the `created → planning` state transition. Previously, a crash between the two writes would leave the run in `planning` with no stored contract, permanently blocking orphan recovery.
- **`_on_done` callback `CancelledError` crash** — `asyncio.Task.exception()` raises `CancelledError` on cancelled tasks. The callback now checks `t.cancelled()` first; cancelled tasks are logged and cleaned up without attempting to read the exception.
- **Agent Redis client cleanup on shutdown** — `close_async_redis_client()` is now called in the FastAPI lifespan shutdown block alongside `close_async_job_managers()`. Prevents event-loop contamination across successive `TestClient` uses in integration tests.
- **Falsy-dict treated as `None`** — `get_approved_contract` used `result if result else None`, which would return `None` for a stored empty dict `{}`. Changed to `result if result is not None else None`.
- **Silent `JSONDecodeError`** — Malformed JSON in the `approved_contract` or `proposed_contract` Redis fields now emits a `logger.warning` before returning `None`.
- **`text_threshold` now actually filters** (`GroundingDINODetector.detect()`). Prior to this release the parameter was silently discarded — a regression introduced during the inference-path cleanup. Token-level filtering is now applied: tokens whose sigmoided score is <= `text_threshold` are zeroed before the per-class mean is computed, diluting the class score when some tokens are sub-threshold.
- **NaN propagation guard** in `logits_to_class_scores`. When the masking multiply produces `NaN * 0.0` (IEEE 754 quirk), `torch.nan_to_num` now clamps those entries to 0.0, matching the intent of the mask.
- **NaN `text_threshold` raises immediately** instead of silently disabling filtering. `logits_to_class_scores` now raises `ValueError` if `text_threshold` is `float('nan')`.

### Added

- `StateMachine.store_approved_contract` / `get_approved_contract` — persist the approved contract in the run's Redis HASH so startup recovery can reconstruct the Coordinator after a container restart.
- `StateMachine.scan_non_terminal_run_ids` — SCAN-based classmethod that returns run IDs not in a terminal state; used by orphan recovery.
- `StateMachine.exists` — lightweight existence check without raising on missing keys.
- TODOs #21-24 filed: `pending_contract_approval` exit endpoint missing, `failed_retrying` retry dispatch missing, worker task self-termination stub, multi-instance Coordinator collision risk.

### Breaking changes

- **`detect()` default `text_threshold=0.5` now filters.** In v0.1.0 the default had no effect. Callers who relied on the v0.1.0 no-op behaviour (intentionally or not) will see fewer detections. To restore the old unfiltered behaviour, pass `text_threshold=0.0` explicitly.

### Security

- TODO comment added to `POST /api/agent/approve` and `POST /api/agent/gate/{rid}/{action}` noting that these endpoints have no authentication and should have an API-key guard before public exposure.

## [0.1.0] — initial release
