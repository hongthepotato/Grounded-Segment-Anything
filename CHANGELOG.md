# Changelog

All notable changes to this project will be documented in this file.

## [0.1.11] — 2026-05-06

### Fixed

- **`alpha_corf` typo corrected in `semi_transparent` low-intensity `RandomFog` rule** — `augmentation/characteristic_translator.py` line 301 had `"alpha_corf": RangeParameter.scalar(0.06)` where the key should be `"alpha_coef"` (consistent with the medium and high intensity entries at lines 314 and 327). The misspelling would silently pass an unrecognised parameter key to albumentations' `RandomFog` at inference time instead of the intended alpha coefficient.

### Tests

- **`TestCharacteristicSchemaIntegrity` expanded to all 9 characteristics** — Previously all four parametrized test methods (`test_all_three_intensities_present`, `test_every_intensity_has_at_least_one_transform`, `test_every_transform_has_probability`, `test_translate_with_each_intensity_succeeds`) spot-checked only 5 of the 9 characteristics in `CHARACTERISTIC_RULES`. The four uncovered characteristics (`changes_size`, `semi_transparent`, `similar_to_background`, `multiple_objects`) are now included. Test count: 20 → 36. The parametrize argument is now derived directly from `CharacteristicTranslator.CHARACTERISTIC_RULES.keys()` so new characteristics are automatically covered without a manual list update. GitHub issue #63. TODO #25.

## [0.1.10] — 2026-05-06

### Fixed

- **`AgentLoop` never self-terminated after run reached terminal state** — `AgentLoop.should_stop()` now overrides the `StreamConsumer` base-class stub (which always returned `False`) with the same pattern already used by `ExecutorWorker` and `EvaluatorWorker`: instantiate `StateMachine`, call `current_state()`, return `True` if the state is in `TERMINAL_STATES`, catch `KeyError` for uninitialized state and return `False`. Since `Coordinator.run()` waits on all three coroutines via `asyncio.gather`, an `AgentLoop` that never exited meant the gather never returned and the Coordinator task leaked for every completed run. The loop now exits cleanly on the next `should_stop()` check after the state transitions to terminal.

### Tests

- **5 new tests in `TestAgentLoopShouldStop`** — `test_returns_true_when_run_is_terminal`, `test_returns_true_for_all_four_terminal_states` (exhaustively covers `done`, `failed_unrecoverable`, `escalated`, `cancelled`), `test_returns_false_when_run_is_non_terminal`, `test_returns_false_when_state_key_absent` (verifies `KeyError` is caught and does not propagate), `test_run_exits_early_when_state_becomes_terminal` (integration: loop exits after event handler transitions state to terminal, second queued event is not processed). GitHub issue #55. TODO #23.

## [0.1.9] — 2026-05-05

### Changed

- **`save_merged_model` checkpoint format redesigned** — Framework integrity fields (`format`, `peft_merged`, `requires_peft`) are now stored exclusively in `checkpoint["metadata"]`. Caller-supplied training provenance (epochs, metrics, git SHA, etc.) is stored in a separate `checkpoint["training_info"]` top-level key, written only when the caller provides a non-empty dict. The two namespaces can no longer interfere — reserved key names in `training_info` are preserved intact rather than silently dropped. Parameter renamed from `extra_metadata` to `training_info` for honesty. Old checkpoints (no `training_info` key) load cleanly — `load_merged_model` reads only `checkpoint["metadata"]`.

### Tests

- **Merger test suite updated for new checkpoint layout** — `test_training_info_stored_at_separate_checkpoint_key` (renamed from `test_extra_metadata_merged_into_metadata_dict`) asserts that caller provenance lives in `checkpoint["training_info"]` and that framework fields are absent from it. `test_training_info_cannot_affect_metadata_even_with_reserved_key_names` (renamed from `test_extra_metadata_cannot_clobber_framework_defaults`) proves physical isolation: the same key name in `training_info` does not affect `metadata`. Added: absence assertion verifying `training_info` is not written when caller provides nothing.

## [0.1.8] — 2026-05-05

### Fixed

- **Schema validation gaps closed** — `ApiResponse.code` now enforces HTTP range (`ge=100, le=599`); `ApiResponse.status`, `JobCreate.job_type`, `AutoLabelRequest.output_mode`, and `WorkerResponse.status` are tightened from plain `str` to `Literal` types so invalid values are rejected at the boundary with a clear `ValidationError` instead of a downstream dispatch error. `AutoLabelRequest.image_paths` and `classes` gain `min_length=1` constraints.
- **`DistillationRequest` paired-field invariant enforced** — A new `_check_paired_fields` model validator ensures `teacher_dir` and `unlabeled_image_paths` are either both provided or both absent; previously only one could be set without raising.
- **Merger metadata merge order fixed** — `save_merged_model` previously applied `extra_metadata` after the framework integrity keys (`format`, `peft_merged`, `requires_peft`), allowing callers to silently overwrite them. Framework keys now overwrite caller keys so `load_merged_model`'s format-prefix check stays reliable. Deferred architectural redesign tracked in TODO #30.
- **`load_merged_model` raises clean errors on malformed checkpoints** — Missing `model_state_dict` key and non-dict `metadata` values now raise `RuntimeError` with a descriptive message instead of a bare `KeyError` or `AttributeError`.
- **`create_export_package` cleanup always runs** — The temp `package_dir` is now deleted in a `try/finally` block so it is cleaned up even when any of the four build steps raises. A secondary guard wraps the `rmtree` call to prevent cleanup failures from masking the original exception.
- **`_build_autolabeler_config` validates output mode upfront** — Unknown `output_mode` strings raise `ValueError` immediately instead of silently falling through. Boxes-only mode without a fine-tuned detector also raises early, symmetric with the existing segmenter check. Both detector and segmenter resolver-invariant violations now raise `RuntimeError` (replacing bare `assert` which is stripped under `-O`).

### Tests

- **45 xfail markers removed** — All 45 `@pytest.mark.xfail(strict=True)` markers across five test files have been resolved: the production code gaps they tracked are now fixed. Tests in `test_schemas.py`, `test_merger.py`, `test_packager.py`, `test_pseudo_label.py`, and `test_distillation_ros2_integration.py` now pass unconditionally. `strict=True` enforcement means any regression will be caught immediately.
- **F841 per-file-ignores removed from pyproject.toml** — Two suppression entries for unused-local-variable warnings (`F841`) in `test_distillation_ros2_integration.py` and `test_registry.py` are removed; the actual unused variables have been cleaned up.

## [0.1.7] — 2026-05-03

### Fixed

- **PEL replay wastes a retry slot after SIGKILL** — If the coordinator process was killed (OOM-kill, host crash) between `sm.transition("failed_retrying")` and the subsequent re-dispatch call, the Redis Stream PEL replay would present the event again with `current_state() == "failed_retrying"`. The old handler treated `"failed_retrying"` as the failed stage name and attempted an invalid self-arc transition, routing to `failed_unrecoverable` and burning a retry slot. The fix: `retry_work_stage` is now stored atomically in SM metadata alongside the `failed_retrying` transition; on replay the handler detects `current == "failed_retrying"`, reads the stage back from metadata, skips the budget charge, and converges at the shared re-dispatch path. Missing or corrupt metadata routes to `failed_unrecoverable` with an error log instead of silently swallowing the exception.

### Refactored

- **`_handle_job_failed` extracted from `on_event`** — The `job_failed` branch of `on_event` is now a dedicated `_handle_job_failed(event, sm, state, current)` method, keeping `on_event` readable and making the two-path logic (fresh retry vs. PEL replay) explicit.

### Tests

- **14 new tests in `TestRetryDispatch`** — cover the PEL replay path: correct stage recovery (not `"failed_retrying"`), retry-count not double-charged, budget double-charge regression guard (max_retries=2 boundary), missing/empty/corrupt/wrong-key metadata → `failed_unrecoverable`, dispatch raises → `failed_unrecoverable`, SM re-entry transition raises → `failed_unrecoverable`, all three retryable stages (`auto_labeling`, `teacher_training`, `student_distillation`), override forwarding, and no-LLM-call invariant. TODO #27.
- **Fix async/dependency mocking in three ros2-integration test files** — `test_deploy_api.py` was patching the sync `get_job_manager` instead of the FastAPI `Depends(get_manager)` dependency, causing `_validate_completed_job` to hit a real Redis and hang in CI. `test_distillation_ros2_integration.py` was calling `_complete_job(job, str(path))` but the method now expects a `SubprocessResult` object. `test_yolo_node.py` had `len(boxes) == 0` by default (MagicMock), so no detections were ever mapped in `test_single_box_mapped_correctly`. All 21 tests in these files now pass cleanly.

## [0.1.6] — 2026-05-02

### Fixed

- **`failed_retrying` stuck state** — When a job fails with retries remaining, `on_event` previously transitioned to `failed_retrying` and returned, leaving the run stranded with no active job and no event to wake the Coordinator. The handler now captures `failed_stage = current` before transitioning away, then executes the sequence `failed_stage → failed_retrying → failed_stage` and calls `DispatchStageTool.execute()` directly to re-enqueue the job. Dispatch failure or SM-transition failure both route to `failed_unrecoverable`. Original LLM-chosen `stage_dispatch_overrides` from `LoopState` are forwarded verbatim on retry.

### Tests

- **14 new tests in `TestRetryDispatch`** — cover all three retryable stages, exact stage-name forwarding, override propagation (including empty-dict-not-None boundary), retry_count bookkeeping across two consecutive retries, exhausted-budget boundary (last allowed retry still dispatches), dispatch failure → `failed_unrecoverable`, error_message persistence, and no-LLM-call invariant. GitHub issue #54.

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
