# 6. mypy baseline cleanup — drive 416 errors to zero, then flip the gate

**Status:** Path A chosen 2026-04-25 (full strict across maintained dirs).
Fully complete 2026-04-26.

**Final baseline:** 0 errors across 119 source files (`core ml_engine api
augmentation`). Mypy now gates merges (PR #40, `ci/gate-mypy-checks`) the
same way ruff does (TODO #9 / PR #33 set the precedent).

## PR chain (9 PRs total)

- PR #34 (`chore/mypy-stubs-and-todos`): Step 1 — `types-PyYAML` stub install (416 → 410)
- PR #35 (`mypy-baseline-core`): Step 2.1 — core/ clean (410 → 400)
- PR #37 (`mypy-baseline-api`): Step 2.2 — api/ clean (400 → 388)
- PR #38 (`mypy-baseline-augmentation`): Step 2.3 — augmentation/ clean (388 → 219)
- PR #39 (`mypy-baseline-ml-engine`): Step 2.4 — ml_engine/ clean across 8 sub-commits (219 → 0)
- PR #40 (`ci/gate-mypy-checks`): Step 3 — flipped `continue-on-error: false` + added augmentation/ to scope

## Per-step progress

- **Step 1**: installed `types-PyYAML` (cleared 6 yaml errors). Baseline 416 → 410.
- **Step 2.1**: `core/` clean — 4 functions in `config.py` widened `path: str` → `path: str | Path`; 2 formatter locals in `logging_config.py` annotated as base `logging.Formatter`. Baseline 410 → 400.
- **Step 2.2**: `api/` clean — 13 own errors fixed. Patterns: 4 `Path(job.output_dir)` calls now guarded with explicit None checks (HTTPException 500 if missing); 3 var-annotated dict/list literals; 1 `job_to_response(Job | None)` guarded; deleted `autolabel.py`'s stale duplicate of `job_to_response` (canonical lives in `jobs.py`); replaced broken `subscribe_to_job_async` call in `websocket.py` with explicit `subscription_unavailable` frame + clean close (filed item 15 for proper implementation). Baseline 400 → 388.
- **Step 2.3**: `augmentation/` clean — 169 own errors fixed. Two correctness fixes: (a) `AugmentationRule.intensity_ranges` annotation was off by one nesting level (`Dict[str, Dict[str, AlbumentationsParameter]]` → `Dict[str, Dict[str, Dict[str, Any]]]`) — that single fix cleared 153 of the 169 errors; (b) `_validate_input_data` + 3 `_validate_{masks,keypoints,bboxes}` helpers now take `Optional[List[...]]` matching what `apply()` actually passes in. Plus an `apply()` refactor: kept `has_X` booleans (used downstream) but added parallel inline `is not None and len(...) > 0` checks so mypy can narrow the Optional types (booleans aren't type guards). Annotated `aug_input: Dict[str, Any]` so mypy doesn't narrow from the first ndarray entry. Baseline 388 → 219.
- **Step 2.4**: `ml_engine/` clean — 219 own errors fixed across 8 sub-PRs on `mypy-baseline-ml-engine`:
  - **2.4.1 data/** (58 errors → 0): `Dict[str, Any]` annotations on heterogeneous validator-result dicts (one annotation cleared 25+ "object" errors), `.get()` → `[]` for required COCO keys (failing loud on malformed data), preprocessor `self.model: Any` boundary cast, real bug fixed: `scaleFill=False` → `scale_fill=False` (ultralytics renamed kwarg, `YOLOPreprocessor` has zero unit-test coverage so the runtime crash was masked).
  - **2.4.2 jobs/** (53 errors → 0): redis-py `Awaitable[T] | T` overload artifact via `self.redis: Any`, `multiprocessing.Event` factory-vs-class confusion via `from multiprocessing.synchronize import Event as MpEvent`, `asyncio.gather` heterogeneous-type narrowing by pulling the only non-int call out of gather.
  - **2.4.3 training/** (49 errors → 0): `self.model: Any` + `self.criterion: Any` in `model_trainers/base.py` (boundary cast covered all PEFT-attribute access in subclasses), `torch.amp.GradScaler` (canonical) + `LRScheduler` (public) name updates, **structural simplification**: dropped `GroundingDINOCriterion.losses` list + `get_loss()` dispatch (unused flexibility — only caller hardcoded ["labels", "boxes"]), inlined both loss calls in `forward()`. Surfaced TODO #16: `CreateByInfo.job_id` and `BundleManifest.lineage` had to be widened to `Optional[str]` because the trainer can't see its parent job_id — value is known at submission time but dropped at subprocess boundary.
  - **2.4.4 agent/** (16 errors → 0): same redis `self._r: Any` workaround across `state_machine`, `memory`, and `stream_consumer` (the StreamConsumer fix covers all subclasses via inheritance), `text_parts` for-loop variable shadowing fixed by renaming to `user_text_parts` / `assistant_text_parts`.
  - **2.4.5 evaluation/** (24 errors → 0): real bug fixed in `visualizer.py` — `else []` fallback returned Python list to a function typed `np.ndarray`, replaced with `np.array([])` so the type matches the contract; `peft_model: Any = model` boundary cast for `model.predict()`; `int()` cast for `Dict[int, list[float]]` indexing on `Tensor.item()` returns; lazy-import `self._plt: Any` pattern.
  - **2.4.6 models/teacher/** (23 errors → 0): `self.model: Any` boundary cast in both `GroundingDINOLoRA` and `SAMHQLoRA` (16 PEFT attribute-access errors collapsed); `_get_base_model() -> Any` cascaded through `_get_image_encoder/_prompt_encoder/_mask_decoder` so SAM-specific method calls (`prompt_encoder.get_dense_pe()`) type-check.
  - **2.4.7 inference/** (22 errors → 0): real bug filed as TODO #17 — `text_threshold` flowed from API → handler → DetectionThresholds → AutoLabeler and was silently dropped at `detector.detect()` (the original token-level filter was lost in refactor); restored as accepted-but-unused param with `_ = text_threshold` placeholder + restored `text_threshold=self.config.thresholds.text` in AutoLabeler. Lazy-init `self._model: Any = None` pattern across 3 segmenter/detector files.
  - **2.4.8 distillation/ + experiment/ + export/templates/** (12 errors → 0): same `MpEvent` fix in `trial_runner.py`; explicit None re-narrowing in `mutators.py` all() generator (mypy can't propagate filter narrowing into nested generators); `assert manifest is not None` after the resolver-invariant guard in `pseudo_label.py`; fallback to spec class-level defaults when manifest fields are Optional.
- **Step 3**: removed `continue-on-error: true` from ci.yml's mypy step + added `augmentation/` to the gated source set (was only `core ml_engine api` previously). Same pattern as the ruff gate flip in PR #33.

## Real bugs surfaced and fixed during cleanup

- ultralytics `scaleFill` → `scale_fill` rename had broken `YOLOPreprocessor` at construction time; zero unit-test coverage masked the runtime crash (Step 2.4.1)
- `PredictionVisualizer._save_single` `else []` fallback returned a Python list to a function typed `np.ndarray` — silent contract violation (Step 2.4.5)
- `text_threshold` parameter flowed from public API down to `detector.detect()` and was silently dropped — filed as TODO #17 (Step 2.4.7)

## Real bugs surfaced and DEFERRED (filed as separate TODOs)

- TODO #16: `CreateByInfo.job_id` widened to `Optional[str]` because the trainer can't see its parent job_id — value is known at submission time but dropped at the subprocess boundary. (Subsequently shipped — see [16-job-id-lineage-plumbing.md](16-job-id-lineage-plumbing.md).)
- TODO #17: token-level filtering removed from GroundingDINODetector during a refactor; user-supplied `text_threshold` is currently a no-op.

## Patterns established (grep-able, all cross-reference each other)

- `self.X: Any` boundary cast for PEFT-via-`nn.Module.__getattr__` (12+ sites)
- `self.redis: Any` for redis-py's `Awaitable[T] | T` overload artifact (5 sites)
- `from multiprocessing.synchronize import Event as MpEvent` for `mp.Event` factory-vs-class confusion (8 files)
- Path/str shadowing → fresh `path: Path` rebind (5 sites)
- Lazy-init `self._X: Any = None` for modules/predictors set in `_load_model()` (4 sites)
- `Dict[str, Any]` annotation on heterogeneous result/result-like dicts (6+ sites — covers ~30 cascading "object has no attribute" errors)

## Lessons learned

- The original Step-1 estimate of "100-150 errors from missing stubs" was wildly optimistic. Reality: 6. The `--ignore-missing-imports` flag turns out to be a sledgehammer that hides most stub gaps, so removing it would expose them — but removing it is exactly what Step 3 does.
- Per-case judgment patterns that worked across all steps:
  - Path-as-string parameter pattern → widen to `str | Path` (or rebind to fresh `path: Path`)
  - Subclass-narrowing pattern → annotate as base type
  - Variable shadowing → rename or refactor
  - Truly intractable cases (PEFT-like `__getattr__` delegation) → `Any` cast at boundary, with comment

## Three pragmatic alternatives that were considered (and rejected)

1. **Full strict (the path taken)** — multi-week effort, ongoing per-PR cost to maintain types. Worth it because type bugs WERE biting in production (3 real bugs surfaced and fixed during cleanup).
2. **Pragmatic non-gating** — keep mypy advisory forever, fix specific high-value modules opportunistically. Costs nothing now; defers all benefits. Status quo.
3. **Strict only on critical modules** — flip mypy to gating ONLY for `core/` and `api/schemas.py`. ~1 PR of work; captures most of the bug-prevention value with minimal cleanup overhead. Rationale: type discipline matters most where the wire format is defined and other modules import from. Rejected because (a) the surfaced production bugs were in `ml_engine/` not `core/`, validating the broader effort, and (b) once `core/` is strict, drift between strict/non-strict modules creates ongoing cognitive load.

## Honest scope estimate (kept for posterity)

The original estimate was ~2-3 weeks of focused work, broken down as:
- ~100-150 errors are quick wins (install stubs, add obvious annotations) — ~1 hour
- ~150-200 are mechanical per-case (variable shadowing, simple Optional, missing return types) — half-day per directory
- ~50-100 require real semantic work (Optional refactor, type-narrowing assertions) — case-by-case

Actual: shipped in ~2 days of focused work across the 9 PRs above. The
estimate was too pessimistic on the "quick wins" bucket (most "missing
stubs" errors didn't actually exist due to `--ignore-missing-imports`),
and too pessimistic on the "real semantic work" bucket (most of the hard
cases collapsed into the boundary-`Any` pattern once it was established).

## Why this is NOT a single PR (kept for posterity)

Unlike the ruff baseline (mostly mechanical whitespace + import order, ~90%
auto-fixable), mypy errors require per-case judgment. Each one is a real
type ambiguity:

| Error class | Example | Effort per fix |
|---|---|---|
| Missing 3rd-party stubs | `Library stubs not installed for "yaml"` | 1 line per dep — install `types-PyYAML` etc. |
| Missing variable annotations | `Need type annotation for "annotation_counts"` | 1 line per variable |
| `Optional` mishandling | `Argument 1 to "Path" has incompatible type "str \| None"` | Per-case judgment (assert non-None? refactor signature? change caller?) |
| Variable shadowing | `path: str = ...; path = Path(path)` rebinds incompatibly | Rename or refactor |
| Subclass narrowing | `formatter: ColoredTextFormatter = TextFormatter(...)` | Often signals real type design issues |
| Library `__getattr__` (PEFT pattern) | What item 13 fixed in `merger.py` | Case-by-case `Any` cast |
