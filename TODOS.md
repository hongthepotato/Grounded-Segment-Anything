# TODOs

Deferred work items with enough context to pick up later. Keep entries self-contained — don't assume the reader has the conversation in which they were captured.

---

## Agent Coordinator — lifecycle durability

Three related design gaps surfaced during the first `/plan → /approve` integration attempt on branch `agentic`. All three contribute to the same symptom class: a run gets stuck in a non-terminal state with no way for the system to recover on its own. See the RCA for run `5327209d-af51-47bc-8179-37f22786383f` (2026-04-21) for a concrete trace.

### 1. Auto-resume orphaned Coordinator tasks on FastAPI startup

**What:** On FastAPI app startup, scan Redis for runs in a non-terminal state and re-launch their Coordinator `asyncio.Task`. Today, Coordinators are only launched from `POST /api/agent/approve` ([api/routes/agent.py:68-103](api/routes/agent.py#L68-L103)).

**Why:** The Coordinator lives as an in-memory asyncio task in the FastAPI process. Any container restart kills the task silently. Redis state and the event stream are persistent, so the run's data is intact, but nothing is consuming the stream — the state is orphaned. The user has no way to recover short of manual Redis edits.

**Pros:**
- Restarts (deploys, crashes, `docker compose up` during dev) stop being data-loss events.
- Prerequisite for any multi-instance deployment.
- Removes a class of "stuck at `planning`" bug reports.

**Cons:**
- Need to handle leader election or single-writer guarantee if >1 FastAPI replica runs. For current single-instance dev, trivial; for prod, nontrivial.
- Startup scan adds latency and Redis load proportional to active-run count. Probably fine (expect <100 active runs).
- Resumed Coordinator must tolerate re-reading events it already processed — need to verify `stream_consumer` uses a durable `last_id` per run, not `"0-0"` from scratch.

**Context:** `_start_coordinator` at [api/routes/agent.py:68](api/routes/agent.py#L68) is the launch point today. Look at `StateMachine` + `TERMINAL_STATES` in [ml_engine/agent/state_machine.py](ml_engine/agent/state_machine.py#L59) for the "non-terminal" set. FastAPI startup hook goes in the app factory (search for `app = FastAPI(` in `api/`). Keep it idempotent — `_start_coordinator` already no-ops if a task exists.

**Depends on / blocked by:** Verifying `stream_consumer` resume semantics first. If consumer starts from `"0-0"` on every Coordinator launch, re-running past events is a correctness risk and must be fixed before auto-resume ships.

---

### 2. Transition state to `failed_unrecoverable` when Coordinator task crashes

**What:** In the `_on_done` callback at [api/routes/agent.py:93-99](api/routes/agent.py#L93-L99), when the Coordinator task raises, transition the run's state to `failed_unrecoverable` with the exception message stored in `error_message`, in addition to the existing `logger.error(...)`.

**Why:** Today, a Coordinator crash logs and vanishes. The run stays in whatever state it was in when the task started (typically `planning`). `/status` returns a healthy-looking response with `coordinator_active: false` and no error context — indistinguishable from "running normally." The only visible symptom is that subsequent `/approve` calls return a cryptic state-transition 400.

**Pros:**
- Crashes become visible via `/status` and the event stream.
- Frontend can show a clear failure state instead of a hung spinner.
- Terminal state prevents nonsense retry attempts.

**Cons:**
- `_on_done` runs in an async context but isn't itself async — need to schedule the state transition (e.g. `asyncio.create_task(...)`) since `_on_done` is a sync callback.
- "Crash" is not always unrecoverable; a transient import error is different from a logic bug. Consider `failed_retrying` for some cases. Start simple (always `failed_unrecoverable`), refine later.

**Context:** Current `_on_done` at [api/routes/agent.py:93-99](api/routes/agent.py#L93-L99). `StateMachine.transition` accepts `error_message` as a kwarg, see [ml_engine/agent/state_machine.py:122-139](ml_engine/agent/state_machine.py#L122-L139). Allowed transitions from `planning` include `failed_unrecoverable` ([state_machine.py:69](ml_engine/agent/state_machine.py#L69)). From other states, check `TRANSITIONS` — failed exits may not be reachable from every state and will need adding.

**Depends on / blocked by:** None. Smallest fix of the three; can ship independently.

---

### 3. Make `POST /api/agent/approve` idempotent

**What:** When `/approve` is called on a run whose state is already past `created` (i.e. already approved), return a non-error response that either re-spawns the Coordinator task (if absent) or returns 409 with a clear "already approved" message. Don't return the current 400 with an internal state-transition error.

**Why:** The current behavior leaks internals: the client gets `"Invalid transition 'planning' -> 'planning'. Allowed: ['pending_contract_approval', 'failed_unrecoverable']"`. That's a state-machine implementation detail with no actionable information for a caller who just wants to retry after a network hiccup or a deploy. Combined with #1 (no auto-resume), this is the worst case: the run IS stuck, and the retry path is blocked too.

**Pros:**
- Retries after transient failures (timeouts, container restart, network blips) become safe.
- Pairs well with #1 — a UI that polls `/status` can safely re-issue `/approve` to resume.
- Consistent with REST idempotency expectations for non-POST-like POSTs.

**Cons:**
- Need to decide idempotent semantics precisely: (a) silently re-spawn the task if absent, or (b) 409 with `{run_id, state, note}`. Behaviors are different — (a) is friendlier, (b) is stricter. Recommend (a) for dev ergonomics.
- Slightly complicates the endpoint — need to branch on current state before transitioning.

**Context:** `/approve` at [api/routes/agent.py:178-215](api/routes/agent.py#L178-L215). Current flow: unconditional `sm.transition("planning")` then `_start_coordinator`. Replace with: read current state first; if `created`, transition; if `planning` or beyond and not terminal, skip transition and still call `_start_coordinator` (which is already idempotent); if terminal, return 409.

**Depends on / blocked by:** Best done together with #1, because re-spawning the Coordinator only makes sense once auto-resume infrastructure exists to verify the Task wasn't already running. Ship #2 first for observability, then #1 + #3 together.

---

## CI / Testing — follow-ups from ci-and-tests PR

Six items deferred during the `/plan-eng-review` of the CI/test infrastructure PR. Filed 2026-04-23. All depend on the `ci-and-tests` branch merging first.

### 5. Dependabot config for Python + GitHub Actions updates

**Status:** 🟡 PR open on `chore/dependabot` (2026-04-24). Scope expanded in-flight to include the Dockerfile `TORCH_CUDA_ARCH_LIST` broad-arch expansion — one coordinated PR covers both "stay current on deps" and "run on all NVIDIA generations."

**What:** Add `.github/dependabot.yml` with weekly schedule for `pip` (uv ecosystem via pyproject.toml) and `github-actions` ecosystems.

**Why:** Unpatched ML deps are a supply-chain risk vector: pickle deserialization bugs, pydantic validation bypasses, transformers tokenizer flaws. Automated weekly update PRs exercise CI automatically.

**Pros:** No manual upkeep on dep updates. PRs auto-verified by CI. Pairs well with CI-green-required branch protection.

**Cons:** PR noise (5-10 PRs/week for an ML project). May need grouping rules (`groups:` syntax) to collapse noisy ecosystems. Some PRs need judgment calls that Dependabot can't make.

**Context:** The `ci-and-tests` PR ships with CI that will auto-run on any Dependabot PR. This just turns on the firehose.

**Depends on / blocked by:** `ci-and-tests` PR merging. Optional: triage rules for high-frequency deps (transformers, torch).

---

### 6. Typed mypy overrides — remove `--ignore-missing-imports`

**What:** Add `[[tool.mypy.overrides]]` blocks in `pyproject.toml` for untyped third-party deps (`groundingdino.*`, `segment_anything.*`, `peft.*`, `transformers.*`). Remove the workflow-level `--ignore-missing-imports` flag and `continue-on-error: true` from the mypy step.

**Why:** Current CI runs mypy lax during the 2-week ramp-up to avoid blocking on volume of existing errors. After ramp, tighten so mypy actually gates the build.

**Pros:** Real type checking. Catches Optional[X] vs X bugs, wrong-shape dict returns, return-type drift. High value in ML code where tensor shapes rarely match docstrings.

**Cons:** Will surface existing type violations that need cleanup. Unclear volume until lax mypy has run for ~2 weeks and the error count is measurable.

**Context:** Workflow env currently sets `--ignore-missing-imports` globally. Replace with narrow per-package overrides. Start with untyped-but-stable packages (groundingdino, segment_anything); leave actively-upgrading packages (transformers) behind `ignore_missing_imports = true` for longer.

**Depends on / blocked by:** 2 weeks of running CI with lax mypy to establish baseline error count.

---

### 7. Venv-sharing CI optimization (prep job + artifact)

**What:** Replace per-job `uv sync --frozen --extra test` with a single prep job that installs once and uploads `.venv/` as a workflow artifact. Downstream jobs (`unit`, `contract`, `coverage-gate`) download and `source .venv/bin/activate`.

**Why:** Current design runs `uv sync` in 3 jobs independently. On warm cache each takes ~30s, but cold cache adds ~2 min/job. Saves ~2 min per CI run. Compound savings at scale: 20 runs/day × 1.5 min × 250 days ≈ 125 hours/year.

**Pros:** Faster CI runs. Slightly lower GitHub Actions minute consumption.

**Cons:** +1 job in the workflow. Artifact upload (~500MB compressed) + download latency (~20s each direction) eats ~40s of the savings. More YAML to maintain.

**Context:** Tagged as "may denote as extension" by user during `/plan-eng-review`. Explicitly deferred from the initial ci-and-tests PR to keep the shipping shape simple. Revisit when PR volume makes the savings compelling (>15 PRs/week).

**Depends on / blocked by:** `ci-and-tests` PR merging. No hard dependency.

---

### 8. Self-hosted GPU runner for real CUDA kernel verification

**What:** Register the team's 4× RTX 4090 dev box as a GitHub Actions self-hosted runner. Add a `runs-on: [self-hosted, gpu]` job matrix that runs GPU-marked tests (currently auto-skipped via `@pytest.mark.gpu` hook).

**Why:** Unlocks real CUDA kernel correctness testing in CI. No more "compile-only" safety via nightly Docker smoke. Catches bugs that only manifest on GPU (NaN gradients under real autocast, kernel launch failures, CUDA OOM patterns).

**Pros:** Full CI coverage of the GPU code paths. Elimination of the `@pytest.mark.gpu` skip gap that currently loses test value in CI.

**Cons (security is the blocker):** Repo appears public (`github.com/hongthepotato/Grounded-Segment-Anything`). Self-hosted runners on public repos mean any PR — including from strangers — can execute on the registered machine. Need `approve-only-for-external-contributors` workflow trigger setting + careful review of every external PR. Also: runner must be online when PRs arrive, adds maintenance.

**Context:** GitHub Actions has two runner types. GitHub-hosted (free VMs, no GPU) is what ci-and-tests PR uses. Self-hosted (your hardware, whatever specs) can serve GPU jobs via runner labels. Setup: install actions-runner on the box, register with repo-scoped token, tag with `self-hosted, gpu, linux`. Then workflows opt in via `runs-on`.

**Depends on / blocked by:** Security hardening FIRST. Set `Require approval for first-time contributors` in repo settings. Alternative path: keep self-hosted runner on a private fork, only run GPU CI on branches pushed by maintainers.

---

### 9. Clean up ruff baseline (3593 findings at ci-and-tests merge time)

**What:** Work through the ruff baseline in staged PRs. Measured at ci-and-tests merge: 1486 `W293` (blank-line-whitespace), 1027 `E501` (line-too-long), 367 `I001` (unsorted-imports), 291 `F401` (unused-import), 183 `W291` (trailing-whitespace), 34 `F841` (unused-variable), 31 `F821` (undefined-name) — the last one may include real bugs worth investigating. First pass: `uv run ruff check . --fix` auto-fixes ~1732 of these. Second pass: review and fix the non-auto-fixable ones.

**Why:** CI currently runs ruff with `continue-on-error: true` — findings are reported but do not gate PRs. This defeats the point of a lint gate long-term. Also: `F821` (31 undefined-name errors) may flag real bugs in existing code.

**Pros:** Real lint gate. Catches new drift before it lands. Uncovers F821 bugs that may be lurking.

**Cons:** Large diff (~2000 lines touched by auto-fix alone). Reviewer fatigue if done in one PR. Best done per-directory in sequence (start with `core/`, then `api/`, then `ml_engine/`).

**Context:** Ruff was configured in `pyproject.toml` from the project's first commit but never wired into CI or run manually. ci-and-tests PR wires it up but makes it non-blocking to ship the CI infrastructure without coupling to a cleanup effort.

**Depends on / blocked by:** `ci-and-tests` PR merging. Each cleanup PR should remove `continue-on-error: true` from the affected file scope once that directory's findings reach zero.

---

### 11. Inference module integration tests (detectors, segmenters, auto_labeler, visualizer)

**What:** Add integration tests under `tests/integration/` against tiny real SAM/GroundingDINO weights for `ml_engine/inference/detectors/*`, `ml_engine/inference/segmenters/*`, `ml_engine/inference/auto_labeler.py`, `ml_engine/inference/visualizer.py`. Use a minimal checkpoint (e.g., `SAM-small` variant or custom-trained tiny weights) hosted as a CI fixture.

**Why:** These modules are currently untested at the unit level because they are thin wrappers around real models (plan explicitly out-of-scope'd them). But thin wrappers still have bugs — dtype conversions, preprocessing shape assumptions, output postprocessing correctness. Integration test against a tiny real model catches wrapper-layer bugs without needing GPU.

**Pros:** Closes the last untested-module gap in the CI-and-tests PR's scope. Catches regressions that only manifest when a real model is in the loop.

**Cons:** Needs a tiny-but-representative checkpoint. Self-training one is ~1 day of work; downloading someone else's adds a supply-chain dependency. Tests are slower than pure-unit (several seconds per test vs milliseconds).

**Context:** Explicitly out-of-scope in ci-and-tests PR per design doc. Marker: `@pytest.mark.integration` + `@pytest.mark.slow`. Lives in `tests/integration/test_inference_pipelines.py`. CI coverage by nightly workflow only (too slow for PR CI).

**Depends on / blocked by:** Choosing/producing a tiny representative checkpoint. `ci-and-tests` PR merging (for the integration test infrastructure it adds).

---

### 12. Priority-2 unit test roster — 6 remaining files from ci-and-tests design doc

**What:** Add six unit test files that were listed in the `ci-and-tests` design doc's Priority-2 roster ("include if time permits") but were not shipped in the PR and are not tracked elsewhere. Per-file targets (all target modules verified present on `agentic` at PR-close time):

- `tests/unit/ml_engine/export/test_merger.py` — covers `ml_engine/export/merger.py` (~150 lines). Compatible artifacts merge, incompatible artifacts reject, metadata preservation.
- `tests/unit/ml_engine/export/test_packager.py` — covers `ml_engine/export/packager.py` (~185 lines). Packaging output format, file inclusion/exclusion rules, filename sanitization.
- `tests/unit/ml_engine/distillation/test_pseudo_label.py` — covers `ml_engine/distillation/pseudo_label.py` (~141 lines). Pseudo-label generation on synthetic teacher outputs: confidence thresholds, empty-prediction handling, label-format correctness.
- `tests/unit/augmentation/test_parameter_system.py` — covers `augmentation/parameter_system.py` (~165 lines). Parameter parsing + validation: out-of-range values, type coercion, defaults.
- `tests/unit/augmentation/test_characteristic_translator.py` — covers `augmentation/characteristic_translator.py` (~1362 lines — biggest target; triage to highest-value code paths first, don't boil this sub-ocean in one PR). Translation correctness between characteristic spec and concrete transform params.
- `tests/unit/api/test_schemas.py` — covers `api/schemas.py` (~382 lines). API pydantic schemas: request/response envelope structure, required-field enforcement, validation-error shape.

**Why:** The design doc (`~/.gstack/projects/hongthepotato-Grounded-Segment-Anything/shen_h-agentic-design-20260422-223526.md` lines 204-212) explicitly listed these as "include if time permits, follow-up PR otherwise." They were deferred, and without a tracking entry the promise was about to fall off the radar. Captured here so the bookkeeping survives the merge. The seventh P2 file (`test_augmentation_factory.py`) was covered by item 10 which has now shipped (see Completed section below).

**Pros:**
- Raises unit coverage above the 52% baseline the ratchet starts at, giving real headroom toward the 70% design target.
- Each module is a thin-ish, pure-Python unit target — no CUDA, no model weights, no network. Fast tests.
- Six small focused PRs (or two grouped PRs by domain: `export+distillation`, `augmentation+api`) is lower reviewer fatigue than one combined PR.

**Cons:**
- Six files, even at Priority-2 scope, is 300-600 lines of test code. Spread across multiple PRs to keep reviews tight.
- `test_characteristic_translator.py` targets a 1362-line module and could expand scope indefinitely — cap at highest-value paths for the first pass (factory dispatch + the 5 most-used characteristics), add a TODO for the rest.

**Context:** Deferred from the original ci-and-tests PR (~3400 lines new) to keep that PR shippable. PR is CI-green at 52% combined coverage, merged with COVERAGE_MIN ratchet seeded at 50. Each new test file from this roster bumps real coverage and unlocks a COVERAGE_MIN bump in the same PR.

**Depends on / blocked by:** `ci-and-tests` PR merging (for the test infrastructure and markers). No other blockers. Recommended sequencing: look at `tests/unit/augmentation/test_augmentation_factory.py` (shipped in item 10) for the pytest-parametrize + class-grouping pattern when adding the augmentation/ tests in this item.

---

## Completed

Historical record of items that have shipped. Kept rather than deleted so external
references (commit messages, PR descriptions mentioning "item N") resolve.

### 4. Pre-commit hooks ✅

**Completed:** 2026-04-24 via PR #26 merged into `agentic`.

**What shipped:** `.pre-commit-config.yaml` with file-hygiene hooks (trailing
whitespace, EOF newline, YAML/TOML validity, large-file cap, merge conflict
markers, private-key detection, debug-statements, Python AST validity) plus
`ruff check`, `ruff format --check`, and `mypy` via `uv run --no-sync` so the
hook binaries match CI exactly. Added `pre-commit>=3.6.0` to the `dev` extra.
Onboarding: `uv sync --extra dev && uv run pre-commit install`.

**Design notes worth remembering:** `--fix` deliberately NOT enabled for ruff
because the ~3500-finding baseline would produce a destructive first-run diff.
Per-file cleanup happens naturally as pre-commit flags new violations when
you touch a file — see item 9.

### 10. Error-path coverage for `augmentation_factory._validate_bboxes` ✅

**Completed:** 2026-04-24 in the same PR as this Completed-section addition.

**What shipped:** `tests/unit/augmentation/test_augmentation_factory.py` — 44
parametrized tests across 7 test classes, one per error class:
`TestValidInputs` (5), `TestInvalidContainer` (5), `TestInvalidBboxElement`
(8), `TestInvalidCoordinateTypes` (8), `TestInvalidDimensions` (6),
`TestOutOfBounds` (8), `TestErrorIndexing` (1). Every branch of the COCO
validator exits under test: type mismatches raise `TypeError`, value/bounds
mismatches raise `ValueError`, and the reported bbox index is 1-indexed.

**Design notes worth remembering:** validator is called with `self=None`
because its body uses no instance state — `self` is only there because it's
an instance method. Avoids constructing a full augmentation pipeline for
every test. Fast (~8s cold for all 44 tests), isolated from albumentations
state.

### 13. Type-annotate `ml_engine/export/merger.py` ✅

**Completed:** 2026-04-25 on `chore/mypy-merger-hygiene`, stacked on `ci/gate-ruff-checks`.

**What shipped:** Three pre-existing mypy errors in `ml_engine/export/merger.py`
fixed without `# type: ignore` band-aids:
- Lines 46 + 54: `peft_model.merge_and_unload()` was flagged `"Tensor" not callable`
  because `nn.Module.__getattr__` is typed `Tensor | Module` and mypy can't see
  through PEFT's runtime delegation. Fix: rebind via `peft_model: Any = model.model`
  (and `direct_peft: Any = model` on the direct branch) — local `Any` annotation
  signals "trust the duck-typed runtime contract here, mypy can't help."
- Line 105: `checkpoint["metadata"].update(extra_metadata)` was flagged because
  mypy widened the dict literal to `Collection[Any]`. Fix: explicit
  `checkpoint: Dict[str, Any] = {...}` annotation prevents the inference widening.

**Why `Any` and NOT a Protocol:** the original sketch in this TODO proposed a
`@runtime_checkable Protocol`. Investigation showed PEFT ships `py.typed` but
`PeftModel.merge_and_unload` is exposed via `__getattr__` delegation to
`self.base_model` (a `LoraModel`). mypy fundamentally cannot follow `__getattr__`
for arbitrary attribute names — so even `from peft import PeftModel` and using
`isinstance(x, PeftModel)` would NOT have helped. The Protocol approach would
have worked but added 15 lines of new abstraction for a 3-line type fix. `Any`
is honest about what's actually true: mypy can't help here, the runtime
hasattr() checks ARE the contract.

**The SKIP=mypy precedent:** This TODO existed because `SKIP=mypy git commit`
became habitual across 5 PRs. With merger.py now mypy-clean, the bypass should
no longer be needed for ml_engine/export/. (Other dirs still have baseline mypy
errors covered by item 6 — broader cleanup.)

**Design notes worth remembering:** before assuming a 3rd-party library's typing
will solve a static-checking problem, verify it's not using `__getattr__`-based
delegation. Many ML libraries (PEFT, transformers, accelerate) lean on dynamic
composition and have similar gaps. `Any` at the boundary is a legitimate fix
when the dynamic surface can't be statically modeled.
