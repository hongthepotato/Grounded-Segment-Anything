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

### 6. mypy baseline cleanup — drive 416 errors to zero, then flip the gate

**Status:** Path A chosen 2026-04-25 — committed to driving the baseline to zero across all maintained directories.

**Live baseline:** **219 errors in 42 files** (after Step 1 stubs + Step 2.1 core/ + Step 2.2 api/ + Step 2.3 augmentation/ — see progress below).

`continue-on-error: true` on the mypy step in `ci.yml` keeps CI passing while this baseline exists. Step 3 flips that off after Step 2 finishes.

**Progress:**
- ✅ Step 1: installed `types-PyYAML` (cleared 6 yaml errors). Baseline 416 → 410.
- ✅ Step 2.1: `core/` clean — 4 functions in `config.py` widened `path: str` → `path: str | Path`; 2 formatter locals in `logging_config.py` annotated as base `logging.Formatter`. Baseline 410 → 400.
- ✅ Step 2.2: `api/` clean — 13 own errors fixed. Patterns: 4 `Path(job.output_dir)` calls now guarded with explicit None checks (HTTPException 500 if missing); 3 var-annotated dict/list literals; 1 `job_to_response(Job | None)` guarded; deleted `autolabel.py`'s stale duplicate of `job_to_response` (canonical lives in `jobs.py`); replaced broken `subscribe_to_job_async` call in `websocket.py` with explicit `subscription_unavailable` frame + clean close (filed item 15 for proper implementation). Baseline 400 → 388.
- ✅ Step 2.3: `augmentation/` clean — 169 own errors fixed. Two correctness fixes: (a) `AugmentationRule.intensity_ranges` annotation was off by one nesting level (`Dict[str, Dict[str, AlbumentationsParameter]]` → `Dict[str, Dict[str, Dict[str, Any]]]`) — that single fix cleared 153 of the 169 errors; (b) `_validate_input_data` + 3 `_validate_{masks,keypoints,bboxes}` helpers now take `Optional[List[...]]` matching what `apply()` actually passes in. Plus an `apply()` refactor: kept `has_X` booleans (used downstream) but added parallel inline `is not None and len(...) > 0` checks so mypy can narrow the Optional types (booleans aren't type guards). Annotated `aug_input: Dict[str, Any]` so mypy doesn't narrow from the first ndarray entry. Baseline 388 → 219.
- ⬜ Step 2.4: `ml_engine/` (the bulk; will need sub-PR splits like ruff cleanup did)
- ⬜ Step 3: flip `continue-on-error` → false in ci.yml's mypy step

**Step 1 outcome (shipped):** Installed `types-PyYAML` only. Surveyed our other third-party imports (PIL, tqdm, requests) — type stubs exist for those too, but installing them would NOT drop the count today: `--ignore-missing-imports` already suppresses errors for libraries without inline type info, and adding their stubs would START flagging code that uses them. That's a NET INCREASE in the visible baseline, not a decrease — wrong direction for a "quick win" PR. Those stubs should ride with the per-directory cleanup PRs (Step 2) when the related code is being touched anyway.

**Lesson learned:** the original Step-1 estimate of "100-150 errors from missing stubs" was wildly optimistic. Reality: 6. The `--ignore-missing-imports` flag turns out to be a sledgehammer that hides most stub gaps, so removing it would expose them — but removing it is exactly what Step 3 does. So the order doesn't matter much; what matters is committing to the multi-week cleanup or pivoting to alternative 3 (strict-on-critical-only).

**What:** Drive the mypy baseline to zero, then remove `continue-on-error: true` from the mypy step in `ci.yml` (mirror what item 9 + the gate-flip PR did for ruff). End state: mypy actually gates merges.

**Why this is NOT a single PR:** unlike the ruff baseline (mostly mechanical whitespace + import order, ~90% auto-fixable), mypy errors require per-case judgment. Each one is a real type ambiguity:

| Error class | Example | Effort per fix |
|---|---|---|
| Missing 3rd-party stubs | `Library stubs not installed for "yaml"` | 1 line per dep — install `types-PyYAML` etc. |
| Missing variable annotations | `Need type annotation for "annotation_counts"` | 1 line per variable |
| `Optional` mishandling | `Argument 1 to "Path" has incompatible type "str \| None"` | Per-case judgment (assert non-None? refactor signature? change caller?) |
| Variable shadowing | `path: str = ...; path = Path(path)` rebinds incompatibly | Rename or refactor |
| Subclass narrowing | `formatter: ColoredTextFormatter = TextFormatter(...)` | Often signals real type design issues |
| Library `__getattr__` (PEFT pattern) | What item 13 fixed in `merger.py` | Case-by-case `Any` cast |

**Honest scope estimate (~2-3 weeks of focused work):**
- ~100-150 errors are quick wins (install stubs, add obvious annotations) — ~1 hour
- ~150-200 are mechanical per-case (variable shadowing, simple Optional, missing return types) — half-day per directory
- ~50-100 require real semantic work (Optional refactor, type-narrowing assertions) — case-by-case

**Recommended sequencing (mirrors item 9's per-directory rollout, ~6 PRs total):**

```
Step 1 (1 PR, ~1 hr):
  deps: install missing type stubs (types-PyYAML, types-Pillow,
  types-requests, pandas-stubs, etc.) into the lint extra.
  Drops baseline by ~25-100 errors immediately. Zero runtime risk.

Step 2 (4 PRs, ~half-day each):
  mypy-baseline-core/        (~10 errors)
  mypy-baseline-api/         (~20-30 estimated)
  mypy-baseline-augmentation/ (?)
  mypy-baseline-ml-engine/   (bulk; may need sub-PR splits like ruff did)

Step 3 (1 PR, 1 line):
  ci(mypy): flip continue-on-error → false in ci.yml's mypy step
  (mirrors what ci/gate-ruff-checks did for ruff)
```

**Three pragmatic alternatives if 2-3 weeks doesn't justify the ROI:**

1. **Full strict (the path above)** — multi-week effort, ongoing per-PR cost to maintain types. Worth it if type bugs have been biting in production.
2. **Pragmatic non-gating** — keep mypy advisory forever, fix specific high-value modules opportunistically. The status quo. Costs nothing now; defers all benefits.
3. **Strict only on critical modules** — flip mypy to gating ONLY for `core/` (config, logging) and `api/schemas.py` (request/response wire format). Keep `ml_engine/` and others advisory. **~1 PR of work**, captures most of the bug-prevention value with minimal cleanup overhead. Rationale: type discipline matters most where (a) the wire format is defined and (b) other modules import from. ML training internals tend to fail "math is wrong" not "dict had unexpected key" — mypy doesn't help with the former anyway.

**Pros (of full strict):** Real type checking. Catches Optional[X] vs X bugs, wrong-shape dict returns, return-type drift. High value in ML code where tensor shapes rarely match docstrings.

**Cons:** 416 existing errors require per-file judgment work. Ongoing per-PR cost to maintain types. ML libraries (PEFT, transformers, accelerate) use heavy `__getattr__` delegation that mypy can't model — `Any` casts will be needed at boundaries even after cleanup (item 13 documented this for PEFT).

**Recommended next action: continue Step 2.** Path A is the chosen direction. Next is `mypy-baseline-ml-engine/` (~219 remaining errors across 42 files — the bulk of the baseline). Will need sub-PR splits by subdirectory like ruff cleanup did. Use the same per-case judgment that worked for prior steps:
- Path-as-string parameter pattern → widen to `str | Path`
- Subclass-narrowing pattern → annotate as base type
- Variable shadowing → rename or refactor
- Truly intractable cases (PEFT-like `__getattr__` delegation) → `Any` cast at boundary, with comment

If a particular file proves harder than expected, file a sub-TODO and move on. Don't let one stubborn file block the whole rollout.

**Depends on / blocked by:** none. Step 2.2/2.3/2.4 don't block each other; can be done in any order or even in parallel.

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

### 14. `tests/test_sam_lora.py` — 13 pre-existing failures + a CI scope gap

**What:** Investigate and resolve 13 pre-existing test failures in `tests/test_sam_lora.py`. Discovered 2026-04-25 while running the full `pytest tests` (vs the narrower `pytest tests/unit tests/integration tests/contract` that CI runs). Verified pre-existing via `git stash` against `agentic`'s baseline — NOT introduced by any PR in this work stream.

**Sample failures (concrete signal for the investigator):**
- `TestSAMHQLoRAConfig::test_lora_target_modules_format` — asserts `'q_proj' in target_modules` but actual `target_modules` is `['qkv', 'proj']`. Either the target-module naming changed (and the test wasn't updated), OR there's a real regression in `ml_engine/models/teacher/sam_lora.py`.
- `TestSAMHQLoRAForwardPass::test_forward_returns_expected_keys` — expected output shape `(2, 3, 256, 256)`, got `(2, 3, 1024, 1024)`. Indicates either the SAM forward pass was rewritten to return native 1024-resolution masks (without updating the test) OR the test was wrong from the start.

**Why this needs attention:**
1. **Real bug vs stale test** — both interpretations have evidence. Either way, the truth needs to be established. If real bug: `sam_lora.py` is silently broken in production. If stale test: confusing signal whenever someone runs the test file directly.
2. **CI scope gap** — `tests/test_sam_lora.py` is a root-level test file (`tests/*.py`) that CI's `pytest tests/unit tests/integration tests/contract` never picks up. So these failures have been invisible to CI for as long as they've existed. CI was designed to skip root-level tests intentionally (item 11 of the ci-and-tests design doc lists root-level files as "audit for staleness — `test_data_manager.py`, `test_sam_lora.py`, `test_auto_labeler.py`"), but the audit never finished. This is the unfinished half of that audit.

**Pros:**
- Resolves a real signal/noise problem (right now nobody knows if these are bugs)
- Closes the CI scope gap — either delete dead tests or move them under `tests/unit/` so CI runs them
- May reveal an actual production bug in SAM-HQ LoRA wrapper

**Cons:**
- Investigation requires understanding SAM-HQ LoRA internals (target_modules, forward pass shape contract)
- If the tests turn out to be stale, deletion + reasoning needs to be documented
- If a real bug is found, the fix may be deeper than expected (LoRA adapter shape contract changes are usually invasive)

**Context:** The 3 root-level tests in `tests/` are: `test_sam_lora.py`, `test_data_manager.py`, `test_auto_labeler.py`. The ci-and-tests design doc explicitly flagged these as "audit for staleness" but didn't follow through. Same investigation should cover all three. Suggested minimal path:

1. Run each root-level test file with verbose pytest (`-v --tb=long`). Cluster failures.
2. For each cluster, decide: real bug → file separately + fix; stale test → delete with one-line PR commit explaining why; outdated assumption → update the test.
3. After per-file decisions: either move the file under `tests/unit/` (so CI catches future regressions) or delete it.

**Depends on / blocked by:** None. Independent of in-flight work.

---

### 15. WebSocket route `/ws/jobs/{job_id}` — restore live event tailing

**What:** The `/ws/jobs/{job_id}` route used to live-tail Redis pub/sub events for a job's lifetime. The subscription mechanism (`AsyncJobManager.subscribe_to_job_async`) was deleted in commit `bfdff7f` ("remove subscription to job publish; change timing format") and never replaced. The route's caller (`api/routes/websocket.py:97`) kept calling the deleted method — which would have crashed with `AttributeError` on every connection if anyone exercised it (the integration test was `@pytest.mark.skip`'d during ci-and-tests work, so the bug never surfaced in CI).

**Current degraded behavior** (set by the mypy-baseline-api PR): the route now serves the current job state (initial snapshot + terminal payload if applicable), then sends an explicit `subscription_unavailable` frame and closes cleanly. Clients that need live updates must poll `GET /api/jobs/{job_id}` instead.

**Why fix it properly:**
- Polling is wasteful at scale (every client poll hits Redis); pub/sub is event-driven
- The route currently advertises a websocket URL but doesn't deliver the value websockets exist for
- The `progress.update_progress` call path in `ml_engine/jobs/redis_store.py` already publishes events (see `publish_event` on `AsyncJobManager`) — there's a producer with no consumer

**Pros:**
- Closes the gap between the route's URL contract and its actual behavior
- Real-time training progress for UI clients without per-second polling
- Pairs naturally with the cancel-on-disconnect flow that's part of `AsyncJobManager`

**Cons:**
- Reintroducing pub/sub-to-asyncio bridging is non-trivial. Three plausible approaches:
  1. **Polling-loop replacement:** simpler. Drop the websocket nature; convert to long-polling REST. Fewer moving parts but loses true live updates.
  2. **Async Redis subscription via `redis.asyncio.client.PubSub`:** native asyncio path, no thread bridging. Requires careful task lifecycle management (cancel subscription on disconnect, handle Redis disconnects).
  3. **Background thread + `run_coroutine_threadsafe`:** what the deleted code tried to do. Works but threads-in-async is finicky (the old code's `asyncio.get_event_loop()` call in a thread is a known footgun in modern Python).
- Need integration test that actually connects and verifies a published event reaches the client (the existing test was skipped — needs unskipping AND event-flow assertions).

**Context:** Producer side already emits events: `AsyncJobManager.publish_event(job_id, event)` is called from `ml_engine/jobs/worker.py` (training subprocess events) and `ml_engine/agent/loop.py` (agent runs). The Redis stream key pattern is `agent:{run_id}:events` for agents and `job:{job_id}:events` (or similar — verify) for jobs. Consumer side is what's missing.

**Recommended approach:** option 2 (async Redis subscription). Cleaner than the deleted thread-based version, and asyncio-native fits FastAPI's runtime model. Sketch:

```python
async def _subscribe_to_events(redis_async, job_id: str):
    pubsub = redis_async.pubsub()
    await pubsub.subscribe(f"job:{job_id}:events")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])
    finally:
        await pubsub.unsubscribe(f"job:{job_id}:events")
        await pubsub.close()
```

Then the route's main loop becomes `async for event in _subscribe_to_events(...): await ws.send_json(event)` with the existing terminal-state detection.

**Depends on / blocked by:** None for the implementation. Should NOT proceed until someone confirms the publish_event channel naming convention (`job:{id}:events` vs other) and the event payload schema. The deleted code's tests would have documented this — they're gone now.

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
