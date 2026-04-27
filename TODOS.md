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

### 17. Restore `text_threshold` token-level filtering in GroundingDINODetector (silent drop bug)

**What:** The `text_threshold` parameter flows from the public API
(`api/schemas.py:247`, default 0.5) all the way through:
`POST /api/autolabel` → `autolabel.py:75` → `auto_label.py:82` →
`DetectionThresholds.text` → `AutoLabeler.config.thresholds.text` →
`detector.detect(text_threshold=...)` → and is **silently dropped** by
`GroundingDINODetector.detect()` (`ml_engine/inference/detectors/grounding_dino.py`).
The implementation currently accepts the param but does nothing with it
(marked `_ = text_threshold` for linter silence). The original token-level
filter (`logit > text_threshold` inside `logits_to_class_scores` /
`get_phrases_from_posmap` per the demo files in `grounded_sam_demo.py:86`)
was removed during a refactor and never restored.

**Why this is a real bug:** The API contract advertises a knob the user
expects to control text-prompt sensitivity. Today, changing
`text_threshold` from 0.5 to 0.9 (or 0.1) has zero effect on the returned
detections — only `box_threshold` and `nms_threshold` filter results.
Users who tune the knob expecting changed behavior will report it as
"the detector ignores my config" — and they'll be right.

**Pros:**
- Honors the public API contract; eliminates a silent no-op knob
- Restores the original GroundingDINO paper's token-level filtering, which
  matters when prompts have ambiguous/overlapping tokens (e.g.
  "person, person riding a bike" — token-level filtering disambiguates)
- Removes one of the few remaining `# TODO: text_threshold` comments and
  the explanatory note in `DetectorProtocol.detect`'s docstring

**Cons:**
- Need to inspect `logits_to_class_scores` and decide where the threshold
  applies — at the per-token sigmoid stage, or at the per-class aggregation
  stage. The demos use it at the per-token stage
  (`get_phrases_from_posmap(logit > text_threshold, ...)`); modern detect
  doesn't use the phrase-extraction code path, so a 1:1 port doesn't apply.
- Changing detection behavior is a behavior change; need a test that
  asserts text_threshold actually filters (and isn't a placebo). Without
  a test, "fixed" can silently regress to "still ignored" later.

**Plumbing context (already partly done in Step 2.4.7 of TODO #6):**
- `DetectorProtocol.detect` in `ml_engine/inference/detectors/base.py`
  declares `text_threshold: float = 0.5` (kept).
- `GroundingDINODetector.detect` in
  `ml_engine/inference/detectors/grounding_dino.py` accepts but ignores
  the param (`_ = text_threshold` placeholder + docstring TODO link).
- `AutoLabeler` in `ml_engine/inference/auto_labeler.py:153` now passes
  `text_threshold=self.config.thresholds.text` through (was previously
  dropped at this layer too — restored as part of the typing fix so the
  Protocol matches).

**Test surface:**
- `tests/unit/ml_engine/inference/test_grounding_dino_detector.py` (new):
  - `test_text_threshold_filters_low_confidence_tokens` — synthesize a
    prediction tensor where one token is high-confidence (>0.9) and one is
    low (~0.3); assert that `detect(prompts=[...], text_threshold=0.5)`
    keeps the high one and drops the low one. Will FAIL today (bug
    documented), pass after the fix.
  - `test_text_threshold_at_extremes` — text_threshold=0.0 keeps all,
    text_threshold=1.0 drops all (sanity bookends).
- `tests/integration/test_autolabel_text_threshold_e2e.py` (optional but
  high value): submit an autolabel job with a high text_threshold against
  a known-ambiguous image, assert fewer detections than with default 0.5.
  Catches plumbing regressions in CI.

**Context:** Surfaced 2026-04-26 during Step 2.4.7 of TODO #6 (mypy cleanup
of `ml_engine/inference/`). The mypy gate flagged a Protocol/impl signature
mismatch (`DetectorProtocol.detect` requires text_threshold; impl had it
commented out). The typing-fix path could have been "delete from Protocol"
(declare it dead) — but tracing upstream showed the value flows from a
real public-API field, so the right fix is to honor the API contract
instead of pruning it. Marker comment in `grounding_dino.py:detect()`
docstring + `_ = text_threshold` placeholder point here.

**Depends on / blocked by:** None. Independent of TODO #16's plumbing
work. Recommend shipping with the test above so the fix can't silently
regress.

---

### 18. Tighten remaining `api/schemas.py` enum/range validators (7 truly-breaking categories — needs frontend audit)

**What:** 7 `api/schemas.py` fields still have docstring-vs-validator
gaps after the safe subset shipped in PR test/p2-api-schemas. Each is
either an enum (plain `str` where docstring lists allowed values), a
range gap on a client-supplied field, or a non-empty-list constraint —
all of which would BREAK existing callers if tightened naively.
`tests/unit/api/test_schemas.py` documents each remaining gap as
`xfail(strict=True)` — **37 individual xfails across 7 categories**.
When each validator is tightened, the strict-xfail trips (test
unexpectedly passes), CI fails, and the developer flips the
`@pytest.mark.xfail` decorator off so the test becomes a regression
guard.

**Already shipped in test/p2-api-schemas (8 SAFE categories closed,
32 xfails turned into regular passing tests):** overall_progress range,
current_epoch ≤ total_epochs invariant, split_config sum=1.0, COCO
width/height positive, COCO bbox length=4, COCO iscrowd binary, COCO
score range, VisualizationInfo annotation_count non-negative.

**Remaining 7 categories — DEFERRED for client coordination
(37 xfails total — each test class in `test_schemas.py` carries the
exact `reason=` line + source pointer):**

- **4 enum gaps** (need frontend audit first — clients may be sending
  capitalization variants or legacy synonyms today):
  - `ApiResponse.status` — line 45, docstring says
    `'succeed'|'failed'`. Tighten to `Literal['succeed', 'failed']`.
    *9 xfails* in `TestApiResponseStatusEnumGap` covering empty string,
    capitalization variants, semantic synonyms (`'OK'`, `'true'`,
    `'yes'`, `'passed'`, etc.), and whitespace-padding.
  - `JobCreate.job_type` — line 124, docstring says
    `teacher_training|student_distillation` (plus `auto_label`,
    `experiment_loop` per the handler registry). Build the Literal
    from `JobType` enum in `ml_engine/jobs/models.py`. *6 xfails* in
    `TestJobCreateJobTypeEnumGap`.
  - `AutoLabelRequest.output_mode` — line 245, docstring says
    `'boxes'|'masks'|'both'`. *7 xfails* in
    `TestAutoLabelRequestOutputModeEnumGap`.
  - `WorkerResponse.status` — line 188, docstring says
    `'idle'|'busy'|'offline'`. *5 xfails* in
    `TestWorkerResponseStatusEnumGap`.
- **1 HTTP code range gap:**
  - `ApiResponse.code` — line 44, should be `Field(ge=100, le=599)`.
    *6 xfails* in `TestApiResponseCodeRangeGap` covering negatives,
    zero, two-digit, four-digit codes. Risky because helper functions
    / external callers may currently pass `code=0` or `code=499`
    (custom). Audit `success_response` / `error_response` callsites
    before tightening.
- **1 non-empty-list gap:**
  - `AutoLabelRequest.image_paths` + `classes` — lines 243-244, empty
    list is meaningless for autolabel. Tighten to
    `Field(min_length=1)`. *2 xfails* (1 per field) in
    `TestAutoLabelRequestNonEmptyListGap`. Risky if any "validate
    config / list known classes" path POSTs with empty lists for
    introspection.
- **1 paired-flag invariant:**
  - `DistillationRequest.teacher_dir` + `unlabeled_image_paths` —
    lines 266-271. Per docstring they're a paired flag (one without
    the other is a misconfig). Needs `@model_validator(mode="after")`.
    *2 xfails* (one per direction) in
    `TestDistillationPairedFieldGap`.

**Total: 9 + 6 + 7 + 5 + 6 + 2 + 2 = 37 xfails remaining.**

**Why deferred:** Each of these would break clients currently sending
non-conforming data with no warning. Need:
- **Frontend audit** — does the FE send `"OK"` instead of `"succeed"`?
  Capitalization variants for `output_mode`?
- **Helper-callsite audit for `ApiResponse.code`** —
  `success_response(code=...)` may have unusual int values in tests/
  scripts.
- **Decision on `image_paths=[]` semantics** — is "list known classes"
  a use case?
- **Coordination on paired-flag check** — current behavior may be
  relied on by callers using only one half.

**Pros:**
- Closes the remaining 7 docstring-vs-reality gaps.
- Turns the remaining 37 xfails in `test_schemas.py` into regression
  guards.
- Catches client bugs at the API boundary instead of deep in handler
  code (e.g. unknown `output_mode` currently triggers a cryptic
  `KeyError` somewhere in the autolabeler).

**Cons:**
- Breaking change. Each enum tightening needs an API-changelog note
  and a frontend release that stops sending non-conforming values
  before the schema flips.
- Need to enumerate callers (FE, internal scripts, integration tests)
  before each individual flip.

**Recommended sequencing (2 sub-PRs after frontend audit):**

1. **Enum tightenings** (4 categories, 27 xfails: 9 + 6 + 7 + 5) —
   once frontend audit confirms no capitalization variants in flight.
   Use `Literal[...]` from a shared module (consider
   `ml_engine/jobs/models.py:JobType` for `job_type`).
2. **HTTP code range + non-empty list + paired flag** (3 categories,
   10 xfails: 6 + 2 + 2) — independent; can ship together once their
   respective callers are audited.

**Test surface:** No new tests needed — `tests/unit/api/test_schemas.py`
already has them as `xfail(strict=True)`. Per gap fixed, flip the
corresponding `@pytest.mark.xfail` decorator off. Strict-xfail means
the test fails loudly if you forget to flip it. The class names all
end in `Gap` for grep-ability — when all are fixed, `grep -r
"EnumGap\|RangeGap\|TruncationGap\|InvariantGap\|PairedFieldGap" tests/`
should return nothing.

**Context:** Originally filed 2026-04-27 with 15 categories
(69 xfails) during item #12.1. The 8 safe categories shipped in the
same PR after auditing for client-breakage risk (32 xfails turned into
regular tests); this entry now tracks only the 7 that genuinely need
coordination before flipping. Each remaining xfail's `reason=` line
points at the exact source line + recommended fix.

**Depends on / blocked by:** Frontend audit for the enum tightenings.
HTTP code + non-empty list + paired flag are independent of each other
and the audit, but each wants a callsite enumeration before shipping.

---

## Completed

One-line stubs for items that have shipped. Long-form context (what shipped,
patterns established, lessons learned) lives in [docs/decisions/](docs/decisions/).
Item numbers are stable so commit messages and PR descriptions referencing
"item N" / "TODO #N" still resolve.

- **#4** Pre-commit hooks ✅ → [docs/decisions/04-pre-commit-hooks.md](docs/decisions/04-pre-commit-hooks.md) — local + CI lint parity via `uv run --no-sync`. Shipped 2026-04-24 (PR #26).
- **#6** mypy baseline cleanup — drive 416 errors to zero, then flip the gate ✅ → [docs/decisions/06-mypy-baseline-cleanup.md](docs/decisions/06-mypy-baseline-cleanup.md) — 9 PRs total. Mypy now gates merges across `core ml_engine api augmentation` (119 source files, 0 errors). Surfaced 3 real production bugs + filed TODOs #16 and #17 for follow-ups. Establishes the boundary-`Any`, redis-overload, `MpEvent`, path-shadow, and lazy-init-`Any` patterns most subsequent type work follows. Shipped 2026-04-26.
- **#9** Clean up ruff baseline (3593 findings at ci-and-tests merge time) ✅ → [docs/decisions/09-ruff-baseline-cleanup.md](docs/decisions/09-ruff-baseline-cleanup.md) — 5 PRs total (#29-32 per-directory cleanup + #33 gate flip). Ruff now gates merges. Set the per-directory-cleanup-then-flip-the-gate precedent that #6 later followed for mypy. Shipped 2026-04-25.
- **#10** Error-path coverage for `augmentation_factory._validate_bboxes` ✅ → [docs/decisions/10-augmentation-validator-tests.md](docs/decisions/10-augmentation-validator-tests.md) — 44 parametrized tests, every error branch under test, ~8s runtime. Shipped 2026-04-24.
- **#13** Type-annotate `ml_engine/export/merger.py` ✅ → [docs/decisions/13-merger-py-mypy-fix.md](docs/decisions/13-merger-py-mypy-fix.md) — established the `Any`-at-the-boundary precedent for PEFT's `__getattr__` delegation. Shipped 2026-04-25 (`chore/mypy-merger-hygiene`).
- **#16** Plumb `job_id` through training pipeline so artifact manifests carry real lineage ✅ → [docs/decisions/16-job-id-lineage-plumbing.md](docs/decisions/16-job-id-lineage-plumbing.md) — first follow-up surfaced by #6. Trial subprocesses use composed `f"{job_id}/{trial_id}"` form. Shipped 2026-04-27 (PR #41).
