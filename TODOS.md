# TODOs

Deferred work items with enough context to pick up later. Keep entries self-contained — don't assume the reader has the conversation in which they were captured.

---

## Agent Coordinator — lifecycle durability

Three related design gaps surfaced during the first `/plan → /approve` integration attempt on branch `agentic`. All three contribute to the same symptom class: a run gets stuck in a non-terminal state with no way for the system to recover on its own. See the RCA for run `5327209d-af51-47bc-8179-37f22786383f` (2026-04-21) for a concrete trace.

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


### 20. Refine Coordinator crash classification (failed_retrying for transient errors)

All Coordinator crashes currently → `failed_unrecoverable` (shipped in #2). Transient
failures (import errors, Redis connection reset, network timeout) should route to
`failed_retrying` instead to allow automatic recovery. Needs: (a) classify the exception
type at the `_on_done` site in `api/routes/agent.py`; (b) verify `failed_retrying` has a
re-launch path back to an active state (current TRANSITIONS do; `_start_coordinator` is
already idempotent). Full context in
[docs/decisions/02-coordinator-crash-failed-unrecoverable.md](docs/decisions/02-coordinator-crash-failed-unrecoverable.md).

**Depends on / blocked by:** TODO #1 (auto-resume infra) — retrying is only useful once a
re-launched Coordinator can actually pick up the run.

---

### ~~21. `pending_contract_approval` — no endpoint to advance out of it (blocker)~~

**Completed:** v0.1.4 (2026-04-28) — Extended `POST /api/agent/gate/{run_id}/{action}` to handle `pending_contract_approval`. Chose Option A (human-gated pause): approve → `auto_labeling` (event: `contract_approved`), reject → `cancelled`. Also fixed pre-existing bug where `pending_approval` reject incorrectly targeted `escalated` (not a valid SM transition). 13 unit tests in `TestHumanGate`. GitHub issue #52.

**Depends on / blocked by:** None. Self-contained. Should be resolved before any end-to-end integration testing since the happy path goes through this state.

---

### 22. `failed_retrying` — no retry dispatch (stuck state)

**What:** When a job fails and `retry_count < max_retries`, `on_event` transitions the SM to `failed_retrying` and returns ([coordinator.py:462-467](ml_engine/agent/coordinator.py#L462-L467)). Nothing dispatches a retry job. No event is published to trigger a re-dispatch. On restart, the Coordinator resumes and waits — but there is no pending event in the stream. The run is stuck in `failed_retrying` forever.

**Why:** The transition to `failed_retrying` was added to allow retries, but the actual retry dispatch logic was never implemented.

**Fix:** After `sm.transition("failed_retrying")`, publish a `retry_requested` event to the stream (`type`, `stage`, `run_id`, `retry_count`). The Coordinator's `on_event` handler picks it up on the next loop iteration, reads the last dispatched stage from `LoopState` or the SM metadata, and re-dispatches via `dispatch_stage`. Alternatively, the Coordinator can call `dispatch_stage` directly in the `job_failed` handler before returning (simpler, but skips the event log).

**Context:** `on_event` job_failed branch at [ml_engine/agent/coordinator.py:461-468](ml_engine/agent/coordinator.py#L461-L468). `TRANSITIONS["failed_retrying"]` at [ml_engine/agent/state_machine.py:84-89](ml_engine/agent/state_machine.py#L84-L89).

**Depends on / blocked by:** TODO #20 (crash classification) — the distinction between `failed_retrying` and `failed_unrecoverable` only matters once retry dispatch actually works.

---

### 23. Worker tasks never self-terminate after run reaches terminal state

**What:** `StreamConsumer.should_stop()` always returns `False` ([stream_consumer.py:320-322](ml_engine/agent/stream_consumer.py#L320-L322)). After a run reaches `done`, `escalated`, or `cancelled` (e.g., via `/gate/{run_id}/approve`), the Coordinator, ExecutorWorker, and EvaluatorWorker tasks keep polling the stream indefinitely via `asyncio.gather`. `_on_done` (in `_start_coordinator`) fires only when the task completes — which never happens. The tasks are leaked per completed run.

**Why:** `should_stop()` was left as a no-op stub. The state machine check inside `on_event` silently drops events when terminal, but doesn't cause the loop to exit.

**Fix:** Override `should_stop()` in each worker (or in `AgentLoop` / the base class) to call `sm.current_state()` and return `True` when the state is in `TERMINAL_STATES`. The `should_stop()` call is inside the main `while True` loop in `StreamConsumer.run()`, so returning `True` breaks out cleanly.

**Context:** `should_stop()` stub at [ml_engine/agent/stream_consumer.py:320-322](ml_engine/agent/stream_consumer.py#L320-L322). Main loop check at [ml_engine/agent/stream_consumer.py:148-154](ml_engine/agent/stream_consumer.py#L148-L154). `asyncio.gather` in `Coordinator.run()` at [ml_engine/agent/coordinator.py:616-620](ml_engine/agent/coordinator.py#L616-L620).

**Depends on / blocked by:** None. Low urgency — impact is a handful of idle polling tasks per completed run, not a correctness issue.

---

### 24. Multi-instance Coordinator collision — no distributed lock on resume

**What:** `resume_orphaned_coordinators()` has no distributed lock. If two FastAPI replicas start simultaneously (rolling deploy, blue/green, crash-loop restart), both scan Redis and both call `_start_coordinator` for the same non-terminal runs. Two Coordinator tasks share the same consumer group name (`"coordinator"`) with the same `consumer_name="coordinator-0"`. Redis delivers each message to one consumer in the group, but two concurrent consumers with the same name have undefined delivery behavior — messages can be double-processed or starved.

**Why:** `_start_coordinator` is idempotent within a single process (`_coordinator_tasks` dict prevents duplicate tasks in-process), but the guard is in-memory and not shared across replicas.

**Fix:** Before launching a Coordinator, acquire a Redis lock (`SET run:{run_id}:coordinator_lock NX PX 30000`). Release on Coordinator exit (or let it TTL if the process dies). Only one replica holds the lock at a time. The other replica's `_start_coordinator` call silently no-ops if the lock is taken.

**Context:** `resume_orphaned_coordinators` at [api/routes/agent.py:142-171](api/routes/agent.py#L142-L171). `_start_coordinator` idempotency guard at [api/routes/agent.py:77-79](api/routes/agent.py#L77-L79).

**Depends on / blocked by:** Only relevant at >1 replica. Current single-instance dev is unaffected. Defer until multi-instance deployment is planned.

---

### 25. `CharacteristicSchemaIntegrity` tests cover only 5 of 9 characteristics

**What:** `TestCharacteristicSchemaIntegrity::test_every_transform_has_probability` spot-checks `changes_shape`, `low_contrast`, `reflective_surface`, `partially_hidden`, `moves_or_vibrates`. Four characteristics are unchecked: `changes_size`, `semi_transparent`, `similar_to_background`, `multiple_objects`. A missing `p` key in any of their transforms would evade the test and crash `_keep_higher_p` at runtime.

**Also:** `semi_transparent` low-intensity `RandomFog` uses key `alpha_corf` — likely a typo for `alpha_coef` (used consistently elsewhere at lines 314 and 330). The schema test doesn't catch data typos, only missing `p`.

**Fix (2 parts):**
- Extend the parametrize list in `test_every_transform_has_probability` to include all 9 characteristics.
- Verify `semi_transparent`'s `alpha_corf` key is intentional or correct it.

**Why deferred:** Same pattern as TODO #19. No live overlap collision today, but adding a new rule for any of the 4 unchecked characteristics without `p` would crash silently.

**Context:** Surfaced 2026-04-28 during `/ship` adversarial review of PR #50.

**Depends on / blocked by:** None. Mechanical extension.

---

### 26. Environment rules have no schema integrity tests

**What:** `CHARACTERISTIC_RULES` has schema tests in `TestCharacteristicSchemaIntegrity`, but `ENVIRONMENT_RULES` (12 environment rules: `variable_lighting`, `fixed_camera`, etc.) has no equivalent. `_keep_higher_p` is called in both the characteristic dedup loop (line 1183) and the environment dedup loop (line 1207). An environment rule missing `p` in any transform would crash `_keep_higher_p` at runtime.

**Fix:** Add `TestEnvironmentSchemaIntegrity` class mirroring `TestCharacteristicSchemaIntegrity` — parametrize over all environment rule keys, assert every transform at every intensity has a `p` key.

**Why deferred:** No live environment rule violates the constraint today. Defensive coverage.

**Context:** Surfaced 2026-04-28 during `/ship` adversarial review of PR #50.

**Depends on / blocked by:** None. ~20 lines of test.

---

## Completed

One-line stubs for items that have shipped. Long-form context (what shipped,
patterns established, lessons learned) lives in [docs/decisions/](docs/decisions/).
Item numbers are stable so commit messages and PR descriptions referencing
"item N" / "TODO #N" still resolve.

- **#1** Auto-resume orphaned Coordinator tasks on FastAPI startup ✅ → [docs/decisions/01-03-coordinator-durability.md](docs/decisions/01-03-coordinator-durability.md) — `resume_orphaned_coordinators()` scans Redis on startup, skips pre-approve and terminal runs, relaunches the rest; `store_approved_contract` / `get_approved_contract` persist the contract in the Redis HASH so startup recovery can reconstruct the Coordinator; ~20 unit tests. Shipped 2026-04-27.
- **#2** Coordinator crash → `failed_unrecoverable` ✅ → [docs/decisions/02-coordinator-crash-failed-unrecoverable.md](docs/decisions/02-coordinator-crash-failed-unrecoverable.md) — `_on_done` now schedules state transition on task exception; TRANSITIONS expanded to allow `failed_unrecoverable` from all 5 previously-missing non-terminal states; 5 unit tests. Shipped 2026-04-27.
- **#3** Make `POST /api/agent/approve` idempotent ✅ → [docs/decisions/01-03-coordinator-durability.md](docs/decisions/01-03-coordinator-durability.md) — reads current state first; on first call transitions `created → planning` and publishes `contract_approved`; on re-approve skips both (Coordinator resumes from stream PEL); always persists approved contract and calls `_start_coordinator`; 409 on terminal state. Shipped 2026-04-27.
- **#4** Pre-commit hooks ✅ → [docs/decisions/04-pre-commit-hooks.md](docs/decisions/04-pre-commit-hooks.md) — local + CI lint parity via `uv run --no-sync`. Shipped 2026-04-24 (PR #26).
- **#6** mypy baseline cleanup — drive 416 errors to zero, then flip the gate ✅ → [docs/decisions/06-mypy-baseline-cleanup.md](docs/decisions/06-mypy-baseline-cleanup.md) — 9 PRs total. Mypy now gates merges across `core ml_engine api augmentation` (119 source files, 0 errors). Surfaced 3 real production bugs + filed TODOs #16 and #17 for follow-ups. Establishes the boundary-`Any`, redis-overload, `MpEvent`, path-shadow, and lazy-init-`Any` patterns most subsequent type work follows. Shipped 2026-04-26.
- **#9** Clean up ruff baseline (3593 findings at ci-and-tests merge time) ✅ → [docs/decisions/09-ruff-baseline-cleanup.md](docs/decisions/09-ruff-baseline-cleanup.md) — 5 PRs total (#29-32 per-directory cleanup + #33 gate flip). Ruff now gates merges. Set the per-directory-cleanup-then-flip-the-gate precedent that #6 later followed for mypy. Shipped 2026-04-25.
- **#10** Error-path coverage for `augmentation_factory._validate_bboxes` ✅ → [docs/decisions/10-augmentation-validator-tests.md](docs/decisions/10-augmentation-validator-tests.md) — 44 parametrized tests, every error branch under test, ~8s runtime. Shipped 2026-04-24.
- **#12** Priority-2 unit test roster — 6 remaining files from ci-and-tests design doc ✅ → [docs/decisions/12-p2-unit-test-roster.md](docs/decisions/12-p2-unit-test-roster.md) — 4 PRs total. ~470 new tests (375 passing + 47 xfail markers documenting source gaps; 13 of those gaps fixed inline, 38 remain in TODOs #18 + #19). Established the `xfail(strict=True)` as embedded to-do list pattern. Shipped 2026-04-27.
- **#13** Type-annotate `ml_engine/export/merger.py` ✅ → [docs/decisions/13-merger-py-mypy-fix.md](docs/decisions/13-merger-py-mypy-fix.md) — established the `Any`-at-the-boundary precedent for PEFT's `__getattr__` delegation. Shipped 2026-04-25 (`chore/mypy-merger-hygiene`).
- **#17** Restore `text_threshold` token-level filtering in `GroundingDINODetector` ✅ → [docs/decisions/17-text-threshold-filtering.md](docs/decisions/17-text-threshold-filtering.md) — `logits_to_class_scores` now zeros sub-threshold tokens before per-class mean; `detect()` passes the param through instead of discarding it; 32 adversarial tests (boundary, mutation, class-flip, dilution, NMS, monotone sweep). Shipped 2026-04-28.
- **#16** Plumb `job_id` through training pipeline so artifact manifests carry real lineage ✅ → [docs/decisions/16-job-id-lineage-plumbing.md](docs/decisions/16-job-id-lineage-plumbing.md) — first follow-up surfaced by #6. Trial subprocesses use composed `f"{job_id}/{trial_id}"` form. Shipped 2026-04-27 (PR #41).
- **#14** `tests/test_sam_lora.py` audit — 13 stale failures + CI scope gap ✅ — All 13 failures were stale tests. Moved to `tests/unit/ml_engine/test_sam_lora.py` so CI picks them up. Fixed 3 real bugs: `upscale_masks()` crashes on 5D multimask tensors (5D reshape path added); `SegmentationLoss` ignored `iou_predictions` (IoU quality MSE regression added); `box_prompts=[N=0]` now raises clear `ValueError` at call site. Pre-landing: added explicit `[B,N]` shape guard + clear error on `iou_predictions`, `iou_quality` key in default weights dict, rank guard on `upscale_masks`. 24 tests. Shipped 2026-04-28.
- **#19** `_keep_higher_p` KeyError on missing `p` key ✅ — Added guard at `characteristic_translator.py:1109` that raises `ValueError` with a clear message when either params dict is missing `p`. xfail test promoted to passing regression guard (93 pass, 0 xfail). Shipped 2026-04-28.
