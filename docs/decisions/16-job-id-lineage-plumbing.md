# 16. Plumb `job_id` through training pipeline so artifact manifests carry real lineage

**Completed:** 2026-04-27 on `feat/job-id-lineage-plumbing` (PR #41).

## What shipped

`job_id` now flows from `_job_entry_point` → `JobHandler.run()` →
`Trainer.__init__` → `BaseModelTrainer.__init__` → `CreateByInfo` and
`BundleManifest.lineage`. For experiment trials, `TrialRunner.run()`
composes `f"{job_id}/{trial_id}"` (e.g. `"job-abc/trial_003"`) before
passing into the trial subprocess so per-trial manifests are uniquely
identifiable while still tracing back to the parent experiment job.

`CreateByInfo.job_id` and `BundleManifest.lineage` reverted from
`Optional[str]` back to `str` / `Dict[str, str]`. New
`tests/unit/ml_engine/test_lineage.py` (16 tests) pins the contract:
handler signatures must accept `job_id`, the trial subprocess's first
positional arg is `composed_job_id`, schema types are non-Optional,
manifest save/load roundtrip preserves the value verbatim.

## Why

Today the artifact manifests written by training jobs had
`created_by.job_id = None` and `lineage = {"job_id": None}` because the
trainer couldn't see the parent job_id. The fact that the schemas had to
be widened to `Optional[str]` in Step 2.4.3 of TODO #6 was the smell —
the value is *known at job submission time* (line 79 of
`api/routes/jobs.py` enqueues with a real id) but was dropped at the
subprocess boundary in `subprocess_runner._job_entry_point` (line 409),
which called `handler.run(job_config=..., output_dir=...,
progress_queue=..., cancel_event=...)` without forwarding the job_id it
already held (line 333).

Without lineage, you can't go from a `bundle.manifest.json` on disk back
to the job that produced it. That breaks: post-hoc debugging ("which job
made this bad model?"), audit/compliance ("who/when produced this
artifact?"), automated cleanup ("delete artifacts from cancelled jobs"),
and any future eval-tracking that wants to correlate model performance
with training-job config.

## Plumbing path (top-down)

1. `ml_engine/jobs/handlers/base.py`: added `job_id: str` to abstract
   `run()` signature
2. `ml_engine/jobs/subprocess_runner.py:409`: passed `job_id=job_id` to
   `handler.run(...)` (the variable already existed on line 333)
3. Each handler subclass — accepts and forwards:
   - `auto_label.py` — accepts (no Trainer downstream; threaded for
     Protocol/abstract compliance)
   - `teacher.py` — passes to `Trainer(job_id=job_id, ...)`
   - `distillation.py` — accepts (StudentDistillationHandler doesn't
     write CreateByInfo manifests today; ready when it does)
   - `experiment_loop.py` — passes to `ExperimentLoop.run(job_id=...)`,
     which forwards to `TrialRunner.run(job_id=...)`
4. `ml_engine/training/trainer.py`:
   - `Trainer.__init__` adds `job_id: str` parameter, stores
     `self.job_id: str`
   - Forwards to per-model trainers in `_init_trainers`
   - `_save_adapters` uses `self.job_id` in
     `BundleManifest(lineage={"job_id": self.job_id})`
5. `ml_engine/training/model_trainers/base.py`:
   - `BaseModelTrainer.__init__` adds `job_id: str`, stores
     `self.job_id: str`
   - `save_adapters()` uses `self.job_id` in
     `CreateByInfo(job_id=self.job_id, timestamp=...)`
6. `ml_engine/experiment/trial_runner.py`:
   - `_trial_subprocess` accepts `composed_job_id: str` as first
     positional arg
   - Composes `f"{job_id}/{trial_id}"` and passes to `Trainer(job_id=...)`
7. `ml_engine/artifacts/schemas.py` — reverted:
   - `CreateByInfo.job_id: Optional[str]` → `str`
   - `BundleManifest.lineage: Dict[str, Optional[str]]` → `Dict[str, str]`
   - `BaseAdapterMismatch.__init__(expected, actual)` stayed `Optional[str]`
     — that one is genuinely about Optional fields on `BaseModelRef`,
     unrelated to this gap

## Trial-mode design choice (option C)

Three options were considered for what populates `job_id` in per-trial
manifests:
- **(a)** reuse the parent job_id (clean — every trial in an experiment
  shares the experiment's job_id, lineage groups them)
- **(b)** use the trial_id (e.g. `trial_a1b2c3d4`) so each trial's
  artifact is uniquely identifiable
- **(c) chosen:** compose `f"{job_id}/{trial_id}"` — preserves both the
  parent-job link and per-trial uniqueness without changing the schema
  field name

Composition is one-way (no parser added), but the format is recoverable
via `composed.split("/", 1)` if a future tool needs to extract parent
vs trial.

## Test surface

New `tests/unit/ml_engine/test_lineage.py` — 16 tests:
- Every JobHandler subclass accepts `job_id: str` (parametrized over
  abstract base + 4 concretes); catches future drift if someone adds a
  new handler that forgets the param
- Trainer + BaseModelTrainer + GroundingDINOTrainer + SAMTrainer
  `__init__` all accept `job_id: str` (signature + type hint)
- TrialRunner.run + ExperimentLoop.run accept `job_id`
- `_trial_subprocess`'s first positional arg is `composed_job_id` —
  pinning name + position prevents accidental reorders that would put
  data_manager into the job_id slot of the spawn-context Process args
- CreateByInfo.job_id and BundleManifest.lineage type contracts are
  non-Optional
- AdapterManifest save/load roundtrip preserves job_id verbatim
  (including the composed `parent/trial` form)

Existing test updated:
- `tests/unit/ml_engine/test_experiment_loop_handler.py` — 3
  `ExperimentLoopHandler.run()` callsites now pass
  `job_id="test-job-..."` fixture strings.

## Outcome

Manifests written by training jobs from now on carry real lineage:
- For standalone teacher_training jobs: `created_by.job_id = the parent
  job id from JobManager.submit_job`
- For experiment trials: `created_by.job_id =
  parent_experiment_job_id/trial_NNN`

This unlocks: post-hoc debugging, audit/compliance, automated cleanup of
artifacts from cancelled jobs, and future eval-tracking that correlates
performance with training config.
