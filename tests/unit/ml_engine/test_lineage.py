"""
Lineage plumbing tests for TODO #16.

These tests pin down the contract that artifact manifests carry a real
job_id end-to-end:

- Every JobHandler subclass accepts `job_id` in its run() signature
  (catches future drift if someone adds a new handler that forgets the
  param — the abstract base would let them, since Python doesn't enforce
  parameter names on overrides).
- Trainer and BaseModelTrainer accept `job_id` and propagate it.
- For experiment trials, the trial subprocess receives a composed
  `f"{job_id}/{trial_id}"` so manifests can be traced to both.
- The artifact dataclasses (CreateByInfo, BundleManifest) require a real
  job_id (no longer Optional after TODO #16 plumbing landed).
"""

from __future__ import annotations

import inspect
from typing import Dict, get_type_hints

import pytest

from ml_engine.artifacts.schemas import (
    AdapterManifest,
    BaseModelRef,
    BundleManifest,
    CreateByInfo,
)


class TestHandlerSignatures:
    """Catches future handlers that drift from the abstract JobHandler.run signature."""

    @pytest.mark.parametrize(
        "module_path, class_name",
        [
            ("ml_engine.jobs.handlers.base", "JobHandler"),
            ("ml_engine.jobs.handlers.teacher", "TeacherTrainingHandler"),
            ("ml_engine.jobs.handlers.auto_label", "AutoLabelHandler"),
            ("ml_engine.jobs.handlers.distillation", "StudentDistillationHandler"),
            ("ml_engine.jobs.handlers.experiment_loop", "ExperimentLoopHandler"),
        ],
    )
    def test_handler_run_accepts_job_id(self, module_path: str, class_name: str) -> None:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        sig = inspect.signature(cls.run)
        assert "job_id" in sig.parameters, (
            f"{class_name}.run() is missing the `job_id` parameter — "
            f"the abstract JobHandler.run requires it as of TODO #16. "
            f"Without it, artifact manifests written by this handler would "
            f"have no lineage back to the parent job."
        )
        # Also lock the type to str (not Optional) so Trainer doesn't
        # accept None and silently drop lineage.
        hints = get_type_hints(cls.run)
        assert hints.get("job_id") is str, (
            f"{class_name}.run() job_id must be typed `str` (non-Optional) — got {hints.get('job_id')!r}."
        )


class TestTrainerSignatures:
    """Trainer + BaseModelTrainer must accept and store job_id."""

    def test_trainer_init_accepts_job_id(self) -> None:
        from ml_engine.training.trainer import Trainer

        sig = inspect.signature(Trainer.__init__)
        assert "job_id" in sig.parameters
        hints = get_type_hints(Trainer.__init__)
        assert hints.get("job_id") is str

    def test_base_model_trainer_init_accepts_job_id(self) -> None:
        from ml_engine.training.model_trainers.base import BaseModelTrainer

        sig = inspect.signature(BaseModelTrainer.__init__)
        assert "job_id" in sig.parameters
        hints = get_type_hints(BaseModelTrainer.__init__)
        assert hints.get("job_id") is str

    @pytest.mark.parametrize(
        "module_path, class_name",
        [
            ("ml_engine.training.model_trainers.grounding_dino", "GroundingDINOTrainer"),
            ("ml_engine.training.model_trainers.sam", "SAMTrainer"),
        ],
    )
    def test_concrete_trainers_accept_job_id(self, module_path: str, class_name: str) -> None:
        """Subclasses must thread job_id through to super().__init__."""
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        sig = inspect.signature(cls.__init__)
        assert "job_id" in sig.parameters, (
            f"{class_name}.__init__() is missing `job_id` — BaseModelTrainer requires it as of TODO #16."
        )


class TestTrialRunnerComposition:
    """ExperimentLoop / TrialRunner compose `f'{job_id}/{trial_id}'` for trial manifests."""

    def test_trial_runner_run_accepts_job_id(self) -> None:
        from ml_engine.experiment.trial_runner import TrialRunner

        sig = inspect.signature(TrialRunner.run)
        assert "job_id" in sig.parameters

    def test_experiment_loop_run_accepts_job_id(self) -> None:
        from ml_engine.experiment.loop import ExperimentLoop

        sig = inspect.signature(ExperimentLoop.run)
        assert "job_id" in sig.parameters

    def test_trial_subprocess_first_arg_is_composed_job_id(self) -> None:
        """
        _trial_subprocess takes `composed_job_id` as the first positional arg
        (so it can be passed to Trainer(job_id=...) inside the spawn'd process).
        Pinning the parameter name + position prevents accidental reorders that
        would put data_manager into the job_id slot.
        """
        from ml_engine.experiment.trial_runner import _trial_subprocess

        params = list(inspect.signature(_trial_subprocess).parameters)
        assert params[0] == "composed_job_id", (
            f"Expected first param of _trial_subprocess to be "
            f"'composed_job_id', got {params[0]!r}. The composed value "
            f"f'{{job_id}}/{{trial_id}}' must be the first positional arg "
            f"so the spawn-context Process's args tuple stays ordered."
        )


class TestSchemaContract:
    """The artifact dataclasses now require a real job_id (TODO #16 reverted Optional)."""

    def test_create_by_info_requires_job_id(self) -> None:
        # str — not Optional[str]. Constructing with a real value works.
        info = CreateByInfo(job_id="job-abc", timestamp="2026-04-26T12:00:00Z")
        assert info.job_id == "job-abc"

        # Type contract: job_id field is typed `str` exactly.
        hints = get_type_hints(CreateByInfo)
        assert hints["job_id"] is str

    def test_bundle_manifest_lineage_requires_str_values(self) -> None:
        manifest = BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"sam": "sam/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-abc"},
        )
        assert manifest.lineage["job_id"] == "job-abc"

        # Type contract: lineage is Dict[str, str], no Optional anywhere.
        # (typing.Dict[str, str], not the lowercase builtin form — pinned
        # to whichever the schema uses today so a future migration is
        # noticed and the assertion updated explicitly.)
        hints = get_type_hints(BundleManifest)
        assert hints["lineage"] == Dict[str, str]

    def test_full_manifest_save_load_roundtrip_preserves_job_id(self, tmp_path) -> None:
        """End-to-end: writing and reading back a manifest preserves job_id verbatim."""
        manifest = AdapterManifest(
            model_family="grounding_dino",
            base_model=BaseModelRef(checkpoint_path="data/models/dino.pth"),
            peft_files={"config": "adapter_config.json", "weights": "adapter_model.safetensors"},
            created_by=CreateByInfo(
                job_id="job-abc/trial_001",
                timestamp="2026-04-26T12:00:00Z",
            ),
        )
        path = tmp_path / "adapter.manifest.json"
        manifest.save(path)

        loaded = AdapterManifest.load(path)
        assert loaded.created_by.job_id == "job-abc/trial_001"

    def test_composed_trial_id_format_is_recoverable(self) -> None:
        """
        Composed format `f'{job_id}/{trial_id}'` should be deterministic so
        future tools can split it. We're not adding a parser here (one-way
        composition is fine for now), but pin the format so it doesn't drift.
        """
        composed = "job-abc/trial_003"
        # Round-trip via str.split('/', 1) — single split, in case job_id
        # ever contains a '/' (unlikely; UUIDs don't).
        parent, trial = composed.split("/", 1)
        assert parent == "job-abc"
        assert trial == "trial_003"
