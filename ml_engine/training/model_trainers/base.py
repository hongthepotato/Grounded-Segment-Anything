"""
Base Model Trainer for teacher model training.

This module provides the base class that all model-specific trainers extend.
It handles common functionality:
- Optimizer and scheduler creation
- Training/validation step orchestration
- Checkpoint management
- TensorBoard logging
- LoRA adapter saving
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from core.constants import DEFAULT_CONFIGS_DIR
from ml_engine.training.checkpoint_manager import CheckpointManager
from ml_engine.training.training_manager import TrainingManager

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Base class for model-specific trainers.

    Subclasses must implement:
    - model_name: Class attribute identifying the model
    - _load_model(): Load and return the model
    - _create_criterion(): Create and return the loss function
    - compute_loss(): Compute loss for a batch

    Example:
        >>> class MyModelTrainer(BaseModelTrainer):
        >>>     model_name = "my_model"
        >>>
        >>>     def _load_model(self):
        >>>         return MyModel(...)
        >>>
        >>>     def _create_criterion(self):
        >>>         return MyLoss(...)
        >>>
        >>>     def compute_loss(self, batch):
        >>>         outputs = self.model(batch['images'])
        >>>         return self.criterion(outputs, batch['targets'])
    """

    model_name: str = "base"

    def __init__(
        self,
        job_id: str,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
        dataset_info: Dict[str, Any],
    ):
        """
        Initialize the base trainer.

        Args:
            job_id: Lineage id forwarded by the parent Trainer (which got
                it from the JobHandler / subprocess_runner). Recorded in
                CreateByInfo.job_id when save_adapters() writes a manifest.
            config: Model-specific configuration
            device: Device to train on
            output_dir: Root output directory
            dataset_info: Dataset metadata (class mapping, etc.)
        """
        self.job_id = job_id
        self.config = config
        self.device = device
        self.output_dir = output_dir / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_info = dataset_info

        # Load model (subclass implements). Annotated as Any (not nn.Module)
        # because subclasses access PEFT-wrapped attributes that go through
        # nn.Module.__getattr__ — mypy resolves those to Tensor/Module via
        # the stub and rejects every `.transformer`, `.tokenizer`, `.predict`
        # access in grounding_dino.py / sam.py. Same boundary pattern as
        # merger.py + peft_utils.save_lora_adapters (TODO #13). Trade-off:
        # lose type-checking on nn.Module's actual methods (parameters,
        # train, eval) which still work at runtime via duck typing.
        logger.info("Loading %s...", self.model_name)
        self.model: Any = self._load_model()
        self.model.to(self.device)
        logger.info("%s loaded", self.model_name)

        # Create criterion (subclass implements). Annotated as Any for the
        # same reason as self.model above: subclasses access criterion-
        # specific attributes (e.g. GroundingDINOCriterion.weight_dict)
        # that aren't on the nn.Module base, and the criterion's __call__
        # signature varies per task. mypy can't narrow nn.Module to the
        # concrete subclass through the factory method.
        self.criterion: Any = self._create_criterion()

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training manager for AMP and gradient handling.
        # config may carry a 'training_dynamics' sub-dict from the experiment loop;
        # pass it through so AutoResearch HPO mutations take effect.
        self.training_manager = TrainingManager(
            model=self.model,
            optimizer=self.optimizer,
            config_path=str(DEFAULT_CONFIGS_DIR / "training_dynamics.yaml"),
            config_overrides=config.get("training_dynamics"),
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=str(self.output_dir),
            config_path=str(DEFAULT_CONFIGS_DIR / "checkpoint_config.yaml"),
            monitor_metric=f"val_{self.model_name}_total_loss",
            mode="min",
        )

        self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))

        logger.info("%s trainer initialized", self.model_name)

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load and return the model. Subclass must implement."""
        raise NotImplementedError

    @abstractmethod
    def _create_criterion(self) -> nn.Module:
        """Create and return the loss function. Subclass must implement."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch. Subclass must implement.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with 'loss' key and any additional metrics
        """
        raise NotImplementedError

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for trainable parameters."""
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        optimizer_type = self.config.get("optimizer", "AdamW")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Annotate as the base type so SGD doesn't fail to assign to a
        # mypy-narrowed AdamW slot.
        optimizer: torch.optim.Optimizer
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            momentum = self.config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        logger.info("  Optimizer: %s (lr=%s)", optimizer_type, lr)
        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler."""
        total_epochs = self.config.get("epochs", 50)
        warmup_epochs = self.config.get("warmup_epochs", 3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(1, total_epochs - warmup_epochs), T_mult=1
        )
        return scheduler

    def train_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict of loss values (float)
        """
        # model.train() is called inside training_manager.training_step() so
        # that BN re-freeze (if configured) happens atomically with the mode switch.

        def _compute_loss(batch):
            return self.compute_loss(batch)

        loss_dict = self.training_manager.training_step(batch, _compute_loss)

        # Convert tensors to floats for logging
        result = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                result[key] = value.item()
            else:
                result[key] = value

        return result

    @torch.no_grad()
    def validate_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one validation step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict of loss values (float)
        """
        self.model.eval()

        loss_dict = self.compute_loss(batch)

        # Convert tensors to floats
        result = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                result[key] = value.item()
            else:
                result[key] = value

        return result

    def step_scheduler(self) -> None:
        """Step the LR scheduler. Called once per epoch by Trainer."""
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Save checkpoint if improved.

        Args:
            epoch: Current epoch
            metrics: All training/validation metrics

        Returns:
            True if this was the best checkpoint
        """
        self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            metrics=metrics,
            scheduler=self.scheduler,
            scaler=self.training_manager.scaler,
            extra_info={"config": self.config},
        )

        return self.checkpoint_manager.best_epoch == epoch

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and return the epoch number.

        Args:
            path: Path to checkpoint (or 'best', 'last')

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.training_manager.scaler,
        )
        return checkpoint.get("epoch", 0)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log metrics to TensorBoard and logger.

        Args:
            metrics: Metrics to log
            step: Global step (usually epoch)
            prefix: Prefix for metric names ('train' or 'val')
        """
        # Filter metrics for this model
        model_metrics = {
            k.replace(f"{self.model_name}_", ""): v for k, v in metrics.items() if self.model_name in k
        }

        # Log to TensorBoard
        for key, value in model_metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

        # Log to console
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in model_metrics.items())
        logger.info("  [%s] %s: %s", prefix, self.model_name, metrics_str)

    def save_adapters(self) -> Optional[Path]:
        """Save LoRA adapters for deployment."""
        if hasattr(self.model, "save_lora_adapters"):
            adapter_dir = self.output_dir / "lora_adapters"
            # Same PEFT-via-nn.Module __getattr__ pattern as merger.py
            # (TODO #13) and peft_utils.save_lora_adapters: cast through
            # Any so the method call type-checks.
            peft_model: Any = self.model
            peft_model.save_lora_adapters(
                output_dir=str(adapter_dir),
                # safe_serialization=True
            )
            peft_files: Dict[str, str] = {}
            for f in adapter_dir.iterdir():
                if f.name.startswith("adapter_config"):
                    peft_files["config"] = f.name
                elif f.name.startswith("adapter_model"):
                    peft_files["weights"] = f.name

            from ml_engine.artifacts import AdapterManifest, BaseModelRef, CreateByInfo

            model_cfg = self.config.get("model", {})
            base_model = BaseModelRef(
                checkpoint_path=model_cfg.get("base_checkpoint", None),
                model_type=model_cfg.get("model_type", None),
                config_path=model_cfg.get("config_path", None),
            )
            manifest = AdapterManifest(
                model_family=self.model_name,
                base_model=base_model,
                peft_files=peft_files,
                created_by=CreateByInfo(job_id=self.job_id, timestamp=datetime.now().isoformat()),
                checksums=None,
            )
            manifest_path = adapter_dir / "adapter.manifest.json"
            manifest.save(manifest_path)
            logger.info("✓ Saved LoRA adapters to: %s", adapter_dir)
            return manifest_path
        else:
            logger.warning("%s does not support save_lora_adapters", self.model_name)
            return None

    def get_model(self) -> nn.Module:
        """Return the underlying model."""
        return self.model

    def get_checkpoint_manager(self) -> CheckpointManager:
        """Return the checkpoint manager."""
        return self.checkpoint_manager

    @property
    def should_stop(self) -> bool:
        """Check if early stopping was triggered."""
        return self.checkpoint_manager.should_stop

    def close(self) -> None:
        """Cleanup resources."""
        self.writer.close()
        logger.info("✓ %s trainer closed", self.model_name)
