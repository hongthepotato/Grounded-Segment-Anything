"""
Student model trainer using ultralytics.

Wraps ultralytics YOLO training with config from distillation.yaml,
mapping platform config keys to ultralytics train() arguments.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class StudentTrainer:
    """
    Trains a YOLOv8/YOLOv8-seg student model via ultralytics.

    Example:
        trainer = StudentTrainer(
            data_yaml="experiments/student/yolo_dataset/data.yaml",
            model_name="yolov8s-seg",
            config=distillation_cfg,
            output_dir="experiments/student/training",
        )
        best_pt = trainer.train()
    """

    def __init__(
        self,
        data_yaml: str,
        model_name: str,
        config: Dict[str, Any],
        output_dir: str,
    ):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_train_args(self) -> Dict[str, Any]:
        """Map platform config to ultralytics train() keyword arguments."""
        training = self.config.get("training", {})
        augmentation = self.config.get("augmentation", {})
        scheduler = self.config.get("scheduler", {})
        evaluation = self.config.get("evaluation", {})

        args: Dict[str, Any] = {
            "data": self.data_yaml,
            "project": str(self.output_dir),
            "name": "student",
            "exist_ok": True,
            "epochs": training.get("epochs", 300),
            "batch": training.get("batch_size", 32),
            "imgsz": training.get("imgsz", 640),
            "optimizer": training.get("optimizer", "SGD"),
            "lr0": training.get("learning_rate", 1e-3),
            "weight_decay": training.get("weight_decay", 5e-4),
            "momentum": training.get("momentum", 0.937),
            "workers": training.get("num_workers", 4),
            "warmup_epochs": scheduler.get("warmup_epochs", 3),
            "lrf": min(1.0, scheduler.get("min_lr", 1e-5) / max(training.get("learning_rate", 1e-3), 1e-8)),
            "val": True,
            "save": True,
            "save_period": evaluation.get("interval", 10),
            "verbose": True,
        }

        aug_keys = [
            "mosaic",
            "mixup",
            "copy_paste",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "flipud",
            "fliplr",
        ]
        for key in aug_keys:
            if key in augmentation:
                args[key] = augmentation[key]

        return args

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> tuple[str, Dict[str, float]]:
        """
        Train the student model.

        Args:
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function returning True to request cancellation

        Returns:
            Tuple of (absolute path to best.pt, final validation metrics dict).
            Metrics use the gate-compatible keys: mIoU (from mask mAP50),
            mAP50 (from box mAP50), plus raw ultralytics keys.
        """
        from ultralytics import YOLO

        from core.constants import PRETRAINED_MODELS_DIR

        pretrained = PRETRAINED_MODELS_DIR / f"{self.model_name}.pt"
        if not pretrained.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrained}. "
                f"Download {self.model_name}.pt to {PRETRAINED_MODELS_DIR}/ first."
            )
        logger.info("Loading pretrained student model: %s", pretrained)
        model = YOLO(str(pretrained))

        train_args = self._build_train_args()
        logger.info(
            "Starting student training: model=%s, epochs=%d, batch=%d",
            self.model_name,
            train_args["epochs"],
            train_args["batch"],
        )

        if progress_callback:

            def _on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                total = trainer.epochs
                metrics = {}
                if hasattr(trainer, "metrics") and trainer.metrics:
                    metrics = {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}
                progress_callback(
                    {
                        "current_epoch": epoch,
                        "total_epochs": total,
                        "train_metrics": metrics,
                        "message": f"Student epoch {epoch}/{total}",
                    }
                )

            model.add_callback("on_train_epoch_end", _on_train_epoch_end)

        if cancel_check:

            def _on_train_batch_end(trainer):
                if cancel_check():
                    logger.info("Cancel requested, stopping student training")
                    trainer.stop = True

            model.add_callback("on_train_batch_end", _on_train_batch_end)

        results = model.train(**train_args)

        best_pt = self.output_dir / "student" / "weights" / "best.pt"
        if not best_pt.exists():
            save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else self.output_dir
            best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(
                f"No best.pt produced at {best_pt}. "
                "Training may have been cancelled before the first saved checkpoint, "
                "or all epochs scored below the pre-trained baseline."
            )

        # Extract gate-compatible metrics from the final results dict.
        # Ultralytics key mapping:
        #   metrics/mAP50(M) — mask mAP50 from YOLOv8-seg → gate key "mIoU"
        #   metrics/mAP50(B) — box  mAP50 from YOLOv8     → gate key "mAP50"
        metrics: Dict[str, float] = {}
        if hasattr(results, "results_dict") and results.results_dict:
            raw = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
            metrics.update(raw)
            if "metrics/mAP50(M)" in raw:
                metrics["mIoU"] = raw["metrics/mAP50(M)"]
            if "metrics/mAP50(B)" in raw:
                metrics["mAP50"] = raw["metrics/mAP50(B)"]
        else:
            logger.warning(
                "results.results_dict absent or empty — gate will escalate "
                "(neither mIoU nor mAP50 extractable). "
                "results type: %s",
                type(results),
            )

        logger.info("Student training complete. Best weights: %s, metrics: %s", best_pt, metrics)
        return str(best_pt.resolve()), metrics
