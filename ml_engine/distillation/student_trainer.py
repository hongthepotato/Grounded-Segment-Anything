"""
Student model trainer using ultralytics.

Wraps ultralytics YOLO training with config from distillation.yaml,
mapping platform config keys to ultralytics train() arguments.

Ultralytics 8.4+ runs AMP self-tests using a hardcoded ``YOLO("yolo26n.pt")`` load
(``ultralytics.utils.checks.check_amp``), not your student weights. That can trigger a
GitHub download unless ``yolo26n.pt`` exists in the process CWD when the check runs.
We temporarily ``chdir`` to ``PRETRAINED_MODELS_DIR`` during ``model.train()`` so a
pre-placed ``data/models/pretrained/yolo26n.pt`` satisfies the check offline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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
        training = self.config.get('training', {})
        augmentation = self.config.get('augmentation', {})
        scheduler = self.config.get('scheduler', {})
        evaluation = self.config.get('evaluation', {})

        args: Dict[str, Any] = {
            'data': self.data_yaml,
            'project': str(self.output_dir),
            'name': 'student',
            'exist_ok': True,

            'epochs': training.get('epochs', 300),
            'batch': training.get('batch_size', 32),
            'imgsz': training.get('imgsz', 640),
            'optimizer': training.get('optimizer', 'SGD'),
            'lr0': training.get('learning_rate', 1e-3),
            'weight_decay': training.get('weight_decay', 5e-4),
            'momentum': training.get('momentum', 0.937),
            'workers': training.get('num_workers', 4),

            'warmup_epochs': scheduler.get('warmup_epochs', 3),
            'lrf': scheduler.get('min_lr', 1e-5) / max(training.get('learning_rate', 1e-3), 1e-8),

            'val': True,
            'save': True,
            'save_period': evaluation.get('interval', 10),

            'verbose': True,
        }

        if 'amp' in training:
            args['amp'] = training['amp']

        aug_keys = [
            'mosaic', 'mixup', 'copy_paste',
            'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale', 'shear',
            'flipud', 'fliplr',
        ]
        for key in aug_keys:
            if key in augmentation:
                args[key] = augmentation[key]

        return args

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> str:
        """
        Train the student model.

        Args:
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function returning True to request cancellation

        Returns:
            Absolute path to the best.pt weights file
        """
        from ultralytics import YOLO
        from core.constants import PRETRAINED_MODELS_DIR

        pretrained = PRETRAINED_MODELS_DIR / f'{self.model_name}.pt'
        if not pretrained.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrained}. "
                f"Download {self.model_name}.pt to {PRETRAINED_MODELS_DIR}/ first."
            )
        logger.info("Loading pretrained student model: %s", pretrained)
        model = YOLO(str(pretrained))

        train_args = self._build_train_args()
        train_args['data'] = str(Path(self.data_yaml).resolve())
        train_args['project'] = str(self.output_dir.resolve())
        logger.info("Starting student training: model=%s, epochs=%d, batch=%d",
                     self.model_name, train_args['epochs'], train_args['batch'])

        amp_probe = PRETRAINED_MODELS_DIR / 'yolo26n.pt'
        if not amp_probe.exists():
            logger.warning(
                "Optional %s missing: Ultralytics AMP check will try to download "
                "yolo26n.pt from GitHub (separate from %s). Place yolo26n.pt there to "
                "train fully offline, or set training.amp=false to disable AMP.",
                amp_probe,
                pretrained.name,
            )

        if progress_callback:
            def _on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                total = trainer.epochs
                metrics = {}
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    metrics = {k: float(v) for k, v in trainer.metrics.items()
                               if isinstance(v, (int, float))}
                progress_callback({
                    'current_epoch': epoch,
                    'total_epochs': total,
                    'train_metrics': metrics,
                    'message': f"Student epoch {epoch}/{total}",
                })

            model.add_callback('on_train_epoch_end', _on_train_epoch_end)

        if cancel_check:
            def _on_train_batch_end(trainer):
                if cancel_check():
                    logger.info("Cancel requested, stopping student training")
                    trainer.stop = True

            model.add_callback('on_train_batch_end', _on_train_batch_end)

        pt_dir = Path(PRETRAINED_MODELS_DIR).resolve()
        prev_cwd = Path.cwd()
        try:
            os.chdir(pt_dir)
            results = model.train(**train_args)
        finally:
            os.chdir(prev_cwd)

        best_pt = self.output_dir / 'student' / 'weights' / 'best.pt'
        if not best_pt.exists():
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else self.output_dir
            best_pt = save_dir / 'weights' / 'best.pt'

        logger.info("Student training complete. Best weights: %s", best_pt)
        return str(best_pt.resolve())
