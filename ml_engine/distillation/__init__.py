"""
Knowledge distillation pipeline.

Converts fine-tuned teacher knowledge into prompt-free student models
through offline pseudo-labeling and ultralytics training.
"""

from ml_engine.distillation.pseudo_label import generate_pseudo_labels
from ml_engine.distillation.student_trainer import StudentTrainer
from ml_engine.distillation.utils import convert_coco_to_yolo_seg, merge_coco_datasets

__all__ = [
    "generate_pseudo_labels",
    "merge_coco_datasets",
    "convert_coco_to_yolo_seg",
    "StudentTrainer",
]
