"""Model-specific trainers for teacher models."""

from .base import BaseModelTrainer
from .grounding_dino import GroundingDINOTrainer
from .sam import SAMTrainer

__all__ = [
    "BaseModelTrainer",
    "GroundingDINOTrainer",
    "SAMTrainer",
]
