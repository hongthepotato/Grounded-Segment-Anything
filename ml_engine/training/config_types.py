"""
Typed configuration dataclasses for the teacher training pipeline.

Two insertion points:
  1. _build_config() in teacher.py returns TeacherTrainingConfig
  2. _init_trainers() in trainer.py passes GroundingDINOConfig/SAMConfig to model trainers

YAML files (configs/defaults/) remain the source of truth for values.
These types document and enforce the shape of what those files produce after merging.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union


class ConfigurationError(Exception):
    """Raised when a config field is missing, wrong type, or incompatible with an external API."""
    pass


@dataclass(frozen=True)
class LoopConfig:
    """Shared training loop hyperparameters (same for all teacher models). Immutable."""
    batch_size: int = 1
    epochs: int = 5
    num_workers: int = 4
    optimizer: str = "AdamW"
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    warmup_ratio: float = 0.1


@dataclass
class LoraConfig:
    """
    LoRA adapter configuration.

    Field names match peft.LoraConfig exactly (except `enabled`).
    Use .to_peft_dict() at the peft call site — it strips `enabled`.
    """
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    enabled: bool = True

    def to_peft_dict(self) -> Dict[str, Any]:
        """Return only the fields peft.LoraConfig accepts (drops 'enabled')."""
        d = dataclasses.asdict(self)
        d.pop("enabled")
        return d


@dataclass
class GroundingDINOConfig:
    """Per-model config for GroundingDINO trainer."""
    lora: LoraConfig
    learning_rate: float
    base_checkpoint: str = "data/models/pretrained/groundingdino_swint_ogc.pth"
    config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    freeze_backbone: bool = True
    freeze_bbox_embed: bool = False
    bert_model_path: Optional[str] = "data/models/pretrained/bert-base-uncased"
    momentum: float = 0.9           # used if loop.optimizer == 'SGD'
    evaluation_metric: str = "mAP50"


@dataclass
class SAMConfig:
    """Per-model config for SAM trainer."""
    lora: LoraConfig
    learning_rate: float
    base_checkpoint: str = "data/models/pretrained/sam_hq_vit_b.pth"
    model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_b"
    image_encoder_mode: Literal["frozen", "lora", "full"] = "lora"
    prompt_encoder_mode: Literal["frozen", "lora", "full"] = "frozen"
    mask_decoder_mode: Literal["frozen", "lora", "full"] = "full"
    mask_decoder_lr_multiplier: float = 0.1
    prompt_type: str = "boxes"
    multimask_output: bool = True
    single_object_sampling: bool = True
    evaluation_metric: str = "mask_IoU"


ModelConfig = Union[GroundingDINOConfig, SAMConfig]


@dataclass
class TeacherTrainingConfig:
    """
    Root config for a teacher training job.

    Constructed once in TeacherTrainingHandler._build_config().
    Passed to Trainer.__init__().

    augmentation / evaluation / checkpointing stay as Dict for now —
    they are consumed whole and their YAML comments serve as documentation.
    """
    loop: LoopConfig                    # frozen — immutable after construction
    num_classes: int
    class_names: List[str]
    class_mapping: Dict[int, str]
    models: Dict[str, ModelConfig]      # key: 'grounding_dino' | 'sam'
    augmentation: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    checkpointing: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to plain dict for save_config() / YAML serialization.

        Note: the saved YAML cannot be re-parsed back into typed objects —
        it is for human inspection and experiment reproducibility only.
        """
        return dataclasses.asdict(self)
