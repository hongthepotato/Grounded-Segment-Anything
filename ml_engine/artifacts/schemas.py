"""
Metadata need to find the proper LoRA adapter.

They will be saved after the teacher training is complete.

The directory structure will be like ():
experiments/{experiment_name}/teachers/
  ├── bundle.manifest.json
  ├── grounding_dino/lora_adapters/
  │   ├── adapter.manifest.json
  │   ├── adapter_config.json
  │   └── adapter_model
  └── sam/lora_adapters/
      ├── adapter.manifest.json
      ├── adapter_model.json
      ├── best.pth
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path
import json


@dataclass
class BaseModelRef:
    """Reference to a base model"""
    checkpoint_path: str
    model_type: str
    # sha256: Optional[str] = None

@dataclass
class CreateByInfo:
    """Information about who created the artifact"""
    job_id: str
    timestamp: str

@dataclass
class AdapterManifest:
    """Metadata for a single LoRA adapter"""
    model_family: str           # sam | grounding_dino
    base_model: BaseModelRef    # checkpoint path + model_type + optional sha256
    peft_files: Dict[str, str]  # {"config": "adapter_config.json", "weights": "adapter_model.safetensors"}
    created_by: CreateByInfo    # job_id + timestamp
    checksums: Optional[Dict[str, str]] = None   # file -> sha256

    def save(self, path: Path) -> None:
        """Save the manifest to a JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AdapterManifest":
        """Load the manifest from a JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["base_model"] = BaseModelRef(**data["base_model"])
            data["created_by"] = CreateByInfo(**data["created_by"])
            return cls(**data)


@dataclass
class BundleManifest:
    """Metadata for a complete teacher training job production"""
    bundle_type: str                        # "teacher_training_output"
    artifacts: Dict[str, str]               # model_name -> relative path (from where bundle.manifest.json is located) of the corresponding adapter.manifest.json
    lineage: Dict[str, str]                 # "job_id"
    merged_checkpoints: Optional[Dict[str, str]] = None      # model_naem -> relative path of merged model

    def save(self, path: Path) -> None:
        """Save the manifest to a JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BundleManifest":
        """Load the manifest from a JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
