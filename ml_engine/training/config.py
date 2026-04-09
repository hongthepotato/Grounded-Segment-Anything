"""
Teacher training config builder.

Extracted from TeacherTrainingHandler._build_config so TrialRunner
can build configs the exact same way without duplicating logic.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _expand_dotted_keys(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand dotted-key overrides into nested dicts for merge_configs.

    merge_configs does a recursive nested-dict merge. It does NOT understand
    dotted-key notation. Without expansion, {"lora.r": 32} becomes a literal
    top-level key "lora.r" instead of config["lora"]["r"] = 32.

    This is the missing step between the ConfigGuard/SimpleMutator dotted-key
    convention and the existing merge_configs implementation.

    Non-dotted keys (including nested dicts) pass through unchanged so the
    function is safe for both experiment-loop overrides (dotted) and normal
    job-config overrides (nested dict, no dots in top-level keys).

    Examples::

        _expand_dotted_keys({"lora.r": 32, "training.batch_size": 8})
        # -> {"lora": {"r": 32}, "training": {"batch_size": 8}}

        _expand_dotted_keys({"learning_rate": 1e-4})  # no dot
        # -> {"learning_rate": 1e-4}  # unchanged
    """
    result: Dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        else:
            result[key] = value
    return result


def build_teacher_training_config(
    data_manager,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a complete teacher training config from YAML defaults + overrides.

    This is the single source of truth for config construction. Both
    TeacherTrainingHandler and TrialRunner call this function so the
    configs are always built the same way.

    Args:
        data_manager: DataManager instance with dataset info.
        overrides: Dict of config overrides, deep-merged over YAML defaults.
            Comes from job_config["training"] for normal jobs, or from
            propose_fn() output for experiment-loop trials.

    Returns:
        Complete training configuration dict.
    """
    from core.config import load_config, merge_configs
    from core.constants import DEFAULT_CONFIGS_DIR

    # Load shared training defaults
    shared_config = load_config(str(DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'))
    logger.info("Loaded teacher_training.yaml defaults")

    # Load model-specific configs based on what the dataset needs
    dataset_info = data_manager.get_dataset_info()
    required_models = data_manager.get_required_models()
    logger.info("Required teacher models: %s", required_models)

    model_configs: Dict[str, Any] = {}
    if 'grounding_dino' in required_models:
        model_configs['grounding_dino'] = load_config(
            str(DEFAULT_CONFIGS_DIR / 'teacher_grounding_dino_lora.yaml')
        )
        logger.info("Loaded teacher_grounding_dino_lora.yaml")

    if 'sam' in required_models:
        model_configs['sam'] = load_config(
            str(DEFAULT_CONFIGS_DIR / 'teacher_sam_lora.yaml')
        )
        logger.info("Loaded teacher_sam_lora.yaml")

    if not model_configs:
        raise ValueError("No models to train — dataset has no valid annotations.")

    config: Dict[str, Any] = {
        **shared_config['training'],
        'num_classes': dataset_info['num_classes'],
        'class_names': list(dataset_info['class_mapping'].values()),
        'class_mapping': dataset_info['class_mapping'],
        'augmentation': shared_config.get('augmentation'),
        'evaluation': shared_config.get('evaluation'),
        'checkpointing': shared_config.get('checkpointing'),
        'models': model_configs,
    }

    if overrides:
        # Expand dotted-key notation (e.g. "lora.r" -> {"lora": {"r": ...}})
        # before handing to merge_configs, which only understands nested dicts.
        expanded = _expand_dotted_keys(overrides)
        config = merge_configs(config, expanded)
        logger.info("Applied config overrides: %s", list(overrides.keys()))

    return config
