"""
Configuration management utilities.

This module provides functions for loading, saving, and generating
configuration files with automatic merging of defaults and user overrides.
"""
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import yaml
from deprecated import deprecated

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = load_config('configs/defaults/teacher_grounding_dino_lora.yaml')
        >>> print(config['learning_rate'])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from: %s", config_path)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    
    Example:
        >>> config = {'learning_rate': 1e-4, 'batch_size': 8}
        >>> save_config(config, 'experiments/exp1/teacher_config.yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved config to: %s", output_path)


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Dictionary containing JSON data
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save data to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@deprecated(reason="Use load_config, merge_configs instead")
def generate_config(
    default_config_path: str,
    dataset_info: Dict[str, Any],
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate configuration by merging defaults with dataset info and CLI overrides.
    
    This is the core of the auto-config system. It:
    1. Loads default config template
    2. Auto-fills dataset-specific values (num_classes, class_names)
    3. Applies CLI overrides
    
    Args:
        default_config_path: Path to default config YAML
        dataset_info: Dictionary from inspect_dataset()
        cli_overrides: Optional dictionary of CLI overrides
    
    Returns:
        Generated configuration dictionary
    
    Example:
        >>> from ml_engine.data.inspection import inspect_dataset
        >>> dataset_info = inspect_dataset(coco_data)
        >>> config = generate_config(
        >>>     default_config_path='configs/defaults/teacher_grounding_dino_lora.yaml',
        >>>     dataset_info=dataset_info,
        >>>     cli_overrides={'batch_size': 16}
        >>> )
    """
    # Load default config
    config = load_config(default_config_path)

    # Auto-fill dataset-specific values
    config['num_classes'] = dataset_info['num_classes']
    config['class_names'] = list(dataset_info['class_mapping'].values())
    config['class_mapping'] = dataset_info['class_mapping']

    # Add dataset metadata
    if 'dataset' not in config:
        config['dataset'] = {}

    config['dataset']['num_images'] = dataset_info['num_images']
    config['dataset']['num_annotations'] = dataset_info['num_annotations']
    config['dataset']['annotation_mode'] = dataset_info['annotation_mode']
    config['dataset']['has_boxes'] = dataset_info['has_boxes']
    config['dataset']['has_masks'] = dataset_info['has_masks']

    # Apply CLI overrides
    if cli_overrides:
        config = merge_configs(config, cli_overrides)
        logger.info("Applied %d CLI overrides", len(cli_overrides))

    # Add generation metadata
    if 'metadata' not in config:
        config['metadata'] = {}

    config['metadata']['generated_at'] = datetime.now().isoformat()
    config['metadata']['default_config'] = str(default_config_path)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    
    Returns:
        Merged configuration
    
    Example:
        >>> base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> override = {'b': {'d': 4}, 'e': 5}
        >>> merged = merge_configs(base, override)
        >>> # Result: {'a': 1, 'b': {'c': 2, 'd': 4}, 'e': 5}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def parse_cli_overrides(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse CLI arguments into nested config dictionary.
    
    Converts dot-notation arguments like 'lora.r=32' into nested dict.
    
    Args:
        args_dict: Dictionary of CLI arguments
    
    Returns:
        Nested configuration dictionary
    
    Example:
        >>> args = {'lora.r': 32, 'lora.alpha': 64, 'batch_size': 16}
        >>> config = parse_cli_overrides(args)
        >>> # Result: {'lora': {'r': 32, 'alpha': 64}, 'batch_size': 16}
    """
    result = {}
    
    for key, value in args_dict.items():
        if '.' in key:
            # Nested key
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Top-level key
            result[key] = value
    
    return result


def create_experiment_dir(
    base_dir: str = 'experiments',
    experiment_name: Optional[str] = None
) -> Path:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name (default: exp_{timestamp})
    
    Returns:
        Path to created experiment directory
    
    Example:
        >>> exp_dir = create_experiment_dir(experiment_name='bag_detection')
        >>> print(exp_dir)  # experiments/bag_detection_20240115_143022
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'exp_{timestamp}'
    else:
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'{experiment_name}_{timestamp}'
    
    exp_dir = base_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'teachers').mkdir(exist_ok=True)
    (exp_dir / 'student').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def save_experiment_metadata(
    exp_dir: Path,
    dataset_info: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment metadata for reproducibility.
    
    Args:
        exp_dir: Experiment directory path
        dataset_info: Dataset inspection info
        cli_args: Optional CLI arguments used
    
    Example:
        >>> save_experiment_metadata(
        >>>     exp_dir=Path('experiments/exp1'),
        >>>     dataset_info=inspect_dataset(coco_data),
        >>>     cli_args={'batch_size': 16, 'epochs': 100}
        >>> )
    """
    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset': {
            'num_classes': dataset_info['num_classes'],
            'num_images': dataset_info['num_images'],
            'num_annotations': dataset_info['num_annotations'],
            'annotation_mode': dataset_info['annotation_mode'],
            'class_mapping': dataset_info['class_mapping']
        }
    }

    if cli_args:
        metadata['cli_args'] = cli_args

    metadata_path = exp_dir / 'metadata.json'
    save_json(metadata, str(metadata_path))
    logger.info("Saved experiment metadata to: %s", metadata_path)


def load_experiment_config(
    exp_dir: str,
    config_name: str = 'teacher_config.yaml'
) -> Dict[str, Any]:
    """
    Load configuration from experiment directory.
    
    Args:
        exp_dir: Experiment directory path
        config_name: Name of config file to load
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(exp_dir) / config_name
    return load_config(str(config_path))
