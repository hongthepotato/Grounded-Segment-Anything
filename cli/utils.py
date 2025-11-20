"""
Utility functions for CLI scripts.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_file_exists(file_path: str, description: str = "File") -> Path:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        description: Description of file for error message
    
    Returns:
        Path object
    
    Raises:
        SystemExit: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"{description} not found: {file_path}")
        sys.exit(1)
    return path


def validate_dir_exists(dir_path: str, description: str = "Directory") -> Path:
    """
    Validate that a directory exists.
    
    Args:
        dir_path: Path to directory
        description: Description for error message
    
    Returns:
        Path object
    
    Raises:
        SystemExit: If directory doesn't exist
    """
    path = Path(dir_path)
    if not path.exists():
        logger.error(f"{description} not found: {dir_path}")
        sys.exit(1)
    return path


def setup_cuda_device(gpu_id: int) -> None:
    """
    Setup CUDA device.
    
    Args:
        gpu_id: GPU ID to use
    """
    import os
    import torch
    
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if torch.cuda.is_available():
            logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")
    else:
        logger.info("Using CPU (GPU disabled)")


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """Print a section separator."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def confirm_action(message: str, default: bool = True) -> bool:
    """
    Ask user to confirm an action.
    
    Args:
        message: Confirmation message
        default: Default choice
    
    Returns:
        True if confirmed, False otherwise
    """
    choices = "Y/n" if default else "y/N"
    choice = input(f"{message} [{choices}]: ").lower().strip()
    
    if not choice:
        return default
    
    return choice in ['y', 'yes']


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def parse_key_value_args(args: list) -> Dict[str, Any]:
    """
    Parse key=value arguments into dictionary.
    
    Args:
        args: List of strings in format "key=value"
    
    Returns:
        Dictionary of parsed arguments
    
    Example:
        >>> parse_key_value_args(['lora.r=32', 'batch_size=16'])
        {'lora.r': 32, 'batch_size': 16}
    """
    result = {}
    for arg in args:
        if '=' not in arg:
            logger.warning(f"Ignoring invalid argument (no '='): {arg}")
            continue
        
        key, value = arg.split('=', 1)
        
        # Try to convert to appropriate type
        try:
            # Try int
            value = int(value)
        except ValueError:
            try:
                # Try float
                value = float(value)
            except ValueError:
                # Try bool
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Keep as string otherwise
        
        result[key] = value
    
    return result


