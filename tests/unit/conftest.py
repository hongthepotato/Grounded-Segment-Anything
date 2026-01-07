"""
Pytest configuration and shared fixtures.

This module provides shared test fixtures and utilities used across all tests.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Temporary directory path
        
    Cleanup:
        Automatically removed after test completes
    """
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide a sample configuration dictionary for testing.
    
    Returns:
        Sample configuration matching typical training config structure
    """
    return {
        'learning_rate': 1e-4,
        'batch_size': 8,
        'epochs': 50,
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'lora': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['value_proj', 'output_proj']
        },
        'dataset': {
            'num_classes': 3,
            'class_names': ['background', 'cat', 'dog']
        }
    }


@pytest.fixture
def sample_dataset_info() -> Dict[str, Any]:
    """
    Provide sample dataset inspection info.
    
    Returns:
        Sample dataset info matching inspect_dataset() output
    """
    return {
        'num_classes': 3,
        'num_images': 100,
        'num_annotations': 350,
        'annotation_mode': 'DETECTION_AND_SEGMENTATION',
        'has_boxes': True,
        'has_masks': True,
        'class_mapping': {
            '0': 'background',
            '1': 'cat',
            '2': 'dog'
        }
    }


@pytest.fixture
def mock_logger(monkeypatch):
    """
    Mock logger to avoid actual file/console output during tests.
    
    Returns:
        Mock logger object that captures log calls
    """
    import logging
    from unittest.mock import MagicMock

    mock = MagicMock(spec=logging.Logger)
    mock.info = MagicMock()
    mock.warning = MagicMock()
    mock.error = MagicMock()
    mock.debug = MagicMock()

    return mock


# Marker registration
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
