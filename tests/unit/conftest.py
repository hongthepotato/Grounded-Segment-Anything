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
