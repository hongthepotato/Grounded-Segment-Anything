"""
Repo-root pytest configuration.

Auto-skips ``@pytest.mark.gpu`` tests when CUDA is unavailable, so CI runners
without a GPU don't need per-suite skip wiring. Tests that legitimately need a
GPU should carry ``@pytest.mark.gpu`` explicitly.

The hook fires regardless of which test directory pytest collects from
(unit, integration, contract). Per-suite conftest.py files extend this with
their own fixtures but do not need to re-implement the GPU skip.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark tests with @pytest.mark.gpu to skip when CUDA is unavailable."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        # If torch import itself fails, treat as no-GPU. Tests that need torch
        # will fail loudly on their own; this only protects the gpu-marked
        # ones from running unconditionally.
        cuda_available = False

    if cuda_available:
        return

    skip_gpu = pytest.mark.skip(reason="CUDA not available; skipping @pytest.mark.gpu test")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
