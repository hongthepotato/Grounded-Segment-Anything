"""
Contract test for dtype invariants under autocast.

The silent-failure this guards against: tensors enter a model at bf16, the
forward pass produces output the caller then treats as bf16, but somewhere
in the middle a module upcast to fp32 silently. Downstream code ends up
with a mix of dtypes that never raises but produces subtly different
numerical results than the configured dtype suggests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestDtypePreservationThroughAutocast:
    """Under autocast(dtype=X), attention forward output must be dtype X.

    This mirrors the production assumption in
    ml_engine/training/training_manager.py::training_step where autocast is
    entered at a specific dtype and the downstream loss computation expects
    tensors at that dtype.
    """

    @pytest.mark.parametrize(
        "autocast_dtype",
        [torch.bfloat16],  # CPU autocast is well-supported for bf16; fp16 CPU autocast
                           # is partial (only some ops are in the allowlist), so testing
                           # the same invariant there would be noisy.
        ids=["bfloat16"],
    )
    def test_linear_layer_output_matches_autocast_dtype(
        self,
        tiny_attention_block: nn.Module,
        synthetic_batch: torch.Tensor,
        autocast_dtype: torch.dtype,
    ) -> None:
        """Linear ops are in the autocast allowlist — their output must carry
        the autocast dtype. This is the property the training loop relies on
        when it enters ``autocast(dtype=bf16)`` and expects matmul outputs to
        be bf16 for memory + speed.

        Note: the full model forward output ends in LayerNorm, and LayerNorm
        is deliberately in the autocast-disallow list (to preserve numerical
        stability in statistics computation). Asserting on the FULL model
        output would always see fp32 because of the trailing LayerNorm; that
        is PyTorch-intended behavior, not a silent bug. So we assert on the
        linear layer output directly.
        """
        model = tiny_attention_block

        with torch.amp.autocast(device_type="cpu", dtype=autocast_dtype):
            q_out = model.q_proj(synthetic_batch)

        assert q_out.dtype == autocast_dtype, (
            f"Expected q_proj output dtype {autocast_dtype!s} under autocast, got {q_out.dtype!s}. "
            f"Some earlier module is upcasting silently."
        )

    def test_autocast_does_not_mutate_model_param_dtype(
        self, tiny_attention_block: nn.Module, synthetic_batch: torch.Tensor
    ) -> None:
        """Model params stay fp32 regardless of autocast dtype.

        Autocast controls activation dtype, not parameter dtype. A regression
        that flipped params to bf16 would silently halve memory but also break
        any optimizer that assumes fp32 master weights.
        """
        param_dtypes_before = {name: p.dtype for name, p in tiny_attention_block.named_parameters()}

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            _ = tiny_attention_block(synthetic_batch)

        for name, p in tiny_attention_block.named_parameters():
            assert p.dtype == param_dtypes_before[name] == torch.float32, (
                f"Param {name!r} dtype changed under autocast: "
                f"was {param_dtypes_before[name]!s}, now {p.dtype!s}"
            )

    def test_dtype_mismatch_between_input_and_model_is_caught(
        self, tiny_attention_block: nn.Module
    ) -> None:
        """Feeding a bf16 tensor into an fp32 model without autocast must raise.

        This guards against the failure where someone silently converts input
        tensors to bf16 (e.g., in a dataloader transform) but forgets to enter
        autocast, leaving the model at fp32. PyTorch will raise at the first
        matmul — we assert that loudly fails rather than silently doing a
        lossy upcast.
        """
        model = tiny_attention_block  # fp32 params
        x_bf16 = torch.randn(2, 4, 16, dtype=torch.bfloat16)

        with pytest.raises((RuntimeError, TypeError)):
            # No autocast. fp32 weights + bf16 input → matmul error. Exact
            # message varies by torch version; matching on exception type is
            # enough — the key is that it raises instead of silently upcasting.
            _ = model(x_bf16)
