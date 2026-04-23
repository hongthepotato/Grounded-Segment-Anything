"""
Shared fixtures for contract tests.

Contract tests assert ML-specific invariants (finite loss, gradient presence,
dtype preservation, LoRA adapter vs. base-model freeze correctness) that
generic unit tests miss. Fixtures here keep each invariant test at CPU-friendly
sizes.

The attention block below deliberately mirrors the Q/K/V projection structure
of the production transformer blocks inside GroundingDINO and SAM, so a LoRA
wrapper applied here exercises the same PEFT target-module logic. This is the
closest we can get to production wiring without loading real SAM weights
(which require a multi-GB checkpoint and GPU).
"""

from __future__ import annotations

from typing import Callable, Dict

import pytest
import torch
import torch.nn as nn

# ============================================================================
# Tiny attention block — mirrors GDINO / SAM attention wiring at CPU-friendly
# size. Has named projection modules (q_proj, v_proj, out_proj) so PEFT LoRA
# with target_modules=['q_proj', 'v_proj'] attaches exactly as in production.
# ============================================================================


class TinyAttentionBlock(nn.Module):
    """Single-head attention + MLP, tiny enough for CPU tests.

    Layer layout matches the shape PEFT LoRA expects for production attention
    target_modules: separate q_proj, k_proj, v_proj, out_proj modules.
    """

    def __init__(self, hidden_dim: int = 16, mlp_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = float(self.hidden_dim) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        x = x + self.out_proj(attn_out)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


@pytest.fixture
def tiny_attention_block() -> TinyAttentionBlock:
    """Fresh TinyAttentionBlock per test."""
    torch.manual_seed(0)
    return TinyAttentionBlock(hidden_dim=16, mlp_dim=32)


@pytest.fixture
def synthetic_batch() -> torch.Tensor:
    """A [batch=2, seq=4, dim=16] synthetic input."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 16)


@pytest.fixture
def synthetic_targets() -> torch.Tensor:
    """Target tensor for MSE loss on attention output."""
    torch.manual_seed(43)
    return torch.randn(2, 4, 16)


# ============================================================================
# Loss helper — MSE between forward pass output and target.
# Shared by multiple invariant tests so gradient/loss asserts have one
# canonical compute path.
# ============================================================================


@pytest.fixture
def compute_loss_fn(synthetic_batch: torch.Tensor, synthetic_targets: torch.Tensor) -> Callable[[nn.Module], Dict[str, torch.Tensor]]:
    """Returns a function that runs model(batch) and computes MSE to targets."""

    def _compute(model: nn.Module) -> Dict[str, torch.Tensor]:
        out = model(synthetic_batch)
        loss = torch.nn.functional.mse_loss(out, synthetic_targets)
        return {"loss": loss, "output": out}

    return _compute


# ============================================================================
# LoRA wrapper — uses PEFT exactly the way production calls it on attention
# blocks, so wiring bugs (wrong target_modules, forgotten freeze) surface.
# ============================================================================


@pytest.fixture
def lora_wrapped_model(tiny_attention_block: TinyAttentionBlock) -> nn.Module:
    """Wrap the attention block with PEFT LoRA on q_proj + v_proj.

    Matches production configuration for SAM-style attention: LoRA adapts
    the query + value projections while key + out_proj stay frozen base.
    """
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    wrapped = get_peft_model(tiny_attention_block, config)
    return wrapped
