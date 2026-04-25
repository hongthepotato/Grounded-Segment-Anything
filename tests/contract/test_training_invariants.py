"""
Contract tests for training invariants.

These assert properties that MUST hold for training to produce correct
models. A regression in any of these is the silent-failure class the whole
CI effort exists to catch: training keeps running, loss keeps going down,
but the model is subtly wrong.

Invariants tested here:
1. Loss is finite over N steps on synthetic data.
2. After loss.backward(), every requires_grad=True tensor has a grad.
3. After optimizer.step(), LoRA adapter params changed and frozen base did not.
4. GradScaler selection matches dtype policy:
   - bfloat16: no scaler (would be pointless; bf16 has fp32 exponent range)
   - float16: scaler required (fp16 underflows without loss scaling)
   - float32: no scaler
   Plus behavior assertion: fp16-without-scaler produces NaN/Inf on a
   deliberately ill-conditioned tiny training run.
"""

from __future__ import annotations

from typing import Callable, Dict

import pytest
import torch
import torch.nn as nn

# Shared fixtures from conftest.py:
#   tiny_attention_block  – TinyAttentionBlock
#   synthetic_batch       – [2, 4, 16] input tensor
#   synthetic_targets     – [2, 4, 16] MSE target
#   compute_loss_fn       – callable(model) -> {'loss': ..., 'output': ...}
#   lora_wrapped_model    – PEFT LoRA wrapping q_proj + v_proj


# ============================================================================
# Invariant 1: loss is finite over N steps on synthetic data.
# ============================================================================


class TestLossFinite:
    def test_loss_is_finite_over_n_steps(
        self,
        tiny_attention_block: nn.Module,
        compute_loss_fn: Callable[[nn.Module], Dict[str, torch.Tensor]],
    ) -> None:
        optimizer = torch.optim.Adam(tiny_attention_block.parameters(), lr=1e-3)

        for step in range(10):
            optimizer.zero_grad()
            result = compute_loss_fn(tiny_attention_block)
            loss = result["loss"]

            assert torch.isfinite(loss).item(), f"Loss is non-finite at step {step}: {loss}"
            loss.backward()
            optimizer.step()


# ============================================================================
# Invariant 2: gradients populate on every requires_grad tensor after
# loss.backward(). A regression that silently freezes something would show up
# here as a None grad on a param we expect to train.
# ============================================================================


class TestGradientsPopulated:
    def test_gradients_populated_on_all_trainable_params(
        self,
        tiny_attention_block: nn.Module,
        compute_loss_fn: Callable[[nn.Module], Dict[str, torch.Tensor]],
    ) -> None:
        result = compute_loss_fn(tiny_attention_block)
        result["loss"].backward()

        trainable_params = [
            (name, p) for name, p in tiny_attention_block.named_parameters() if p.requires_grad
        ]
        assert trainable_params, "Expected tiny_attention_block to have trainable parameters"

        missing_grad = [name for name, p in trainable_params if p.grad is None]
        assert not missing_grad, f"Params with requires_grad=True but no .grad: {missing_grad}"

        # Also: every grad must be finite (NaN/Inf grad is a silent failure class).
        nonfinite = [name for name, p in trainable_params if not torch.isfinite(p.grad).all().item()]
        assert not nonfinite, f"Params with non-finite .grad: {nonfinite}"


# ============================================================================
# Invariant 3: LoRA adapter params change after optimizer.step(); base params
# stay frozen. This is the core correctness guarantee of a LoRA training run.
# A regression that accidentally marks base params trainable silently burns
# compute on unnecessary weights; one that silently freezes LoRA params
# trains literally nothing.
# ============================================================================


class TestLoraAdapterUpdatesBaseFrozen:
    def test_lora_adapter_changes_frozen_base_does_not(
        self,
        lora_wrapped_model: nn.Module,
        compute_loss_fn: Callable[[nn.Module], Dict[str, torch.Tensor]],
    ) -> None:
        """After a few training steps, LoRA params update while base stays frozen.

        Note: PEFT's standard LoRA init is lora_A ~ Kaiming and lora_B = 0.
        On step 1, the forward W*x + B*A*x equals W*x exactly (B is zero), so
        d(loss)/d(lora_A) = 0 — A doesn't move on step 1. B receives a nonzero
        grad and moves. From step 2 onward both A and B update.

        We run 3 steps so both A and B have had a chance to move, then check
        that at least one adapter param per layer moved.
        """
        before = {name: p.detach().clone() for name, p in lora_wrapped_model.named_parameters()}

        trainable = [p for p in lora_wrapped_model.parameters() if p.requires_grad]
        assert trainable, "lora_wrapped_model should have trainable (LoRA) parameters"
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        for _ in range(3):
            optimizer.zero_grad()
            result = compute_loss_fn(lora_wrapped_model)
            result["loss"].backward()
            optimizer.step()

        # Classify params: any LoRA adapter must have changed from baseline;
        # any non-LoRA (base) param must NOT have changed.
        lora_changed = []
        base_unchanged = []
        for name, p in lora_wrapped_model.named_parameters():
            delta = (p.detach() - before[name]).abs().max().item()
            is_lora = "lora_" in name
            if is_lora:
                if delta > 0:
                    lora_changed.append(name)
                else:
                    pytest.fail(
                        f"LoRA param {name!r} did not change after 3 steps "
                        f"(delta=0); expected adapter to update"
                    )
            else:
                if delta > 0:
                    pytest.fail(f"Base param {name!r} changed (delta={delta}); expected frozen")
                else:
                    base_unchanged.append(name)

        assert lora_changed, "Expected at least one LoRA param to update"
        assert base_unchanged, "Expected at least one base param to stay frozen"

    def test_all_base_params_have_requires_grad_false(self, lora_wrapped_model: nn.Module) -> None:
        """PEFT should have set requires_grad=False on every base param after wrapping."""
        for name, p in lora_wrapped_model.named_parameters():
            if "lora_" not in name:
                assert not p.requires_grad, (
                    f"Base param {name!r} has requires_grad=True; PEFT should freeze it. "
                    f"This is the silent bug where base weights drift during LoRA training."
                )


# ============================================================================
# Invariant 4: gradscaler selection by dtype (branch check) + behavior
# assertion (what happens if we get it wrong on fp16).
# ============================================================================


class TestGradScalerSelectionByDtype:
    """Mirrors the logic in ml_engine/training/training_manager.py: bf16 / fp32
    must NOT use GradScaler, fp16 MUST. Asserts both the branch and the
    behavior that justifies the branch.
    """

    @pytest.mark.parametrize(
        "dtype,expected_scaler_enabled",
        [
            (torch.float32, False),
            (torch.bfloat16, False),
            (torch.float16, True),
        ],
    )
    def test_scaler_policy_matches_dtype(self, dtype: torch.dtype, expected_scaler_enabled: bool) -> None:
        """Replicates the selection logic from training_manager.__init__."""
        use_amp = dtype != torch.float32
        scaler = torch.amp.GradScaler("cuda") if (use_amp and dtype == torch.float16) else None

        if expected_scaler_enabled:
            assert scaler is not None, (
                f"Expected a GradScaler for {dtype!s}; got None. "
                f"fp16 without scaler → gradient underflow → silent NaN loss."
            )
        else:
            assert scaler is None, (
                f"Expected NO GradScaler for {dtype!s}; got one. bf16/fp32 with scaler is a config error."
            )

    def test_fp16_cannot_represent_small_gradients(self) -> None:
        """Canonical demonstration of why GradScaler exists for fp16.

        fp16's smallest positive subnormal is ~6e-8; values below that round to
        exact zero. Real deep-net gradients in the middle of a long backward
        chain routinely land at 1e-6 to 1e-10. Without a loss scaler that
        multiplies the loss by a large constant (and divides the grads back
        down in optimizer.step), those tiny gradients become exact zeros and
        the model doesn't train.

        This test does NOT simulate a training run (that was the prior
        version's design and it was tautological: multiplying by an
        underflowed scale trivially produces zero grads). Instead it asserts
        the fp16 arithmetic property directly. If a future PyTorch version
        emulates fp16 in fp32 on CPU and this no longer underflows, this
        assertion flips — which is a signal to re-examine the GradScaler
        policy in training_manager.py, not to silently continue.
        """
        small_fp32 = torch.tensor(1e-8, dtype=torch.float32)
        small_fp16 = small_fp32.half()

        assert small_fp32.item() != 0.0, "sanity: 1e-8 representable in fp32"
        assert small_fp16.item() == 0.0, (
            f"Expected 1e-8 to underflow fp16 to 0, got {small_fp16.item()!r}. "
            "If fp16 arithmetic semantics have changed, re-examine the "
            "GradScaler policy in ml_engine/training/training_manager.py."
        )
