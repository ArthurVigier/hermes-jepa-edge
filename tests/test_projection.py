"""
tests/test_projection.py
Unit tests for LeWM → JEPA projection layer and alignment loss.
"""

import pytest
import torch

from src.adapters.projection import (
    LeWMProjection,
    alignment_loss,
    build_projection_from_state_dict,
)

LEWM_DIM = 256
VJEPA2_DIM = 1024
BATCH = 8


# ─── LeWMProjection ───────────────────────────────────────────────────────────

def test_projection_output_shape_with_hidden():
    proj = LeWMProjection(LEWM_DIM, VJEPA2_DIM, hidden_dim=512)
    x = torch.randn(BATCH, LEWM_DIM)
    out = proj(x)
    assert out.shape == (BATCH, VJEPA2_DIM)


def test_projection_output_shape_direct_linear():
    proj = LeWMProjection(LEWM_DIM, VJEPA2_DIM, hidden_dim=0)
    x = torch.randn(BATCH, LEWM_DIM)
    out = proj(x)
    assert out.shape == (BATCH, VJEPA2_DIM)


def test_projection_no_nan():
    proj = LeWMProjection(LEWM_DIM, VJEPA2_DIM)
    x = torch.randn(BATCH, LEWM_DIM)
    out = proj(x)
    assert not torch.isnan(out).any()


def test_projection_gradients_flow():
    proj = LeWMProjection(LEWM_DIM, VJEPA2_DIM)
    x = torch.randn(BATCH, LEWM_DIM, requires_grad=True)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_build_projection_from_state_dict_mlp_checkpoint():
    original = LeWMProjection(LEWM_DIM, VJEPA2_DIM, hidden_dim=512)
    rebuilt = build_projection_from_state_dict(original.state_dict())
    x = torch.randn(BATCH, LEWM_DIM)
    assert rebuilt(x).shape == (BATCH, VJEPA2_DIM)


def test_build_projection_from_state_dict_linear_checkpoint():
    original = LeWMProjection(LEWM_DIM, VJEPA2_DIM, hidden_dim=0)
    rebuilt = build_projection_from_state_dict(original.state_dict())
    x = torch.randn(BATCH, LEWM_DIM)
    assert rebuilt(x).shape == (BATCH, VJEPA2_DIM)


# ─── Alignment loss ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["l2_normalized", "smooth_l1", "cosine"])
def test_alignment_loss_modes(mode):
    projected = torch.randn(BATCH, VJEPA2_DIM)
    target = torch.randn(BATCH, VJEPA2_DIM)
    loss = alignment_loss(projected, target, mode=mode)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_alignment_loss_zero_for_identical():
    """Perfect alignment should yield ~0 loss."""
    x = torch.randn(BATCH, VJEPA2_DIM)
    loss = alignment_loss(x, x, mode="l2_normalized")
    assert loss.item() < 1e-5


def test_alignment_loss_target_no_grad():
    """Target must be stop-gradient — no grad should flow into it."""
    projected = torch.randn(BATCH, VJEPA2_DIM, requires_grad=True)
    target = torch.randn(BATCH, VJEPA2_DIM, requires_grad=True)
    loss = alignment_loss(projected, target)
    loss.backward()
    # Projected should have grad, target should NOT (detached inside loss)
    assert projected.grad is not None
    assert target.grad is None


def test_alignment_loss_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown loss mode"):
        alignment_loss(torch.randn(4, 64), torch.randn(4, 64), mode="invalid")
