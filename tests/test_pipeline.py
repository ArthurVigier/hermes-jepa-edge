"""
tests/test_pipeline.py
Unit tests for dual loss and action token formatting.
"""

import json

import pytest
import torch

from src.pipeline.hermes_vla import dual_loss, ROBOTIC_TOOLS


BATCH = 4
SEQ_LEN = 32
VOCAB = 1000
DIM = 64


# ─── Dual loss ────────────────────────────────────────────────────────────────

def _make_dual_loss_inputs():
    ntp_logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    ntp_labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    # Mask some positions with -100
    ntp_labels[:, -4:] = -100
    pred_emb = torch.randn(BATCH, DIM, requires_grad=True)
    tgt_emb = torch.randn(BATCH, DIM)
    return ntp_logits, ntp_labels, pred_emb, tgt_emb


def test_dual_loss_returns_scalar():
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    loss, metrics = dual_loss(ntp_logits, ntp_labels, pred_emb, tgt_emb)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_dual_loss_metrics_keys():
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    _, metrics = dual_loss(ntp_logits, ntp_labels, pred_emb, tgt_emb, training=False)
    assert "loss/total" in metrics
    assert "loss/ntp" in metrics
    assert "loss/jepa" in metrics


def test_dual_loss_grad_flows_to_pred():
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    loss, _ = dual_loss(ntp_logits, ntp_labels, pred_emb, tgt_emb, training=False)
    loss.backward()
    assert pred_emb.grad is not None
    assert not torch.isnan(pred_emb.grad).any()


def test_dual_loss_target_no_grad():
    """Target embedding must be stop-gradient."""
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    tgt_emb.requires_grad_(True)
    loss, _ = dual_loss(ntp_logits, ntp_labels, pred_emb, tgt_emb, training=False)
    loss.backward()
    assert tgt_emb.grad is None


def test_dual_loss_lambda_weights():
    """Setting lambda_jepa=0 should make total == ntp."""
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    total, metrics = dual_loss(
        ntp_logits, ntp_labels, pred_emb, tgt_emb,
        lambda_ntp=1.0, lambda_jepa=0.0, training=False
    )
    assert abs(total.item() - metrics["loss/ntp"]) < 1e-5


def test_dual_loss_jepa_dropout_eval():
    """In eval mode (training=False), JEPA loss is always computed."""
    ntp_logits, ntp_labels, pred_emb, tgt_emb = _make_dual_loss_inputs()
    _, metrics = dual_loss(
        ntp_logits, ntp_labels, pred_emb, tgt_emb,
        training=False, jepa_loss_dropout=1.0  # Would always drop in training
    )
    assert metrics["loss/jepa"] > 0.0


# ─── Action schema ────────────────────────────────────────────────────────────

def test_robotic_tools_valid_schema():
    """All tools must have name + description + parameters."""
    for tool in ROBOTIC_TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert "properties" in fn["parameters"]


def test_robotic_tools_json_serializable():
    """Tool schema must be JSON-serializable for Hermes template."""
    serialized = json.dumps(ROBOTIC_TOOLS)
    parsed = json.loads(serialized)
    assert len(parsed) == len(ROBOTIC_TOOLS)


@pytest.mark.parametrize("tool_name", ["move_arm", "grasp", "release", "query_world_state"])
def test_robotic_tools_all_present(tool_name):
    names = [t["function"]["name"] for t in ROBOTIC_TOOLS]
    assert tool_name in names
