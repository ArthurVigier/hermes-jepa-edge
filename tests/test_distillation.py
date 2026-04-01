"""
tests/test_distillation.py
Unit tests for knowledge distillation loss and trajectory parsing.
"""

import json

import pytest
import torch

from src.distillation.teacher_student import (
    _parse_teacher_response,
    knowledge_distillation_loss,
)


BATCH = 4
SEQ_LEN = 32
VOCAB = 1000


# ─── KD loss ──────────────────────────────────────────────────────────────────

def _make_kd_inputs():
    student_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
    teacher_logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels[:, -4:] = -100  # padding
    return student_logits, teacher_logits, labels


def test_kd_loss_returns_scalar():
    s, t, labels = _make_kd_inputs()
    loss, metrics = knowledge_distillation_loss(s, t, labels)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_kd_loss_metrics_present():
    s, t, labels = _make_kd_inputs()
    _, metrics = knowledge_distillation_loss(s, t, labels)
    assert "loss/total" in metrics
    assert "loss/ce" in metrics
    assert "loss/kd" in metrics


def test_kd_loss_teacher_no_grad():
    """Teacher logits must not receive gradients."""
    s, t, labels = _make_kd_inputs()
    t.requires_grad_(True)
    loss, _ = knowledge_distillation_loss(s, t, labels)
    loss.backward()
    assert t.grad is None


def test_kd_loss_alpha_ce_only():
    """alpha_kd=0 → total should equal CE loss only."""
    s, t, labels = _make_kd_inputs()
    total, metrics = knowledge_distillation_loss(s, t, labels, alpha_ce=1.0, alpha_kd=0.0)
    assert abs(total.item() - metrics["loss/ce"]) < 1e-5


def test_kd_loss_temperature_effect():
    """Higher temperature should produce softer distributions → different KD loss."""
    s, t, labels = _make_kd_inputs()
    _, m1 = knowledge_distillation_loss(s, t, labels, temperature=1.0)
    _, m2 = knowledge_distillation_loss(s, t, labels, temperature=4.0)
    # KD loss values should differ (not guaranteed equal)
    assert m1["loss/kd"] != m2["loss/kd"]


# ─── Response parsing ─────────────────────────────────────────────────────────

def test_parse_teacher_response_with_think_and_tools():
    response = (
        "<think>I need to move the arm then grasp.</think>\n"
        '<tool_call>{"name": "move_arm", "arguments": {"target_pose": [0.3, 0.1, 0.4, 0, 0, 0], "velocity": 0.5}}</tool_call>\n'
        '<tool_call>{"name": "grasp", "arguments": {"force_n": 10.0, "width_m": 0.05}}</tool_call>\n'
    )
    reasoning, actions = _parse_teacher_response(response)
    assert "move the arm" in reasoning
    assert len(actions) == 2
    assert actions[0]["name"] == "move_arm"
    assert actions[1]["name"] == "grasp"


def test_parse_teacher_response_no_think():
    response = '<tool_call>{"name": "release", "arguments": {"width_m": 0.1}}</tool_call>'
    reasoning, actions = _parse_teacher_response(response)
    assert reasoning == ""
    assert len(actions) == 1


def test_parse_teacher_response_malformed_tool_call():
    """Malformed JSON tool calls should be silently skipped."""
    response = '<tool_call>not valid json{{{</tool_call>'
    reasoning, actions = _parse_teacher_response(response)
    assert actions == []


def test_parse_teacher_response_empty():
    reasoning, actions = _parse_teacher_response("")
    assert reasoning == ""
    assert actions == []
