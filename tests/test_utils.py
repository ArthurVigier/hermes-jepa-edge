"""
tests/test_utils.py
Unit tests for shared utilities.
"""

import pytest
import torch
import torch.nn as nn

from src.utils import (
    VRAMBudget,
    check_kill_criterion,
    estimate_vram,
    seed_everything,
)


# ─── Seeding ──────────────────────────────────────────────────────────────────

def test_seed_everything_reproducible():
    seed_everything(42)
    t1 = torch.randn(10)
    seed_everything(42)
    t2 = torch.randn(10)
    assert torch.allclose(t1, t2)


def test_seed_different_seeds():
    seed_everything(42)
    t1 = torch.randn(10)
    seed_everything(99)
    t2 = torch.randn(10)
    assert not torch.allclose(t1, t2)


# ─── VRAM estimation ──────────────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)

    def forward(self, x):
        return self.linear(x)


def test_estimate_vram_returns_budget():
    model = TinyModel()
    budget = estimate_vram(model, batch_size=4, dtype=torch.float32)
    assert isinstance(budget, VRAMBudget)
    assert budget.total_gb > 0
    assert budget.param_gb > 0


def test_estimate_vram_frozen_no_grad():
    model = TinyModel()
    budget_train = estimate_vram(model, frozen=False)
    budget_frozen = estimate_vram(model, frozen=True)
    assert budget_frozen.grad_gb == 0.0
    assert budget_frozen.optimizer_gb == 0.0
    assert budget_frozen.total_gb < budget_train.total_gb


def test_estimate_vram_fp16_smaller():
    model = TinyModel()
    budget_fp32 = estimate_vram(model, dtype=torch.float32)
    budget_fp16 = estimate_vram(model, dtype=torch.float16)
    assert budget_fp16.param_gb < budget_fp32.param_gb


# ─── Kill criterion ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase,metric,value,expected_kill", [
    # phase0: cosine_sim < 0.3 → kill
    ("phase0", "cosine_sim", 0.1,  True),
    ("phase0", "cosine_sim", 0.5,  False),
    ("phase0", "cosine_sim", 0.3,  False),  # exactly at threshold → no kill
    # phase1: align_loss > 0.8 → kill
    ("phase1", "align_loss", 0.9,  True),
    ("phase1", "align_loss", 0.5,  False),
    # phase2: task_success < 0.30 → kill
    ("phase2", "task_success", 0.1, True),
    ("phase2", "task_success", 0.5, False),
    # phase3: latency_ms > 200 → kill
    ("phase3", "latency_ms", 250.0, True),
    ("phase3", "latency_ms", 100.0, False),
])
def test_kill_criterion(phase, metric, value, expected_kill):
    result = check_kill_criterion(phase, metric, value)
    assert result == expected_kill


def test_kill_criterion_unknown_metric_no_kill():
    # Unknown metric should not trigger a kill
    result = check_kill_criterion("phase0", "nonexistent_metric", 0.0)
    assert result is False
