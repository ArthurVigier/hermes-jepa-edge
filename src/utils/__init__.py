"""
src/utils/__init__.py
Shared utilities: logging, seeding, VRAM budget estimation, kill criterion checker.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a rich-formatted logger."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)


# ─── Seeding ──────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── VRAM estimation ──────────────────────────────────────────────────────────

@dataclass
class VRAMBudget:
    param_gb: float
    grad_gb: float
    optimizer_gb: float
    activation_gb: float
    total_gb: float

    def __str__(self) -> str:
        return (
            f"VRAM Budget → params={self.param_gb:.2f}GB  grads={self.grad_gb:.2f}GB  "
            f"optim={self.optimizer_gb:.2f}GB  acts={self.activation_gb:.2f}GB  "
            f"TOTAL={self.total_gb:.2f}GB"
        )


def estimate_vram(
    model: torch.nn.Module,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype: torch.dtype = torch.float32,
    optimizer: str = "adamw",
    frozen: bool = False,
) -> VRAMBudget:
    """
    Estimate VRAM usage for a model before training.

    Args:
        model: PyTorch model.
        batch_size: Training batch size.
        seq_len: Sequence length (used for rough activation estimate).
        dtype: Parameter dtype (float32, float16, bfloat16).
        optimizer: 'adamw' (2x param states) or 'sgd' (1x).
        frozen: If True, no grad/optimizer memory allocated.

    Returns:
        VRAMBudget dataclass with per-component and total estimates.
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_gb = (n_params * bytes_per_param) / 1e9

    if frozen:
        grad_gb = 0.0
        optimizer_gb = 0.0
    else:
        grad_gb = (n_trainable * bytes_per_param) / 1e9
        optim_multiplier = 2 if optimizer == "adamw" else 1
        optimizer_gb = (n_trainable * 4 * optim_multiplier) / 1e9  # always fp32 states

    # Rough activation estimate: 2 bytes × batch × seq × hidden (heuristic)
    hidden = getattr(model.config, "hidden_size", 768) if hasattr(model, "config") else 768
    activation_gb = (2 * batch_size * seq_len * hidden) / 1e9

    total_gb = param_gb + grad_gb + optimizer_gb + activation_gb

    return VRAMBudget(
        param_gb=param_gb,
        grad_gb=grad_gb,
        optimizer_gb=optimizer_gb,
        activation_gb=activation_gb,
        total_gb=total_gb,
    )


def check_vram_feasibility(budget: VRAMBudget, available_gb: float = 24.0) -> bool:
    """Warn if estimated VRAM exceeds available GPU memory."""
    logger = get_logger(__name__)
    logger.info(str(budget))
    if budget.total_gb > available_gb * 0.90:
        logger.warning(
            f"⚠️  Estimated VRAM {budget.total_gb:.2f}GB exceeds "
            f"90% of available {available_gb:.0f}GB. Consider gradient checkpointing or smaller batch."
        )
        return False
    return True


# ─── Kill criterion checker ───────────────────────────────────────────────────

KILL_THRESHOLDS: dict[str, dict[str, float]] = {
    "phase0": {"cosine_sim": 0.3},       # below → kill
    "phase1": {"align_loss": 0.8},       # above → kill
    "phase2": {"task_success": 0.30},    # below → kill
    "phase3": {"latency_ms": 200.0},     # above → kill
}

KILL_DIRECTION: dict[str, str] = {
    "cosine_sim": "below",
    "align_loss": "above",
    "task_success": "below",
    "latency_ms": "above",
}


def check_kill_criterion(
    phase: str,
    metric_name: str,
    metric_value: float,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Check if a kill criterion is triggered.

    Args:
        phase: e.g. 'phase0', 'phase1', 'phase2', 'phase3'
        metric_name: e.g. 'cosine_sim', 'align_loss'
        metric_value: Observed value.
        logger: Optional logger.

    Returns:
        True if kill criterion triggered (should stop), False otherwise.
    """
    if logger is None:
        logger = get_logger(__name__)

    thresholds = KILL_THRESHOLDS.get(phase, {})
    if metric_name not in thresholds:
        logger.warning(f"No kill criterion defined for {phase}/{metric_name}")
        return False

    threshold = thresholds[metric_name]
    direction = KILL_DIRECTION.get(metric_name, "above")

    triggered = (
        metric_value < threshold if direction == "below"
        else metric_value > threshold
    )

    if triggered:
        logger.error(
            f"🔴 KILL CRITERION TRIGGERED — {phase} / {metric_name}: "
            f"{metric_value:.4f} {'<' if direction == 'below' else '>'} {threshold}"
        )
    else:
        logger.info(
            f"✅ Kill criterion OK — {phase} / {metric_name}: "
            f"{metric_value:.4f} (threshold {'>' if direction == 'below' else '<'} {threshold})"
        )

    return triggered


# ─── Step timer ───────────────────────────────────────────────────────────────

class StepTimer:
    """Context manager to time a training step and log GPU memory."""

    def __init__(self, step: int, log_every: int = 10, logger: Optional[logging.Logger] = None):
        self.step = step
        self.log_every = log_every
        self.logger = logger or get_logger(__name__)
        self._start: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self.step % self.log_every != 0:
            return
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        self.logger.debug(f"step={self.step}  time={elapsed_ms:.1f}ms  gpu_mem={mem_gb:.2f}GB")
