"""
src/adapters/projection.py
Phase 1 — Train projection layer LeWM → VLA-JEPA target space.

Stage 1: freeze LeWM, train projection only.
Stage 2: unfreeze LeWM, joint fine-tuning.

Kill criterion: align_loss > 0.8 after 500 steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils import (
    StepTimer,
    check_kill_criterion,
    check_vram_feasibility,
    estimate_vram,
    get_logger,
    seed_everything,
)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ProjectionConfig:
    lewm_checkpoint: str
    output_dir: str = "experiments/phase1"

    # Architecture
    lewm_dim: int = 256
    vjepa2_dim: int = 1024
    hidden_dim: int = 512             # Optional bottleneck; set to 0 for direct Linear
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    n_steps_frozen: int = 500         # Stage 1: projection only
    n_steps_joint: int = 1000         # Stage 2: projection + LeWM
    kill_check_step: int = 500        # When to evaluate kill criterion
    log_every: int = 25
    save_every: int = 250
    device: str = "cuda"
    seed: int = 42
    available_vram_gb: float = 24.0   # RTX 4090


# ─── Projection module ────────────────────────────────────────────────────────

class LeWMProjection(nn.Module):
    """
    Projects LeWM latents into VLA-JEPA target embedding space.

    Uses a 2-layer MLP with LayerNorm for stability.
    Set hidden_dim=0 for a single Linear layer.
    """

    def __init__(self, lewm_dim: int, vjepa2_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(lewm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, vjepa2_dim),
                nn.LayerNorm(vjepa2_dim),
            )
        else:
            self.net = nn.Linear(lewm_dim, vjepa2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, lewm_dim]
        Returns:
            [B, vjepa2_dim]
        """
        return self.net(x)


# ─── Alignment loss ───────────────────────────────────────────────────────────

def alignment_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    mode: str = "l2_normalized",
) -> torch.Tensor:
    """
    Compute alignment loss between projected LeWM embeddings and VLA-JEPA targets.

    Args:
        projected: [B, D] — output of LeWMProjection
        target: [B, D] — VLA-JEPA target embeddings (from V-JEPA2 EMA encoder)
        mode: 'l2_normalized' (cosine MSE) | 'smooth_l1' | 'cosine'

    Returns:
        Scalar loss.
    """
    projected_norm = F.normalize(projected, dim=-1)
    target_norm = F.normalize(target.detach(), dim=-1)  # targets are stop-gradient

    if mode == "l2_normalized":
        return F.mse_loss(projected_norm, target_norm)
    elif mode == "smooth_l1":
        return F.smooth_l1_loss(projected_norm, target_norm)
    elif mode == "cosine":
        return 1.0 - (projected_norm * target_norm).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


# ─── Training loop ────────────────────────────────────────────────────────────

class ProjectionTrainer:
    """Two-stage trainer for LeWM → JEPA projection."""

    def __init__(
        self,
        config: ProjectionConfig,
        lewm_encoder: nn.Module,
        dataloader: DataLoader,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = config
        self.logger = logger or get_logger(__name__)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(config.seed)

        self.lewm = lewm_encoder.to(self.device)
        self.projection = LeWMProjection(
            lewm_dim=config.lewm_dim,
            vjepa2_dim=config.vjepa2_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(self.device)

        self.step = 0
        self.best_loss = float("inf")

    def _make_optimizer(self, params, lr: float) -> AdamW:
        return AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)

    def _check_vram(self) -> None:
        budget = estimate_vram(
            self.projection,
            batch_size=self.cfg.batch_size,
            dtype=torch.float32,
            available_gb=self.cfg.available_vram_gb,
        )
        check_vram_feasibility(budget, available_gb=self.cfg.available_vram_gb)

    def _save_checkpoint(self, tag: str) -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "step": self.step,
            "projection_state": self.projection.state_dict(),
            "lewm_state": self.lewm.state_dict(),
            "config": self.cfg,
        }, path)
        self.logger.info(f"Saved checkpoint → {path}")

    def _run_stage(
        self,
        n_steps: int,
        freeze_lewm: bool,
        stage_name: str,
    ) -> dict[str, float]:
        """
        Run one training stage.

        Args:
            n_steps: Number of gradient steps.
            freeze_lewm: If True, only train projection.
            stage_name: Label for logging.

        Returns:
            Dict with 'final_loss' and 'kill_triggered'.
        """
        self.logger.info(f"\n{'═'*60}")
        self.logger.info(f"Stage: {stage_name}  |  freeze_lewm={freeze_lewm}  |  steps={n_steps}")

        # Freeze / unfreeze LeWM
        for p in self.lewm.parameters():
            p.requires_grad = not freeze_lewm

        trainable_params = (
            list(self.projection.parameters())
            if freeze_lewm
            else list(self.projection.parameters()) + list(self.lewm.parameters())
        )

        optimizer = self._make_optimizer(trainable_params, lr=self.cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=self.cfg.lr * 0.1)

        self._check_vram()

        losses: list[float] = []
        data_iter = iter(self.dataloader)
        kill_triggered = False

        for local_step in range(n_steps):
            self.step += 1

            # Get batch (cycle dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            frames = batch["frames"].to(self.device)
            targets = batch["vjepa2_targets"].to(self.device)

            with StepTimer(self.step, log_every=self.cfg.log_every, logger=self.logger):
                # Forward
                if freeze_lewm:
                    with torch.no_grad():
                        lewm_emb = self.lewm(frames)
                else:
                    lewm_emb = self.lewm(frames)

                projected = self.projection(lewm_emb)
                loss = alignment_loss(projected, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

            losses.append(loss.item())

            if self.step % self.cfg.log_every == 0:
                mean_loss = sum(losses[-self.cfg.log_every:]) / self.cfg.log_every
                self.logger.info(
                    f"step={self.step:5d}  loss={mean_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self._save_checkpoint("best")

            if self.step % self.cfg.save_every == 0:
                self._save_checkpoint(f"step{self.step}")

            # Kill criterion check at configured step
            if self.step == self.cfg.kill_check_step:
                mean_loss_at_check = sum(losses) / len(losses)
                kill_triggered = check_kill_criterion(
                    phase="phase1",
                    metric_name="align_loss",
                    metric_value=mean_loss_at_check,
                    logger=self.logger,
                )
                if kill_triggered:
                    self.logger.error(
                        "Phase 1 KILLED at kill_check_step.\n"
                        "LeWM cannot be aligned to JEPA target space with this projection.\n"
                        "Fallback: use V-JEPA2 encoder directly."
                    )
                    return {"final_loss": mean_loss_at_check, "kill_triggered": True}

        final_loss = sum(losses) / len(losses)
        return {"final_loss": final_loss, "kill_triggered": kill_triggered}

    def train(self) -> dict[str, float]:
        """Run both training stages. Returns summary metrics."""
        self.logger.info("Phase 1 — Projection training")

        # Stage 1: projection only
        result_s1 = self._run_stage(
            n_steps=self.cfg.n_steps_frozen,
            freeze_lewm=True,
            stage_name="Stage 1 — Projection only (LeWM frozen)",
        )
        if result_s1["kill_triggered"]:
            return result_s1

        # Stage 2: joint fine-tuning
        result_s2 = self._run_stage(
            n_steps=self.cfg.n_steps_joint,
            freeze_lewm=False,
            stage_name="Stage 2 — Joint fine-tuning (LeWM + Projection)",
        )

        self._save_checkpoint("final")
        self.logger.info(f"\nPhase 1 complete. Final loss: {result_s2['final_loss']:.4f}")
        return result_s2
