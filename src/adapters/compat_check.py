"""
src/adapters/compat_check.py
Phase 0 — Verify LeWM embeddings are compatible with VLA-JEPA target space.

Kill criterion: cosine_sim < 0.3 after 100 steps → abort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import check_kill_criterion, get_logger, seed_everything


@dataclass
class CompatCheckConfig:
    lewm_checkpoint: str                        # Path to LeWM .pt checkpoint
    dataset_name: str = "libero"                # libero | droid | habitat
    n_steps: int = 100                          # Steps before kill criterion check
    batch_size: int = 4
    device: str = "cuda"
    seed: int = 42
    lewm_dim: int = 256                         # LeWM output embedding dim
    vjepa2_dim: int = 1024                      # V-JEPA2 expected dim (ViT-L)
    log_every: int = 10


@dataclass
class CompatCheckResult:
    mean_cosine_sim: float
    std_cosine_sim: float
    kill_triggered: bool
    steps_completed: int
    cosine_sims_per_step: list[float] = field(default_factory=list)


class LeWMWrapper(nn.Module):
    """
    Thin wrapper around a LeWM checkpoint exposing encode(frames) → latent.
    Adapt this to your actual LeWM API.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # LeWM stores encoder under 'encoder' key — adjust if different
        self.encoder = state.get("encoder", state)
        if isinstance(self.encoder, dict):
            raise ValueError(
                "LeWM checkpoint should contain a nn.Module under 'encoder' key. "
                "Got a raw state_dict. Build the model first, then load weights."
            )
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, C, H, W] or [B, T, C, H, W]
        Returns:
            latent: [B, D] where D = lewm_dim
        """
        return self.encoder(frames)


def compute_cosine_similarity_batch(
    lewm_embeddings: torch.Tensor,
    vjepa2_targets: torch.Tensor,
    projection: Optional[nn.Linear] = None,
) -> torch.Tensor:
    """
    Compute mean cosine similarity between projected LeWM embeddings and VLA-JEPA targets.

    Args:
        lewm_embeddings: [B, lewm_dim]
        vjepa2_targets: [B, vjepa2_dim]
        projection: Optional linear layer to align dims.

    Returns:
        Scalar tensor — mean cosine similarity across batch.
    """
    if projection is not None:
        lewm_embeddings = projection(lewm_embeddings)

    # Normalize both sides
    lewm_norm = F.normalize(lewm_embeddings, dim=-1)
    target_norm = F.normalize(vjepa2_targets, dim=-1)

    cos_sim = (lewm_norm * target_norm).sum(dim=-1)  # [B]
    return cos_sim.mean()


def run_compat_check(
    config: CompatCheckConfig,
    dataloader: DataLoader,
    logger: Optional[logging.Logger] = None,
) -> CompatCheckResult:
    """
    Phase 0: run compat check between LeWM and VLA-JEPA target embeddings.

    The dataloader must yield dicts with keys:
        - 'frames': [B, C, H, W] robot observation frames
        - 'vjepa2_targets': [B, vjepa2_dim] pre-computed V-JEPA2 target embeddings

    Args:
        config: CompatCheckConfig
        dataloader: DataLoader yielding (frames, vjepa2_targets)
        logger: Optional logger

    Returns:
        CompatCheckResult with kill_triggered flag.
    """
    if logger is None:
        logger = get_logger(__name__)

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load LeWM encoder
    logger.info(f"Loading LeWM from {config.lewm_checkpoint}")
    lewm = LeWMWrapper(config.lewm_checkpoint, device=str(device))

    # Projection layer to align dims (not trained here — just random init to measure raw alignment)
    projection: Optional[nn.Linear] = None
    if config.lewm_dim != config.vjepa2_dim:
        projection = nn.Linear(config.lewm_dim, config.vjepa2_dim, bias=False).to(device)
        logger.info(
            f"Projection layer: {config.lewm_dim}D → {config.vjepa2_dim}D (random init, untrained)"
        )

    cosine_sims: list[float] = []
    steps_done = 0

    logger.info(f"Running compat check for {config.n_steps} steps...")

    for batch in dataloader:
        if steps_done >= config.n_steps:
            break

        frames = batch["frames"].to(device)           # [B, C, H, W]
        targets = batch["vjepa2_targets"].to(device)  # [B, vjepa2_dim]

        with torch.no_grad():
            lewm_emb = lewm.encode(frames)            # [B, lewm_dim]
            cos_sim = compute_cosine_similarity_batch(lewm_emb, targets, projection)

        cosine_sims.append(cos_sim.item())
        steps_done += 1

        if steps_done % config.log_every == 0:
            running_mean = sum(cosine_sims[-config.log_every:]) / config.log_every
            logger.info(f"step={steps_done:3d}  cosine_sim={running_mean:.4f}")

    mean_sim = sum(cosine_sims) / len(cosine_sims)
    std_sim = float(torch.tensor(cosine_sims).std().item())

    logger.info(f"\n{'─'*50}")
    logger.info(f"Phase 0 result:  mean_cosine_sim={mean_sim:.4f}  std={std_sim:.4f}")

    kill_triggered = check_kill_criterion(
        phase="phase0",
        metric_name="cosine_sim",
        metric_value=mean_sim,
        logger=logger,
    )

    if kill_triggered:
        logger.error(
            "Phase 0 KILLED. LeWM embeddings are too far from VLA-JEPA target space.\n"
            "Options:\n"
            "  1. Fine-tune LeWM on robot domain data before attempting projection.\n"
            "  2. Use V-JEPA2 encoder directly and skip LeWM substitution.\n"
            "  3. Increase n_steps and use a trained projection before re-checking."
        )
    else:
        logger.info(
            "Phase 0 PASSED. LeWM embeddings are sufficiently aligned. "
            "Proceed to Phase 1 (projection layer training)."
        )

    return CompatCheckResult(
        mean_cosine_sim=mean_sim,
        std_cosine_sim=std_sim,
        kill_triggered=kill_triggered,
        steps_completed=steps_done,
        cosine_sims_per_step=cosine_sims,
    )
