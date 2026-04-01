"""
Factory helpers for selecting the visual embedding source.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Optional, Protocol

import torch
import torch.nn as nn

from src.adapters.lewm_loader import load_lewm_encoder
from src.adapters.projection import build_projection_from_state_dict
from src.utils import get_logger


class VisualSourceConfig(Protocol):
    visual_encoder_type: str
    lewm_checkpoint: str
    projection_checkpoint: str
    vjepa2_embeddings_key: str


class DirectVJEPA2Source(nn.Module):
    """Use precomputed V-JEPA2 embeddings directly from the batch."""

    def __init__(self, embeddings_key: str = "vjepa2_embeddings") -> None:
        super().__init__()
        self.embeddings_key = embeddings_key

    def forward(
        self,
        *,
        batch: dict[str, torch.Tensor],
        frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del frames
        if self.embeddings_key not in batch:
            raise KeyError(
                "Direct V-JEPA2 mode requires precomputed current-frame embeddings "
                f"under batch['{self.embeddings_key}']."
            )
        return batch[self.embeddings_key]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []


class LeWMProjectionSource(nn.Module):
    """Encode frames with LeWM, then project into the V-JEPA2 target space."""

    def __init__(self, encoder: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.encoder.eval()

    def forward(
        self,
        *,
        batch: dict[str, torch.Tensor],
        frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del batch
        if frames is None:
            raise ValueError("LeWM projection mode requires `frames` input.")
        with torch.no_grad():
            lewm_emb = self.encoder(frames)
        return self.projection(lewm_emb)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.projection.parameters())

    def train(self, mode: bool = True) -> "LeWMProjectionSource":
        super().train(mode)
        self.encoder.eval()
        self.projection.train(mode)
        return self


def _load_projection_module(
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    projection = build_projection_from_state_dict(state["projection_state"])
    return projection.to(device)


def build_visual_source(
    config: VisualSourceConfig,
    *,
    device: str,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """Build the configured visual embedding source."""
    if logger is None:
        logger = get_logger(__name__)

    if config.visual_encoder_type == "vjepa2_direct":
        logger.info(
            "Visual encoder mode: V-JEPA2 direct "
            f"(expects batch['{config.vjepa2_embeddings_key}'])."
        )
        return DirectVJEPA2Source(config.vjepa2_embeddings_key)

    if config.visual_encoder_type == "lewm_projection":
        logger.info(f"Loading LeWM from {config.lewm_checkpoint}")
        encoder = load_lewm_encoder(config.lewm_checkpoint, device=device)
        projection = _load_projection_module(config.projection_checkpoint, device=device)
        logger.info("Visual encoder mode: LeWM encoder + trained projection.")
        return LeWMProjectionSource(encoder=encoder, projection=projection)

    raise ValueError(
        f"Unknown visual_encoder_type={config.visual_encoder_type!r}. "
        "Expected 'lewm_projection' or 'vjepa2_direct'."
    )


def visual_source_trainable_parameters(source: nn.Module) -> Iterable[nn.Parameter]:
    if hasattr(source, "trainable_parameters"):
        return source.trainable_parameters()
    return source.parameters()
