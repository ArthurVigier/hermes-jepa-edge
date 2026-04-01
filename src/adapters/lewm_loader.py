"""
Utilities for loading LeWM checkpoints into a callable image encoder.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class _LeWMProjector(nn.Module):
    """Mirror the upstream LeWM projector used in public Hugging Face weights."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PublicLeWMEncoder(nn.Module):
    """Expose `forward(frames) -> latent` for weights-only LeWM checkpoints."""

    def __init__(self, encoder: nn.Module, projector: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.output_dim = output_dim

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self(frames)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        output = self.encoder(frames.float(), interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        return self.projector(cls_token)


def _load_transformers_vit() -> tuple[type[Any], type[Any]]:
    try:
        from transformers import ViTConfig, ViTModel
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Loading public LeWM `weights.pt` checkpoints requires a compatible "
            "`transformers` installation with ViT support."
        ) from exc
    return ViTConfig, ViTModel


def _load_json_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find sibling config file for public LeWM checkpoint: {config_path}"
        )
    with config_path.open() as f:
        return json.load(f)


def _looks_like_public_lewm_state_dict(state: Mapping[str, Any]) -> bool:
    return (
        "encoder.embeddings.cls_token" in state
        and "encoder.layernorm.weight" in state
        and "projector.net.0.weight" in state
    )


def _infer_num_attention_heads(hidden_size: int, size_name: str | None) -> int:
    size_to_heads = {
        "tiny": 3,
        "small": 6,
        "base": 12,
        "large": 16,
        "huge": 16,
    }
    if size_name in size_to_heads:
        return size_to_heads[size_name]

    # Fall back to common ViT head widths when the config name is unavailable.
    for head_dim in (64, 32, 16):
        if hidden_size % head_dim == 0:
            return hidden_size // head_dim
    raise ValueError(f"Could not infer attention heads for hidden size {hidden_size}")


def _build_public_lewm_encoder(
    state: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
    device: str,
) -> nn.Module:
    config = _load_json_config(checkpoint_path)
    encoder_cfg = config["encoder"]

    hidden_size = state["encoder.embeddings.cls_token"].shape[-1]
    intermediate_size = state["encoder.encoder.layer.0.intermediate.dense.weight"].shape[0]
    layer_ids = {
        int(key.split(".")[3])
        for key in state
        if key.startswith("encoder.encoder.layer.")
    }
    num_hidden_layers = len(layer_ids)
    num_attention_heads = _infer_num_attention_heads(hidden_size, encoder_cfg.get("size"))

    vit_config_cls, vit_model_cls = _load_transformers_vit()
    vit_config = vit_config_cls(
        image_size=encoder_cfg["image_size"],
        patch_size=encoder_cfg["patch_size"],
        num_channels=3,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        qkv_bias=True,
    )
    encoder = vit_model_cls(vit_config, add_pooling_layer=False)

    projector_in = state["projector.net.0.weight"].shape[1]
    projector_hidden = state["projector.net.0.weight"].shape[0]
    projector_out = state["projector.net.3.weight"].shape[0]
    projector = _LeWMProjector(
        input_dim=projector_in,
        hidden_dim=projector_hidden,
        output_dim=projector_out,
    )

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    projector_state = {
        key.removeprefix("projector."): value
        for key, value in state.items()
        if key.startswith("projector.")
    }

    encoder.load_state_dict(encoder_state)
    projector.load_state_dict(projector_state)

    module = _PublicLeWMEncoder(
        encoder=encoder,
        projector=projector,
        output_dim=projector_out,
    ).to(device)
    module.eval()
    return module


def get_lewm_output_dim(module: nn.Module) -> int | None:
    """Best-effort latent dimension inference for downstream projection setup."""
    for attr_name in ("output_dim", "latent_dim"):
        value = getattr(module, attr_name, None)
        if isinstance(value, int):
            return value

    config = getattr(module, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size

    return None


def load_lewm_encoder(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a LeWM checkpoint into a module exposing `forward(frames) -> latent`.

    Supported formats:
      - a checkpoint containing an instantiated `encoder` module
      - public Hugging Face LeWM `weights.pt` state_dict checkpoints with sibling `config.json`
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LeWM checkpoint not found: {path}")

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # Trusted local checkpoints may serialize full modules instead of tensors.
        state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state, nn.Module):
        module = state.to(device)
        module.eval()
        return module

    if isinstance(state, Mapping):
        encoder = state.get("encoder")
        if isinstance(encoder, nn.Module):
            encoder = encoder.to(device)
            encoder.eval()
            return encoder
        if _looks_like_public_lewm_state_dict(state):
            return _build_public_lewm_encoder(state, path, device)

    raise ValueError(
        "Unsupported LeWM checkpoint format. Expected either an instantiated `encoder` "
        "module or a public Hugging Face `weights.pt` LeWM state_dict."
    )
