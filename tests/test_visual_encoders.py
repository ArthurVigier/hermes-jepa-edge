"""
Unit tests for visual encoder source selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from src.visual_encoders.factory import (
    DirectVJEPA2Source,
    LeWMProjectionSource,
    build_visual_source,
    visual_source_trainable_parameters,
)


@dataclass
class DummyConfig:
    visual_encoder_type: str = "lewm_projection"
    lewm_checkpoint: str = "lewm.pt"
    projection_checkpoint: str = "projection.pt"
    vjepa2_embeddings_key: str = "vjepa2_embeddings"


def test_direct_vjepa2_source_reads_batch_key():
    source = DirectVJEPA2Source()
    batch = {"vjepa2_embeddings": torch.randn(2, 1024)}

    out = source(batch=batch, frames=None)

    assert out.shape == (2, 1024)


def test_direct_vjepa2_source_missing_key_raises():
    source = DirectVJEPA2Source("custom_key")

    with pytest.raises(KeyError, match="custom_key"):
        source(batch={}, frames=None)


def test_lewm_projection_source_keeps_encoder_eval():
    encoder = nn.Linear(4, 4)
    projection = nn.Linear(4, 8)
    source = LeWMProjectionSource(encoder=encoder, projection=projection)

    source.train()

    assert source.encoder.training is False
    assert source.projection.training is True


def test_build_visual_source_direct():
    source = build_visual_source(
        DummyConfig(visual_encoder_type="vjepa2_direct"),
        device="cpu",
    )

    assert isinstance(source, DirectVJEPA2Source)
    assert list(visual_source_trainable_parameters(source)) == []


def test_build_visual_source_lewm_projection(monkeypatch):
    encoder = nn.Linear(4, 4)
    projection = nn.Linear(4, 8)

    monkeypatch.setattr(
        "src.visual_encoders.factory.load_lewm_encoder",
        lambda checkpoint_path, device: encoder,
    )
    monkeypatch.setattr(
        "src.visual_encoders.factory._load_projection_module",
        lambda checkpoint_path, device: projection,
    )

    source = build_visual_source(DummyConfig(), device="cpu")

    assert isinstance(source, LeWMProjectionSource)
    assert source.encoder is encoder
    assert source.projection is projection
    assert list(visual_source_trainable_parameters(source)) == list(projection.parameters())
