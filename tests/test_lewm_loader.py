"""
Unit tests for LeWM checkpoint loading helpers.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn as nn

from src.adapters import lewm_loader


def test_infer_num_attention_heads_from_size_name():
    assert lewm_loader._infer_num_attention_heads(192, "tiny") == 3
    assert lewm_loader._infer_num_attention_heads(384, "small") == 6


def test_get_lewm_output_dim_prefers_explicit_attr():
    module = nn.Identity()
    module.output_dim = 192
    assert lewm_loader.get_lewm_output_dim(module) == 192


def test_load_lewm_encoder_uses_public_builder(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"stub")

    public_state = {
        "encoder.embeddings.cls_token": torch.zeros(1, 1, 192),
        "encoder.layernorm.weight": torch.zeros(192),
        "projector.net.0.weight": torch.zeros(2048, 192),
    }
    built = nn.Identity()

    monkeypatch.setattr(lewm_loader.torch, "load", lambda *args, **kwargs: public_state)
    monkeypatch.setattr(
        lewm_loader,
        "_build_public_lewm_encoder",
        lambda state, path, device: built,
    )

    loaded = lewm_loader.load_lewm_encoder(str(checkpoint), device="cpu")

    assert loaded is built


def test_load_lewm_encoder_falls_back_to_full_pickle(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"stub")
    module = nn.Linear(4, 2)

    calls: list[bool] = []

    def fake_load(_path, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise pickle.UnpicklingError("unsafe globals")
        return module

    monkeypatch.setattr(lewm_loader.torch, "load", fake_load)

    loaded = lewm_loader.load_lewm_encoder(str(checkpoint), device="cpu")

    assert loaded is module
    assert calls == [True, False]


def test_build_public_lewm_encoder_loads_vit_and_projector(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"stub")
    (tmp_path / "config.json").write_text(
        '{"encoder": {"size": "tiny", "patch_size": 14, "image_size": 224}}'
    )

    state = {
        "encoder.embeddings.cls_token": torch.zeros(1, 1, 192),
        "encoder.layernorm.weight": torch.zeros(192),
        "encoder.layernorm.bias": torch.zeros(192),
        "encoder.encoder.layer.0.intermediate.dense.weight": torch.zeros(768, 192),
        "projector.net.0.weight": torch.zeros(2048, 192),
        "projector.net.0.bias": torch.zeros(2048),
        "projector.net.1.weight": torch.ones(2048),
        "projector.net.1.bias": torch.zeros(2048),
        "projector.net.1.running_mean": torch.zeros(2048),
        "projector.net.1.running_var": torch.ones(2048),
        "projector.net.1.num_batches_tracked": torch.tensor(0),
        "projector.net.3.weight": torch.zeros(192, 2048),
        "projector.net.3.bias": torch.zeros(192),
    }

    observed: dict[str, object] = {}

    class FakeViTConfig:
        def __init__(self, **kwargs):
            observed["config"] = kwargs
            self.kwargs = kwargs

    class FakeViTModel(nn.Module):
        def __init__(self, config, add_pooling_layer=False):
            super().__init__()
            observed["add_pooling_layer"] = add_pooling_layer
            observed["vit_config_obj"] = config

        def load_state_dict(self, incoming_state):
            observed["encoder_state"] = incoming_state
            return None

        def forward(self, frames, interpolate_pos_encoding=True):
            batch = frames.shape[0]
            hidden = 192

            class Output:
                last_hidden_state = torch.zeros(batch, 1, hidden)

            return Output()

    monkeypatch.setattr(
        lewm_loader,
        "_load_transformers_vit",
        lambda: (FakeViTConfig, FakeViTModel),
    )

    model = lewm_loader._build_public_lewm_encoder(state, checkpoint, device="cpu")

    assert isinstance(model, nn.Module)
    assert observed["add_pooling_layer"] is False
    assert observed["config"] == {
        "image_size": 224,
        "patch_size": 14,
        "num_channels": 3,
        "hidden_size": 192,
        "intermediate_size": 768,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "qkv_bias": True,
    }
    encoder_state = observed["encoder_state"]
    assert set(encoder_state) == {
        "embeddings.cls_token",
        "layernorm.weight",
        "layernorm.bias",
        "encoder.layer.0.intermediate.dense.weight",
    }
    assert encoder_state["embeddings.cls_token"].shape == (1, 1, 192)
    assert encoder_state["encoder.layer.0.intermediate.dense.weight"].shape == (768, 192)
