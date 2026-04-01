#!/usr/bin/env python3
"""
scripts/phase1_train_adapter.py
Phase 1 entrypoint — train LeWM → JEPA projection layer.

Usage:
    python scripts/phase1_train_adapter.py --config configs/phase1_adapter.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.lewm_loader import get_lewm_output_dim, load_lewm_encoder
from src.adapters.projection import ProjectionConfig, ProjectionTrainer
from src.utils import get_logger


def load_robot_dataloader(config: ProjectionConfig):
    """Build dataloader for projection training."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ProjectionDataset(Dataset):
        def __init__(self, cfg: ProjectionConfig) -> None:
            self.n = 1000
            self.lewm_dim = cfg.lewm_dim
            self.vjepa2_dim = cfg.vjepa2_dim

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            return {
                "frames": torch.randn(3, 224, 224),
                "vjepa2_targets": torch.randn(self.vjepa2_dim),
            }

    ds = ProjectionDataset(config)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Projection layer training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    logger = get_logger("phase1")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    config = ProjectionConfig(**cfg_dict)
    dataloader = load_robot_dataloader(config)

    logger.info(f"Loading LeWM encoder from {config.lewm_checkpoint}")
    lewm_encoder = load_lewm_encoder(config.lewm_checkpoint, device="cpu")
    actual_lewm_dim = get_lewm_output_dim(lewm_encoder)
    if actual_lewm_dim is not None and actual_lewm_dim != config.lewm_dim:
        logger.info(
            f"Detected LeWM embedding dim {actual_lewm_dim}; overriding configured "
            f"lewm_dim={config.lewm_dim}."
        )
        config.lewm_dim = actual_lewm_dim

    trainer = ProjectionTrainer(config, lewm_encoder, dataloader, logger=logger)
    result = trainer.train()

    if result.get("kill_triggered"):
        logger.error("Phase 1 KILLED.")
        sys.exit(1)

    logger.info(f"Phase 1 complete. Final align_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
