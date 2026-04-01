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

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    return DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Projection layer training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    logger = get_logger("phase1")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    config = ProjectionConfig(**cfg_dict)
    dataloader = load_robot_dataloader(config)

    # Load LeWM encoder (adapt to your actual LeWM API)
    logger.info(f"Loading LeWM encoder from {config.lewm_checkpoint}")
    lewm_state = torch.load(config.lewm_checkpoint, map_location="cpu", weights_only=True)
    lewm_encoder = lewm_state.get("encoder")
    if lewm_encoder is None:
        logger.error("Could not find 'encoder' key in LeWM checkpoint. Check your checkpoint format.")
        sys.exit(1)

    trainer = ProjectionTrainer(config, lewm_encoder, dataloader, logger=logger)
    result = trainer.train()

    if result.get("kill_triggered"):
        logger.error("Phase 1 KILLED.")
        sys.exit(1)

    logger.info(f"Phase 1 complete. Final align_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
