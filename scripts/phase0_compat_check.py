#!/usr/bin/env python3
"""
scripts/phase0_compat_check.py
Phase 0 entrypoint — run LeWM ↔ VLA-JEPA compatibility check.

Usage:
    python scripts/phase0_compat_check.py \
        --lewm-checkpoint path/to/lewm.pt \
        --dataset libero \
        --n-steps 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.compat_check import CompatCheckConfig, run_compat_check
from src.utils import get_logger


def make_dummy_dataloader(config: CompatCheckConfig):
    """
    Build a dataloader for compat check.
    In production: load actual LIBERO/Droid frames + pre-computed V-JEPA2 targets.
    Here we provide a real stub that loads from a directory of .pt files.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class RobotFrameDataset(Dataset):
        def __init__(self, dataset_name: str, lewm_dim: int, vjepa2_dim: int) -> None:
            self.dataset_name = dataset_name
            self.lewm_dim = lewm_dim
            self.vjepa2_dim = vjepa2_dim
            # In production: load from LeRobot dataset or preprocessed .pt files
            self.n = 200  # Stub: 200 samples

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            # Stub: replace with actual frame loading
            return {
                "frames": torch.randn(3, 224, 224),
                # V-JEPA2 targets: precompute offline with V-JEPA2 ViT-L
                # python scripts/precompute_vjepa2_targets.py --dataset libero
                "vjepa2_targets": torch.randn(self.vjepa2_dim),
            }

    ds = RobotFrameDataset(config.dataset_name, config.lewm_dim, config.vjepa2_dim)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: LeWM ↔ VLA-JEPA compat check")
    parser.add_argument("--lewm-checkpoint", required=True, help="Path to LeWM .pt checkpoint")
    parser.add_argument("--dataset", default="libero", choices=["libero", "droid", "habitat"])
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lewm-dim", type=int, default=256)
    parser.add_argument("--vjepa2-dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logger = get_logger("phase0")

    config = CompatCheckConfig(
        lewm_checkpoint=args.lewm_checkpoint,
        dataset_name=args.dataset,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lewm_dim=args.lewm_dim,
        vjepa2_dim=args.vjepa2_dim,
        device=args.device,
    )

    dataloader = make_dummy_dataloader(config)
    result = run_compat_check(config, dataloader, logger=logger)

    logger.info(f"\nResult: cosine_sim={result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}")

    if result.kill_triggered:
        logger.error("Phase 0 KILLED. See logs above for remediation options.")
        sys.exit(1)
    else:
        logger.info("Phase 0 PASSED. Proceed to Phase 1.")
        sys.exit(0)


if __name__ == "__main__":
    main()
