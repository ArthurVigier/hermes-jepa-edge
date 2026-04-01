#!/usr/bin/env python3
"""
scripts/phase2b_distillation.py
Phase 2b entrypoint — fine-tune Hermes-3-8B student on teacher trajectories.

HARDWARE: Vast.ai RTX 4090 24GB (~$0.4/h) — very comfortable for 8B QLoRA (~6GB)

Prerequisite:
  1. Phase 2 complete: experiments/phase2/lora_final/ exists
  2. Phase 2 LoRA merged: experiments/phase2/merged_bf16/ exists
  3. Teacher trajectories generated:
     python scripts/generate_teacher_trajectories.py --config configs/phase2b_distill.yaml

Usage:
    python scripts/phase2b_distillation.py --config configs/phase2b_distill.yaml

Output: experiments/phase2b/student_final/ (Hermes-3-8B fine-tuned, HF format)
Next:   python scripts/phase3_export_tensorrt.py → Jetson edge deployment
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.teacher_student import DistillationConfig, StudentDistillationTrainer
from src.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2b: Distill Hermes-4.3-36B → Hermes-3-8B student (RTX 4090)"
    )
    parser.add_argument("--config", required=True, help="Path to configs/phase2b_distill.yaml")
    args = parser.parse_args()

    logger = get_logger("phase2b")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = DistillationConfig(**cfg_dict)

    # ── Verify prerequisites ─────────────────────────────────────────────────
    traj_path = Path(config.teacher_trajectories_path)
    if not traj_path.exists():
        logger.error(
            f"Teacher trajectories not found: {traj_path}\n"
            f"Generate them first with:\n"
            f"  python scripts/generate_teacher_trajectories.py \\\n"
            f"      --config {args.config}"
        )
        sys.exit(1)

    n_lines = sum(1 for _ in traj_path.open())
    logger.info(f"Found {n_lines} teacher trajectories at {traj_path}")
    if n_lines < 50:
        logger.warning(
            f"Only {n_lines} trajectories found. Recommend ≥ 200 for good distillation quality."
        )

    # ── Run distillation ─────────────────────────────────────────────────────
    trainer = StudentDistillationTrainer(config, logger=logger)
    trainer.train()

    logger.info(
        f"\nPhase 2b complete.\n"
        f"Student saved → {config.output_dir}/student_final\n"
        f"\nNext: convert to GGUF Q4_K_M for Jetson:\n"
        f"  python scripts/phase3_export_tensorrt.py --config configs/phase3_edge.yaml\n"
        f"\nOr manually with llama.cpp:\n"
        f"  python llama.cpp/convert_hf_to_gguf.py {config.output_dir}/student_final \\\n"
        f"    --outtype f16 --outfile hermes3_8b_f16.gguf\n"
        f"  ./llama.cpp/build/bin/llama-quantize hermes3_8b_f16.gguf \\\n"
        f"    hermes3_8b_q4km.gguf Q4_K_M"
    )


if __name__ == "__main__":
    main()
