#!/usr/bin/env python3
"""
scripts/generate_teacher_trajectories.py
Generate annotated robot trajectories with Hermes-4.3-36B teacher.

Runs on RunPod A100 80GB. Saves to JSONL for offline student training.

Usage:
    python scripts/generate_teacher_trajectories.py \
        --config configs/phase2b_distill.yaml \
        --scenarios data/task_scenarios.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.teacher_student import DistillationConfig, generate_teacher_trajectories
from src.utils import get_logger


# ── Built-in task scenarios (extend with your own) ────────────────────────────

DEFAULT_SCENARIOS = [
    {"instruction": "Pick up the red cube and place it in the bin",
     "world_state": "Red cube at [0.3, 0.2, 0.1], bin at [0.5, 0.0, 0.0], arm at home position"},
    {"instruction": "Grasp the blue cylinder",
     "world_state": "Blue cylinder at [0.25, -0.1, 0.15], arm at home position"},
    {"instruction": "Move arm to the top-left corner of the workspace",
     "world_state": "Workspace bounds: x[0,0.6] y[-0.3,0.3] z[0,0.5], arm at [0.3, 0.0, 0.2]"},
    {"instruction": "Pick and place: move the green block from table to shelf",
     "world_state": "Green block at [0.2, 0.1, 0.0], shelf at [0.4, -0.2, 0.3]"},
    {"instruction": "Release the object currently held by the gripper",
     "world_state": "Gripper holding small object, target drop zone at [0.35, 0.0, 0.1]"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--scenarios",
        default=None,
        help="JSON file with task scenarios. Uses built-in defaults if not provided.",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Override number of trajectories from config"
    )
    args = parser.parse_args()

    logger = get_logger("teacher_gen")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = DistillationConfig(**cfg_dict)

    if args.n is not None:
        config.n_trajectories = args.n

    # Load scenarios
    if args.scenarios:
        with open(args.scenarios) as f:
            scenarios = json.load(f)
    else:
        # Replicate default scenarios to reach n_trajectories
        base = DEFAULT_SCENARIOS
        scenarios = (base * (config.n_trajectories // len(base) + 1))[:config.n_trajectories]
        logger.info(f"Using {len(scenarios)} built-in scenarios (repeated to fill {config.n_trajectories})")

    trajectories = generate_teacher_trajectories(config, scenarios, logger=logger)

    success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
    logger.info(f"\nGeneration complete: {len(trajectories)} trajectories, {success_rate:.1%} success rate")
    logger.info(f"Saved to: {config.teacher_trajectories_path}")


if __name__ == "__main__":
    main()
