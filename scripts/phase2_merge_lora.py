#!/usr/bin/env python3
"""
scripts/phase2_merge_lora.py
Merge Phase 2 LoRA adapter back into Hermes-4.3-36B BF16 weights.

Required before Phase 2b distillation:
  LoRA adapter (100-500MB) + base Hermes-4.3-36B → merged BF16 model (~72GB)

The merged model is then used as teacher for trajectory generation.
It's a one-time operation. The merged weights are NOT deployed — only
Hermes-3-8B (distilled student) is deployed on Jetson.

HARDWARE: A100 80GB (RunPod) — merge needs ~72GB RAM to hold BF16 weights

Usage:
    python scripts/phase2_merge_lora.py \
        --lora-dir experiments/phase2/lora_final \
        --output-dir experiments/phase2/merged_bf16
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Hermes-4.3-36B BF16")
    parser.add_argument("--lora-dir", required=True, help="Path to LoRA adapter dir (Phase 2 output)")
    parser.add_argument(
        "--output-dir",
        default="experiments/phase2/merged_bf16",
        help="Output path for merged BF16 model",
    )
    parser.add_argument(
        "--base-model-id",
        default="NousResearch/Hermes-4.3-36B",
        help="HuggingFace model ID for base Hermes-4.3-36B",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional HuggingFace repo to push merged model (e.g. youruser/hermes43-vla-robotics)",
    )
    args = parser.parse_args()

    logger = get_logger("merge_lora")

    # ── VRAM check ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 48:
            logger.error(
                f"Only {vram_gb:.0f}GB VRAM detected. "
                f"Merging Hermes-4.3-36B requires ~72GB for BF16. "
                f"Use A100 80GB on RunPod."
            )
            sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        logger.error("Install dependencies: pip install transformers peft")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base model in BF16 ───────────────────────────────────────────────
    logger.info(f"Loading base model: {args.base_model_id} (BF16, ~72GB)")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Load and merge LoRA adapter ───────────────────────────────────────────
    logger.info(f"Loading LoRA adapter from {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)

    logger.info("Merging LoRA weights into base model (this takes ~5-10 minutes)...")
    model = model.merge_and_unload()  # Returns plain AutoModelForCausalLM with merged weights

    # ── Save merged model ─────────────────────────────────────────────────────
    logger.info(f"Saving merged BF16 model → {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    merged_size_gb = sum(
        f.stat().st_size for f in output_dir.rglob("*.safetensors")
    ) / 1e9
    logger.info(f"Merged model saved: {merged_size_gb:.1f}GB")

    # ── Optional HuggingFace push ─────────────────────────────────────────────
    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(args.push_to_hub)
        logger.info(f"Pushed to: https://huggingface.co/{args.push_to_hub}")

    logger.info(
        f"\nMerge complete.\n"
        f"Merged model: {output_dir}\n"
        f"Next: generate teacher trajectories with:\n"
        f"  python scripts/generate_teacher_trajectories.py \\\n"
        f"    --config configs/phase2b_distill.yaml \\\n"
        f"    --teacher-model-path {output_dir}"
    )


if __name__ == "__main__":
    main()
