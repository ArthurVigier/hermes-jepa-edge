#!/usr/bin/env python3
"""
scripts/phase2_hermes_backbone.py
Phase 2 entrypoint — VLA-JEPA training with Hermes-4.3-36B QLoRA backbone.

HARDWARE REQUIRED: RunPod A100 80GB (~$1.5/h)
DO NOT run on RTX 4090: Hermes-4.3-36B QLoRA needs ~25GB VRAM.
The script will detect insufficient VRAM and abort early.

Usage:
    python scripts/phase2_hermes_backbone.py --config configs/phase2_hermes.yaml

Estimated cost: ~$75-100 for 5000 steps (~50h on A100 80GB)
Output: experiments/phase2/checkpoint_best.pt (LoRA adapter weights)
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.lewm_loader import load_lewm_encoder
from src.pipeline.hermes_vla import (
    HermesVLAConfig,
    build_hermes_vla,
    dual_loss,
    format_tool_call_prompt,
)
from src.utils import (
    StepTimer,
    check_kill_criterion,
    get_logger,
    seed_everything,
)


def build_robot_dataloader(config: HermesVLAConfig, tokenizer):
    """
    Build dataloader for Phase 2 training.
    Each batch contains:
      - frames: [B, C, H, W] robot observation
      - vjepa2_targets: [B, vjepa2_dim] target latent (from LeWM projection or V-JEPA2)
      - input_ids / attention_mask / labels: tokenized instruction + action sequence
    Replace the stub Dataset with your actual LIBERO / Droid loader.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class RobotVLADataset(Dataset):
        def __init__(self, cfg: HermesVLAConfig, tok) -> None:
            self.cfg = cfg
            self.tok = tok
            self.n = 2000  # stub — replace with len(your_dataset)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            # --- stub: replace with real LIBERO/Droid loading ---
            frames = torch.randn(3, 224, 224)
            vjepa2_targets = torch.randn(self.cfg.vjepa2_dim)

            # Tokenize a dummy instruction
            prompt = format_tool_call_prompt(
                instruction="Pick up the red cube and place it in the bin.",
                world_state_description="Red cube at [0.3, 0.2, 0.1], bin at [0.5, 0.0, 0.0].",
                tokenizer=self.tok,
                enable_thinking=True,
            )
            tokens = self.tok(
                prompt,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            return {
                "frames": frames,
                "vjepa2_targets": vjepa2_targets,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    ds = RobotVLADataset(config, tokenizer)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Hermes-4.3-36B QLoRA VLA backbone training (A100 80GB)"
    )
    parser.add_argument("--config", required=True, help="Path to configs/phase2_hermes.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and run 5 steps to verify setup, then exit",
    )
    args = parser.parse_args()

    logger = get_logger("phase2")

    # ── Enforce A100 hardware ────────────────────────────────────────────────
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name} ({vram_gb:.0f}GB)")
        if vram_gb < 48:
            logger.error(
                f"\n{'═'*60}\n"
                f"ABORT: {gpu_name} has only {vram_gb:.0f}GB VRAM.\n"
                f"Hermes-4.3-36B QLoRA training requires ≥48GB (A100 80GB recommended).\n"
                f"→ On RTX 4090 (24GB): expected OOM at first backward pass.\n"
                f"→ Rent RunPod A100 80GB: https://runpod.io\n"
                f"{'═'*60}"
            )
            sys.exit(1)
    else:
        logger.warning("No CUDA device detected. Training will be extremely slow on CPU.")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = HermesVLAConfig(**cfg_dict)

    seed_everything(config.seed)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading Hermes-4.3-36B with QLoRA (NF4 4-bit)...")
    hermes, tokenizer, predictor = build_hermes_vla(config, logger=logger)

    device = torch.device(config.device)
    predictor = predictor.to(device)

    # ── Dataloader ───────────────────────────────────────────────────────────
    dataloader = build_robot_dataloader(config, tokenizer)

    # ── Load LeWM encoder + projection ───────────────────────────────────────
    logger.info(f"Loading LeWM from {config.lewm_checkpoint}")
    lewm_encoder = load_lewm_encoder(config.lewm_checkpoint, device=str(device))

    from src.adapters.projection import build_projection_from_state_dict
    proj_state = torch.load(config.projection_checkpoint, map_location="cpu", weights_only=True)
    projection = build_projection_from_state_dict(proj_state["projection_state"]).to(device)

    # ── Optimizer: only LoRA adapters + predictor + projection ───────────────
    trainable_params = (
        list(p for p in hermes.parameters() if p.requires_grad)   # LoRA adapters only
        + list(predictor.parameters())
        + list(projection.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in hermes.parameters()) + n_trainable
    logger.info(
        f"Trainable parameters: {n_trainable:,} "
        f"({100 * n_trainable / n_total:.2f}% of total)"
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_steps, eta_min=config.lr * 0.1)

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    n_steps = 5 if args.dry_run else config.n_steps
    dry_run_suffix = "  [DRY RUN]" if args.dry_run else ""
    logger.info(f"\nStarting Phase 2 training — {n_steps} steps{dry_run_suffix}")

    hermes.train()
    predictor.train()
    projection.train()

    data_iter = iter(dataloader)
    kill_triggered = False

    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        frames = batch["frames"].to(device)
        vjepa2_targets = batch["vjepa2_targets"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with StepTimer(step, log_every=config.log_every, logger=logger):
            # 1. LeWM encode + project
            with torch.no_grad():
                lewm_emb = lewm_encoder(frames)           # [B, lewm_dim]
            projected = projection(lewm_emb)               # [B, vjepa2_dim]

            # 2. Forward through Hermes (NTP logits + hidden states)
            outputs = hermes(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            ntp_logits = outputs.logits                    # [B, T, vocab]

            # 3. Predictor: visual + instruction hidden → predicted future state
            instruction_hidden = outputs.hidden_states[-1] # [B, T, H]
            predicted_embedding = predictor(projected, instruction_hidden)

            # 4. Dual loss
            loss, metrics = dual_loss(
                ntp_logits=ntp_logits,
                ntp_labels=labels,
                predicted_embedding=predicted_embedding,
                target_embedding=vjepa2_targets,
                lambda_ntp=config.lambda_ntp,
                lambda_jepa=config.lambda_jepa,
                jepa_loss_dropout=config.jepa_loss_dropout,
            )

            # 5. Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if step % config.log_every == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"step={step:5d}  "
                f"loss={metrics['loss/total']:.4f}  "
                f"ntp={metrics['loss/ntp']:.4f}  "
                f"jepa={metrics['loss/jepa']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"gpu={mem_gb:.1f}GB"
            )

        if step % config.save_every == 0 and step > 0:
            ckpt_path = output_dir / f"checkpoint_step{step}.pt"
            # Save LoRA adapter only (small, ~100-500MB)
            hermes.save_pretrained(str(output_dir / f"lora_step{step}"))
            torch.save({
                "step": step,
                "predictor_state": predictor.state_dict(),
                "projection_state": projection.state_dict(),
                "config": config,
                "metrics": metrics,
            }, ckpt_path)
            logger.info(f"Checkpoint saved → {ckpt_path}")

        # Kill criterion check
        if step == config.kill_check_step:
            # Placeholder: replace with actual LIBERO eval
            task_success = 0.0  # TODO: run LIBERO eval here
            logger.warning(
                f"Kill criterion check at step {step}. "
                f"task_success={task_success:.2f} (placeholder — run LIBERO eval)"
            )
            kill_triggered = check_kill_criterion(
                phase="phase2",
                metric_name="task_success",
                metric_value=task_success,
                logger=logger,
            )
            if kill_triggered:
                logger.error("Phase 2 KILLED. task_success < 30% threshold.")
                sys.exit(1)

    # ── Final save ───────────────────────────────────────────────────────────
    hermes.save_pretrained(str(output_dir / "lora_final"))
    torch.save({
        "step": n_steps,
        "predictor_state": predictor.state_dict(),
        "projection_state": projection.state_dict(),
        "config": config,
    }, output_dir / "checkpoint_final.pt")

    if args.dry_run:
        logger.info("Dry run complete. Setup verified. Ready for full training run.")
    else:
        logger.info(
            f"\nPhase 2 complete.\n"
            f"LoRA adapter saved → {output_dir / 'lora_final'}\n"
            "Next: merge LoRA → BF16 with: "
            f"python scripts/phase2_merge_lora.py --config {args.config}"
        )


if __name__ == "__main__":
    main()
