# hermes-jepa-edge

**JEPA world model + Hermes LLM → edge robotic agent on Jetson Orin Nano**

Built on top of [VLA-JEPA](https://arxiv.org/abs/2602.10098) (Sun et al., 2026).
Novel contribution: LeWM 15M encoder + Hermes-4.3-36B QLoRA backbone + Hermes-3-8B edge distillation + TensorRT Jetson deployment.

---

## Architecture

```
LeWM (15M JEPA encoder)
    ↓  nn.Linear projection (lewm_dim → vjepa2_dim)
VLA-JEPA Predictor  ←  initialized from Hermes-4.3-36B layers 8-16
    ↓  dual loss: NTP + JEPA alignment (LLM-JEPA style)
Hermes-4.3-36B backbone  [QLoRA, A100 80GB — training only]
    ↓  LoRA adapter merge + distillation
Hermes-3-8B student  [QLoRA, RTX 4090]
    ↓  Q4_K_M GGUF + TensorRT export
Jetson Orin Nano (8GB, <50ms/step)
```

**Why Hermes-4.3-36B as backbone (not 8B):**
- 36B has significantly better physical reasoning and tool-call accuracy
- QLoRA keeps training VRAM at ~25GB → fits A100 80GB comfortably
- LoRA adapter (~100-500MB) is distilled into Hermes-3-8B for edge
- 36B never runs on Jetson — it's the teacher, not the deployed model

---

## Pipeline phases

| Phase | What | Duration | Budget | Kill criterion |
|-------|------|----------|--------|----------------|
| 0 | LeWM ↔ VLA-JEPA compat check | 1-2d | ~$2 | CosSim < 0.3 |
| 1 | Projection layer training | 1w | ~$30 | align_loss > 0.8 @ 500 steps |
| 2 | **Hermes-4.3-36B** QLoRA backbone (A100 80GB) | 1w | ~$100 | task_success < 30% LIBERO |
| 2b | 36B → 8B distillation (RTX 4090) | 1w | ~$27 | — |
| 3 | TensorRT edge deployment | 3-5d | $0 | latency > 200ms on Jetson |
| 4 | Prompt library | 2-3d | $0 | — |

**Total: ~4 weeks, ~$159 compute**

> Phase 2 uses **Hermes-4.3-36B + QLoRA** on A100 80GB (~$1.5/h, ~50-70h).
> Phase 2b distills the fine-tuned 36B → Hermes-3-8B on RTX 4090 (~$0.4/h).
> Only Hermes-3-8B Q4_K_M (~4.9GB) is deployed on Jetson — the 36B never runs on edge.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/ArthurVigier/hermes-jepa-edge
cd hermes-jepa-edge
pip install -e ".[dev]"

# 2. Clone VLA-JEPA dependency
git clone https://github.com/ginwind/VLA-JEPA deps/VLA-JEPA

# 3. Run Phase 0 compat check
python scripts/phase0_compat_check.py \
    --lewm-checkpoint path/to/lewm.pt \
    --dataset libero \
    --n-steps 100

# 4. Run Phase 1 adapter training (Vast.ai RTX 4090)
python scripts/phase1_train_adapter.py \
    --config configs/phase1_adapter.yaml

# 5. Run Phase 2 — Hermes-4.3-36B QLoRA backbone (RunPod A100 80GB)
#    IMPORTANT: Run this on RunPod A100 80GB, NOT on RTX 4090
python scripts/phase2_hermes_backbone.py \
    --config configs/phase2_hermes.yaml

# 5b. Merge LoRA adapter into BF16 weights (prep for distillation)
python scripts/phase2_merge_lora.py \
    --config configs/phase2_hermes.yaml

# 6. Run distillation (Phase 2b)
python scripts/phase2b_distillation.py \
    --config configs/phase2b_distill.yaml

# 7. Export to TensorRT (Jetson)
python scripts/phase3_export_tensorrt.py \
    --config configs/phase3_edge.yaml
```

---

## Repo structure

```
hermes-jepa-edge/
├── configs/                  # YAML experiment configs per phase
├── prompts/                  # Modular Hermes agent prompt library
│   ├── base.xml
│   ├── training.xml
│   ├── edge.xml
│   ├── debug.xml
│   ├── research.xml
│   └── robotics.xml
├── scripts/                  # One entrypoint per pipeline phase
├── src/
│   ├── adapters/             # LeWM → JEPA projection layer
│   ├── distillation/         # 36B → 8B teacher-student
│   ├── edge/                 # TensorRT export + Jetson runtime
│   ├── pipeline/             # VLA-JEPA + Hermes integration
│   └── utils/                # Logging, seeding, VRAM estimation
├── tests/                    # Unit + integration tests
├── experiments/              # Run logs, checkpoints index
└── docs/                     # Architecture notes
```

---

## Hardware requirements

| Stage | Hardware | VRAM | Est. cost | Notes |
|-------|----------|------|-----------|-------|
| Phase 0-1 compat | Vast.ai RTX 4090 | 24GB | ~$0.4/h | Inference only |
| **Phase 2 backbone** | **RunPod A100 80GB** | **~25GB QLoRA** | **~$1.5/h** | **36B QLoRA, NOT RTX 4090** |
| Phase 2b teacher gen | RunPod A100 80GB | ~25GB | ~$1.5/h | One-time, offline |
| Phase 2b student train | Vast.ai RTX 4090 | ~6GB QLoRA | ~$0.4/h | 8B student, comfortable |
| Phase 3 edge export | Jetson Orin Nano | 8GB unified | $0 | TensorRT on device |

---

## References

- VLA-JEPA: https://arxiv.org/abs/2602.10098
- LLM-JEPA: https://arxiv.org/abs/2509.14252
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Hermes-4.3-36B: https://huggingface.co/NousResearch/Hermes-4.3-36B
- VLA-JEPA code: https://github.com/ginwind/VLA-JEPA
- LeWM: Artvv/lewm-habitat-merged (HuggingFace)
