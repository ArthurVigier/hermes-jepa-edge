.PHONY: install install-edge test test-fast coverage lint fmt
.PHONY: phase0 phase1 phase2 phase2-dry phase2-merge teacher phase2b phase3 bench clean

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev,train]"

install-edge:
	pip install -e ".[edge]"

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

coverage:
	pytest tests/ --cov=src --cov-report=term-missing

# ── Lint ─────────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

fmt:
	ruff format src/ tests/ scripts/

# ── Pipeline phases ──────────────────────────────────────────────────────────
# Phase 0-1: Vast.ai RTX 4090 (~$0.4/h)
# Phase 2:   RunPod A100 80GB  (~$1.5/h)  ← REQUIRED for 36B QLoRA
# Phase 2b:  Vast.ai RTX 4090 (~$0.4/h)
# Phase 3:   Jetson Orin Nano  ($0)

phase0:
	python scripts/phase0_compat_check.py \
		--lewm-checkpoint $(LEWM_CKPT) \
		--dataset libero \
		--n-steps 100

phase1:
	python scripts/phase1_train_adapter.py \
		--config configs/phase1_adapter.yaml

# Phase 2: Hermes-4.3-36B QLoRA backbone — REQUIRES A100 80GB
phase2:
	@echo "⚠️  REQUIRES RunPod A100 80GB — do NOT run on RTX 4090 (will OOM)"
	python scripts/phase2_hermes_backbone.py \
		--config configs/phase2_hermes.yaml

# Dry run: verify setup in 5 steps without full training
phase2-dry:
	python scripts/phase2_hermes_backbone.py \
		--config configs/phase2_hermes.yaml \
		--dry-run

# Merge LoRA adapter → BF16 (required before Phase 2b)
phase2-merge:
	@echo "⚠️  REQUIRES A100 80GB — merge needs ~72GB RAM for BF16"
	python scripts/phase2_merge_lora.py \
		--lora-dir experiments/phase2/lora_final \
		--output-dir experiments/phase2/merged_bf16

# Generate teacher trajectories (Hermes-4.3-36B + LoRA, A100 80GB)
teacher:
	python scripts/generate_teacher_trajectories.py \
		--config configs/phase2b_distill.yaml \
		--n $(N_TRAJ)

# Phase 2b: Distill 36B → 8B student (RTX 4090)
phase2b:
	python scripts/phase2b_distillation.py \
		--config configs/phase2b_distill.yaml

# Phase 3: TensorRT export (Jetson)
phase3:
	python scripts/phase3_export_tensorrt.py \
		--config configs/phase3_edge.yaml

bench:
	python scripts/phase3_export_tensorrt.py \
		--config configs/phase3_edge.yaml \
		--bench-only

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
