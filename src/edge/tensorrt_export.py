"""
src/edge/tensorrt_export.py
Phase 3 — Export pipeline to TensorRT for Jetson Orin Nano deployment.

Components exported:
  1. LeWM encoder + projection → ONNX → TensorRT engine (INT8)
  2. Hermes-3-8B student → llama.cpp GGUF Q4_K_M (via separate script)

Kill criterion: latency > 200ms/step on Jetson.

Note: TensorRT export must run ON the Jetson or a machine with the same CUDA compute capability.
      Jetson Orin Nano: SM_87 (Ampere).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils import check_kill_criterion, get_logger


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class EdgeExportConfig:
    # Input checkpoints
    lewm_checkpoint: str
    projection_checkpoint: str
    student_checkpoint: str               # Phase 2b output (Hermes-3-8B fine-tuned)
    visual_encoder_type: str = "lewm_projection"  # lewm_projection | vjepa2_direct
    vjepa2_embeddings_key: str = "vjepa2_embeddings"

    # Export paths
    output_dir: str = "experiments/phase3_edge"
    onnx_path: str = "experiments/phase3_edge/lewm_projection.onnx"
    trt_engine_path: str = "experiments/phase3_edge/lewm_projection.trt"
    gguf_path: str = "experiments/phase3_edge/hermes3_8b_q4km.gguf"

    # Model dims
    lewm_dim: int = 256
    vjepa2_dim: int = 1024
    input_channels: int = 3
    input_height: int = 224
    input_width: int = 224

    # TensorRT settings
    fp16: bool = True                     # FP16 for non-critical layers
    int8: bool = False                    # INT8 requires calibration dataset
    workspace_gb: int = 4                 # Max TRT workspace on Jetson (4GB safe)
    batch_size: int = 1                   # Edge: always batch=1

    # Latency target
    latency_budget_ms: float = 50.0       # LeWM+projection budget (LLM is separate)
    n_warmup: int = 10
    n_bench: int = 100

    device: str = "cuda"


# ─── ONNX export ──────────────────────────────────────────────────────────────

def export_lewm_projection_to_onnx(
    lewm_encoder: nn.Module,
    projection: nn.Module,
    config: EdgeExportConfig,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Export LeWM encoder + projection layer to ONNX with static shapes.

    Args:
        lewm_encoder: Trained LeWM encoder module.
        projection: Trained projection layer (LeWMProjection).
        config: EdgeExportConfig.
        logger: Optional logger.

    Returns:
        Path to saved ONNX file.
    """
    if logger is None:
        logger = get_logger(__name__)

    output_path = Path(config.onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    # Wrap into a single fused module for clean ONNX graph
    class FusedLeWMProjection(nn.Module):
        def __init__(self, encoder: nn.Module, proj: nn.Module) -> None:
            super().__init__()
            self.encoder = encoder
            self.proj = proj

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            emb = self.encoder(frames)   # [B, lewm_dim]
            return self.proj(emb)        # [B, vjepa2_dim]

    fused = FusedLeWMProjection(lewm_encoder, projection).to(device).eval()

    # Static shape dummy input — required for TensorRT compatibility
    dummy_input = torch.randn(
        config.batch_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        device=device,
    )

    logger.info(f"Exporting to ONNX: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            fused,
            dummy_input,
            str(output_path),
            input_names=["frames"],
            output_names=["visual_embedding"],
            opset_version=17,
            do_constant_folding=True,
            # Static shapes — no dynamic axes (required for TensorRT on Jetson)
        )

    logger.info(f"ONNX export complete → {output_path}")
    return output_path


def validate_onnx(onnx_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """Validate ONNX graph integrity."""
    if logger is None:
        logger = get_logger(__name__)
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX validation passed: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


# ─── TensorRT engine build ────────────────────────────────────────────────────

def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    config: EdgeExportConfig,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Build TensorRT engine from ONNX model.

    Must run on the target device (Jetson Orin Nano, SM_87).

    Args:
        onnx_path: Path to ONNX file.
        engine_path: Path to save TRT engine.
        config: EdgeExportConfig.
        logger: Optional logger.

    Returns:
        True if successful.
    """
    if logger is None:
        logger = get_logger(__name__)

    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "TensorRT not found. Install: pip install tensorrt\n"
            "On Jetson: sudo apt install python3-libnvinfer-dev"
        )
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                logger.error(f"TRT parse error: {parser.get_error(err)}")
            return False

    config_trt = builder.create_builder_config()
    config_trt.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        config.workspace_gb * (1 << 30),
    )

    if config.fp16 and builder.platform_has_fast_fp16:
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled for TensorRT")

    if config.int8 and builder.platform_has_fast_int8:
        config_trt.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 enabled for TensorRT (requires calibrator)")

    logger.info("Building TensorRT engine (this may take several minutes on Jetson)...")
    engine = builder.build_serialized_network(network, config_trt)

    if engine is None:
        logger.error("TensorRT engine build failed")
        return False

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine)

    logger.info(f"TensorRT engine saved → {engine_path}")
    return True


# ─── Latency benchmark ────────────────────────────────────────────────────────

def benchmark_trt_latency(
    engine_path: str,
    config: EdgeExportConfig,
    logger: Optional[logging.Logger] = None,
) -> dict[str, float]:
    """
    Benchmark TensorRT engine latency on current device.

    Args:
        engine_path: Path to .trt engine file.
        config: EdgeExportConfig.
        logger: Optional logger.

    Returns:
        Dict with mean_ms, p50_ms, p95_ms, p99_ms, kill_triggered.
    """
    if logger is None:
        logger = get_logger(__name__)

    try:
        import tensorrt as trt
        import numpy as np
    except ImportError:
        logger.error("TensorRT not available for benchmarking")
        return {}

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    dummy = torch.randn(
        config.batch_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        device=config.device,
    ).half() if config.fp16 else torch.randn(
        config.batch_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        device=config.device,
    )
    output = torch.empty(config.batch_size, config.vjepa2_dim, device=config.device)

    bindings = [dummy.data_ptr(), output.data_ptr()]

    # Warmup
    for _ in range(config.n_warmup):
        context.execute_v2(bindings)
    torch.cuda.synchronize()

    # Benchmark
    latencies_ms: list[float] = []
    for _ in range(config.n_bench):
        start = time.perf_counter()
        context.execute_v2(bindings)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    import numpy as np
    arr = np.array(latencies_ms)
    results = {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(arr.max()),
    }

    logger.info(
        f"TRT Latency (LeWM+proj): "
        f"mean={results['mean_ms']:.1f}ms  "
        f"p50={results['p50_ms']:.1f}ms  "
        f"p95={results['p95_ms']:.1f}ms"
    )

    kill_triggered = check_kill_criterion(
        phase="phase3",
        metric_name="latency_ms",
        metric_value=results["mean_ms"],
        logger=logger,
    )
    results["kill_triggered"] = kill_triggered

    return results


# ─── GGUF conversion helper ───────────────────────────────────────────────────

def print_gguf_conversion_instructions(
    student_checkpoint: str,
    output_gguf_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print llama.cpp conversion commands for Hermes-3-8B → GGUF Q4_K_M."""
    if logger is None:
        logger = get_logger(__name__)

    logger.info(
        "\n"
        "═══ GGUF Conversion (Hermes-3-8B → Q4_K_M) ═══\n"
        "Run on a machine with llama.cpp installed:\n\n"
        f"# 1. Convert HF → GGUF FP16\n"
        f"python llama.cpp/convert_hf_to_gguf.py {student_checkpoint} \\\n"
        f"    --outtype f16 --outfile hermes3_8b_f16.gguf\n\n"
        f"# 2. Quantize to Q4_K_M (~4.9GB, recommended for Jetson 8GB)\n"
        f"./llama.cpp/build/bin/llama-quantize \\\n"
        f"    hermes3_8b_f16.gguf {output_gguf_path} Q4_K_M\n\n"
        f"# 3. Verify\n"
        f"./llama.cpp/build/bin/llama-cli \\\n"
        f"    -m {output_gguf_path} \\\n"
        f"    -p 'Move arm to [0.3, 0.1, 0.4, 0, 0, 0]' \\\n"
        f"    -n 100 --temp 0.6\n\n"
        f"# Expected footprint breakdown on Jetson 8GB:\n"
        f"#   Hermes-3-8B Q4_K_M : ~4.9 GB\n"
        f"#   LeWM + projection TRT : ~0.3 GB\n"
        f"#   KV cache (1024 ctx)  : ~0.5 GB\n"
        f"#   OS + overhead        : ~1.0 GB\n"
        f"#   TOTAL                : ~6.7 GB  ✅ within 8GB\n"
    )
