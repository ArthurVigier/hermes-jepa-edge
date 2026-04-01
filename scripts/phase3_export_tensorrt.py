#!/usr/bin/env python3
"""
scripts/phase3_export_tensorrt.py
Phase 3 entrypoint — export pipeline to TensorRT for Jetson edge deployment.

Usage:
    python scripts/phase3_export_tensorrt.py --config configs/phase3_edge.yaml

Must run ON the Jetson Orin Nano (or a machine with SM_87 CUDA capability).
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edge.tensorrt_export import (
    EdgeExportConfig,
    benchmark_trt_latency,
    build_tensorrt_engine,
    export_lewm_projection_to_onnx,
    print_gguf_conversion_instructions,
    validate_onnx,
)
from src.utils import get_logger
from src.visual_encoders.factory import LeWMProjectionSource, build_visual_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: TensorRT edge export")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export if already done")
    parser.add_argument("--skip-trt", action="store_true", help="Skip TRT build if already done")
    parser.add_argument("--bench-only", action="store_true", help="Only run latency benchmark")
    args = parser.parse_args()

    logger = get_logger("phase3")

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    config = EdgeExportConfig(**cfg_dict)

    # ── Load models ──────────────────────────────────────────────────────────
    if not args.skip_onnx and not args.bench_only:
        logger.info("Loading LeWM encoder + projection for ONNX export")
        visual_source = build_visual_source(config, device="cpu", logger=logger)
        visual_source.eval()
        if not isinstance(visual_source, LeWMProjectionSource):
            logger.error(
                "TensorRT export is only supported for visual_encoder_type='lewm_projection'. "
                "Direct V-JEPA2 mode expects external embeddings and has no local image encoder to export."
            )
            sys.exit(1)

        # ── ONNX export ───────────────────────────────────────────────────
        onnx_path = export_lewm_projection_to_onnx(
            visual_source.encoder,
            visual_source.projection,
            config,
            logger=logger,
        )

        # ── Validate ONNX ─────────────────────────────────────────────────
        if not validate_onnx(str(onnx_path), logger=logger):
            logger.error("ONNX validation failed. Aborting.")
            sys.exit(1)

    # ── TensorRT engine build ─────────────────────────────────────────────
    if not args.skip_trt and not args.bench_only:
        success = build_tensorrt_engine(
            onnx_path=config.onnx_path,
            engine_path=config.trt_engine_path,
            config=config,
            logger=logger,
        )
        if not success:
            logger.error("TensorRT build failed. Aborting.")
            sys.exit(1)

    # ── Latency benchmark ─────────────────────────────────────────────────
    logger.info("\nRunning latency benchmark...")
    results = benchmark_trt_latency(config.trt_engine_path, config, logger=logger)

    if results.get("kill_triggered"):
        logger.error(
            f"Phase 3 KILLED: latency {results['mean_ms']:.1f}ms > 200ms target.\n"
            "Options:\n"
            "  1. Enable INT8 quantization (requires calibration dataset)\n"
            "  2. Reduce input resolution (224→160)\n"
            "  3. Simplify LeWM architecture and retrain\n"
            "  4. Switch to Hermes-3-3B instead of 8B for LLM component"
        )
        sys.exit(1)

    # ── GGUF conversion instructions ─────────────────────────────────────
    print_gguf_conversion_instructions(
        student_checkpoint=config.student_checkpoint,
        output_gguf_path=config.gguf_path,
        logger=logger,
    )

    logger.info("\n✅ Phase 3 complete. Pipeline ready for Jetson deployment.")


if __name__ == "__main__":
    main()
