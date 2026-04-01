"""
src/pipeline/hermes_vla.py
Phase 2 — VLA-JEPA with Hermes-4.3-36B as LLM backbone (training on A100 80GB).

Architecture:
  LeWM encoder → projection → VLA-JEPA predictor (init from Hermes-4.3-36B layers 8-16)
  Hermes-4.3-36B (QLoRA, 4-bit) → action token generation → structured tool calls

Training strategy:
  - QLoRA on Hermes-4.3-36B: ~25GB VRAM → requires A100 80GB (RunPod)
  - NOT suitable for RTX 4090 (24GB, too tight with activations)
  - LoRA adapter (~100-500MB) merged after training
  - Distilled to Hermes-3-8B in Phase 2b for edge deployment

Dual loss: NTP (next token prediction) + JEPA alignment (LLM-JEPA style)
Kill criterion: task_success < 30% on LIBERO.

VRAM budget (A100 80GB):
  - Hermes-4.3-36B weights (NF4): ~18GB
  - LoRA adapters + optimizer:    ~3GB
  - Activations (bs=4, seq=512):  ~4GB
  - LeWM + projection:            ~1GB
  - TOTAL:                        ~26GB  (54GB headroom on A100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    StepTimer,
    check_kill_criterion,
    check_vram_feasibility,
    estimate_vram,
    get_logger,
    seed_everything,
)


# ─── Action tokens ────────────────────────────────────────────────────────────

# Hermes native tool-call action schema for robotics
ROBOTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_arm",
            "description": "Move robot arm to a 6-DOF target pose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_pose": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[x, y, z, rx, ry, rz] in robot base frame",
                    },
                    "velocity": {
                        "type": "number",
                        "description": "Normalized velocity 0.0–1.0",
                    },
                },
                "required": ["target_pose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grasp",
            "description": "Close gripper to grasp an object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "force_n": {"type": "number", "description": "Gripper force in Newtons"},
                    "width_m": {"type": "number", "description": "Gripper aperture in meters"},
                },
                "required": ["force_n", "width_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "release",
            "description": "Open gripper to release an object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width_m": {"type": "number", "description": "Target aperture in meters"},
                },
                "required": ["width_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_world_state",
            "description": "Query current world state embedding from LeWM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["rgb", "depth", "proprioception"],
                    }
                },
                "required": ["modality"],
            },
        },
    },
]


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class HermesVLAConfig:
    # Model paths
    # Backbone: Hermes-4.3-36B — requires A100 80GB for QLoRA training (~25GB VRAM)
    # NOT suitable for RTX 4090 (too tight). Use RunPod A100 80GB.
    hermes_model_id: str = "NousResearch/Hermes-4.3-36B"
    visual_encoder_type: str = "lewm_projection"  # lewm_projection | vjepa2_direct
    vjepa2_embeddings_key: str = "vjepa2_embeddings"
    lewm_checkpoint: str = ""
    projection_checkpoint: str = ""       # Phase 1 output
    output_dir: str = "experiments/phase2"

    # QLoRA config (required for 36B on A100 80GB)
    use_qlora: bool = True                # Must be True for 36B
    qlora_bits: int = 4                   # NF4 quantization
    qlora_r: int = 16                     # LoRA rank
    qlora_alpha: int = 32                 # LoRA alpha
    qlora_dropout: float = 0.05
    qlora_target_modules: list = None     # None = auto-detect (q_proj, v_proj, etc.)

    # VLA-JEPA predictor init
    predictor_init_layers: tuple[int, int] = (8, 16)  # Hermes layers to use as predictor

    # Architecture dims
    lewm_dim: int = 256
    vjepa2_dim: int = 1024

    # Dual loss weights
    lambda_ntp: float = 1.0             # Weight for next-token prediction loss
    lambda_jepa: float = 0.5            # Weight for JEPA alignment loss
    jepa_loss_dropout: float = 0.1      # Randomly drop JEPA loss (LLM-JEPA trick)

    # Training — calibrated for A100 80GB
    batch_size: int = 4                 # Safe on A100 with 36B QLoRA
    lr: float = 2e-4                    # Higher LR ok with QLoRA (only adapters trained)
    weight_decay: float = 1e-4
    n_steps: int = 5000
    kill_check_step: int = 2000
    log_every: int = 50
    save_every: int = 500
    gradient_checkpointing: bool = True
    device: str = "cuda"
    seed: int = 42
    available_vram_gb: float = 80.0     # A100 80GB — do NOT run on RTX 4090


# ─── VLA-JEPA predictor ───────────────────────────────────────────────────────

class HermesVLAPredictor(nn.Module):
    """
    VLA-JEPA predictor initialized from Hermes-4.3-36B transformer layers.

    Takes: visual embedding (from LeWM projection) + language instruction tokens
    Predicts: target latent embedding of future world state

    Architecture mirrors VLA-JEPA paper: non-autoregressive latent prediction.
    Layers 8-16 of Hermes-4.3-36B are extracted and used as the predictor backbone.
    """

    def __init__(
        self,
        hermes_model: AutoModelForCausalLM,
        predictor_layers: tuple[int, int],
        visual_dim: int = 1024,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()

        # Extract transformer layers [start, end) from Hermes
        all_layers = hermes_model.model.layers
        start, end = predictor_layers
        self.transformer_layers = nn.ModuleList(all_layers[start:end])

        hidden_size = hermes_model.config.hidden_size

        # Visual input projection → transformer hidden dim
        self.visual_proj = nn.Linear(visual_dim, hidden_size)

        # Output head: predict target embedding
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(
        self,
        visual_emb: torch.Tensor,        # [B, visual_dim] from LeWM projection
        instruction_hidden: torch.Tensor, # [B, T, hidden_size] from Hermes encoder
    ) -> torch.Tensor:
        """
        Args:
            visual_emb: [B, visual_dim]
            instruction_hidden: [B, T, hidden_size]

        Returns:
            predicted_target: [B, output_dim] — predicted future state embedding
        """
        B = visual_emb.shape[0]

        # Project visual to hidden dim and prepend as first token
        visual_token = self.visual_proj(visual_emb).unsqueeze(1)  # [B, 1, H]
        hidden = torch.cat([visual_token, instruction_hidden], dim=1)  # [B, T+1, H]

        # Pass through predictor transformer layers
        for layer in self.transformer_layers:
            layer_out = layer(hidden)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Use first token (visual) as prediction output
        pred = self.output_head(hidden[:, 0, :])  # [B, output_dim]
        return pred


# ─── Dual loss ────────────────────────────────────────────────────────────────

def dual_loss(
    ntp_logits: torch.Tensor,
    ntp_labels: torch.Tensor,
    predicted_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    lambda_ntp: float = 1.0,
    lambda_jepa: float = 0.5,
    jepa_loss_dropout: float = 0.1,
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute dual loss: NTP + JEPA alignment (LLM-JEPA style).

    Args:
        ntp_logits: [B, T, vocab_size]
        ntp_labels: [B, T] token ids
        predicted_embedding: [B, D] predictor output
        target_embedding: [B, D] EMA target (stop-gradient)
        lambda_ntp / lambda_jepa: loss weights
        jepa_loss_dropout: Probability of dropping JEPA loss per step (efficiency trick)
        training: If False, always compute both losses (for eval)

    Returns:
        (total_loss, metrics_dict)
    """
    # NTP loss
    ntp_loss = F.cross_entropy(
        ntp_logits.view(-1, ntp_logits.size(-1)),
        ntp_labels.view(-1),
        ignore_index=-100,
    )

    # JEPA alignment loss (cosine MSE in normalized space)
    use_jepa = not training or (torch.rand(1).item() > jepa_loss_dropout)
    if use_jepa:
        pred_norm = F.normalize(predicted_embedding, dim=-1)
        tgt_norm = F.normalize(target_embedding.detach(), dim=-1)
        jepa_loss = F.mse_loss(pred_norm, tgt_norm)
    else:
        jepa_loss = torch.tensor(0.0, device=ntp_logits.device)

    total = lambda_ntp * ntp_loss + lambda_jepa * jepa_loss

    metrics = {
        "loss/total": total.item(),
        "loss/ntp": ntp_loss.item(),
        "loss/jepa": jepa_loss.item(),
    }
    return total, metrics


# ─── Model builder ────────────────────────────────────────────────────────────

def build_hermes_vla(
    config: HermesVLAConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, HermesVLAPredictor]:
    """
    Load Hermes-4.3-36B with QLoRA (4-bit NF4) and build the VLA predictor.

    Requires A100 80GB. VRAM breakdown:
      - NF4 weights: ~18GB
      - LoRA adapters: ~1GB
      - Optimizer states: ~2GB
      - Activations: ~4GB
      - Total: ~25GB (safe on A100 80GB, OOM risk on RTX 4090)

    Returns:
        (hermes_model_with_lora, tokenizer, predictor)
    """
    if logger is None:
        logger = get_logger(__name__)

    # Hard guard: warn if not on A100-class hardware
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 48:
            logger.error(
                f"Detected only {vram_gb:.0f}GB VRAM. "
                f"Hermes-4.3-36B QLoRA requires A100 80GB (~25GB VRAM for training). "
                f"On RTX 4090 (24GB) this will OOM. "
                f"Use RunPod A100 80GB. Aborting."
            )
            raise RuntimeError(
                f"Insufficient VRAM ({vram_gb:.0f}GB) for Hermes-4.3-36B QLoRA training. "
                f"Minimum 48GB required, A100 80GB recommended."
            )

    logger.info(f"Loading {config.hermes_model_id} with QLoRA (4-bit NF4)")

    tokenizer = AutoTokenizer.from_pretrained(
        config.hermes_model_id,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if config.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",          # NF4 is optimal for LLM weights
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,     # Double quantization saves ~0.4GB
            )

            hermes = AutoModelForCausalLM.from_pretrained(
                config.hermes_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Prepare for k-bit training (cast LayerNorm to fp32, etc.)
            hermes = prepare_model_for_kbit_training(
                hermes,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )

            # Inject LoRA adapters
            target_modules = config.qlora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            lora_config = LoraConfig(
                r=config.qlora_r,
                lora_alpha=config.qlora_alpha,
                lora_dropout=config.qlora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            hermes = get_peft_model(hermes, lora_config)
            hermes.print_trainable_parameters()

        except ImportError as e:
            raise ImportError(
                "QLoRA requires bitsandbytes and peft. "
                "Install with: pip install bitsandbytes peft"
            ) from e
    else:
        # Full BF16 — only use this if you have >80GB VRAM
        logger.warning(
            "use_qlora=False: loading Hermes-4.3-36B in BF16 (~72GB). "
            "This requires an H100 or multi-GPU setup."
        )
        hermes = AutoModelForCausalLM.from_pretrained(
            config.hermes_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if config.gradient_checkpointing:
            hermes.gradient_checkpointing_enable()

    # Build VLA predictor from Hermes layers
    predictor = HermesVLAPredictor(
        hermes_model=hermes,
        predictor_layers=config.predictor_init_layers,
        visual_dim=config.vjepa2_dim,
        output_dim=config.vjepa2_dim,
    )

    # Log VRAM
    budget = estimate_vram(
        hermes,
        batch_size=config.batch_size,
        dtype=torch.bfloat16,
        frozen=False,  # LoRA adapters are trainable
    )
    check_vram_feasibility(budget, available_gb=config.available_vram_gb)

    logger.info(
        f"Hermes-4.3-36B QLoRA loaded. "
        f"Trainable params: LoRA adapters only (~0.1-1% of total). "
        f"Expected VRAM: ~25GB on A100 80GB."
    )

    return hermes, tokenizer, predictor


# ─── Action token formatter ───────────────────────────────────────────────────

def format_tool_call_prompt(
    instruction: str,
    world_state_description: str,
    tokenizer: AutoTokenizer,
    enable_thinking: bool = False,
) -> str:
    """
    Format a robotic task as a Hermes-4.3 ChatML prompt with tool definitions.

    Hermes-4.3-36B uses Llama-3 chat format (NOT ChatML im_start/im_end).
    Tool calls are emitted as <tool_call>{...}</tool_call> tags.

    Args:
        instruction: Natural language task ("pick up the red cube")
        world_state_description: Current world state from LeWM decoder
        tokenizer: Hermes-4.3-36B tokenizer
        enable_thinking: If True, inject thinking=True system prompt for <think> traces

    Returns:
        Formatted prompt string ready for tokenization.
    """
    import json

    tools_str = json.dumps(ROBOTIC_TOOLS, indent=2)

    system_content = (
        f"You are a robotic planning agent with tool-calling capability. "
        f"Current world state: {world_state_description}\n\n"
        f"Available tools:\n{tools_str}"
    )

    # Hermes-4.3-36B thinking mode system prefix (optional)
    if enable_thinking:
        thinking_prefix = (
            "You are a deep thinking AI. Enclose your reasoning inside "
            "<think>...</think> tags before responding. "
        )
        system_content = thinking_prefix + system_content

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": instruction},
    ]

    # Hermes-4.3-36B uses Llama-3 chat format
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        tools=ROBOTIC_TOOLS,  # Native tool injection if supported
    )
