"""Load TerraQ-VL -- a THIRD-PARTY, VRSBench-adapted model -- in 4-bit NF4 on a 6 GB GPU.

Nothing here is Branch 1's own model. TerraQ-VL (CLIP ViT-L/14 + Qwen2.5-3B-Instruct + MLP
connector, LoRA on the LLM) is used as a reference baseline only. See VENDORED.md.

This is a thin wrapper over the vendored, unmodified TerraQ-VL code. The single deviation from
their ``inference.load_vlm``: the LLM is loaded 4-bit NF4 through bitsandbytes and placed with
``device_map`` (their release runs it in bf16, and a bnb model cannot be moved with ``.to()``).
Connector, LoRA adapter, vision tower, prompt template and decoding are theirs, unchanged.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402

import vlm_model.language_model as _lm  # noqa: E402
from inference import run_inference  # noqa: E402,F401  (re-exported: their generation, unchanged)
from training.checkpoint import load_connector_checkpoint, load_lora_adapter  # noqa: E402
from vlm_model.vlm import VLMForCausalLM  # noqa: E402


class _NF4CausalLM:
    """Stand-in for ``AutoModelForCausalLM`` inside vlm_model.language_model during construction."""

    @staticmethod
    def from_pretrained(name, torch_dtype=torch.bfloat16, **kwargs):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            name, dtype=torch_dtype, quantization_config=quant, device_map={"": 0},
            low_cpu_mem_usage=True, **kwargs,
        )


def load_terraq(config_path: str, checkpoint_dir: str) -> Tuple[VLMForCausalLM, Dict[str, Any]]:
    """Build TerraQ-VL from its config with a 4-bit LLM and load the released connector (+ LoRA)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.cuda.reset_peak_memory_stats()
    original = _lm.AutoModelForCausalLM
    _lm.AutoModelForCausalLM = _NF4CausalLM
    try:
        model = VLMForCausalLM(config)
    finally:
        _lm.AutoModelForCausalLM = original

    load_connector_checkpoint(model.connector, checkpoint_dir)
    lora_loaded = False
    if getattr(model.language_model, "is_lora", False):
        lora_loaded = load_lora_adapter(model.language_model.model, checkpoint_dir)
        if not lora_loaded:
            raise RuntimeError(
                f"config has a LoRA block but {checkpoint_dir} has no lora/ adapter; "
                "refusing to evaluate randomly initialised adapters"
            )

    model.vision_encoder.to("cuda")
    model.connector.to("cuda")
    model.eval()

    info = {
        "config": os.path.relpath(config_path, HERE) if config_path.startswith(HERE) else config_path,
        "checkpoint": checkpoint_dir,
        "llm_quantization": "bitsandbytes 4-bit NF4, double-quant, bf16 compute (release: bf16)",
        "lora_loaded": lora_loaded,
        "vram_after_load_bytes": torch.cuda.max_memory_allocated(),
    }
    return model, info
