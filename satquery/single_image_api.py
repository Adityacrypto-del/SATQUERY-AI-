"""Main SatQuery single-image analysis entry point.

Public API:
    analyze_single_image(image_path, query) -> dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from satquery.agent.single_image_router import classify_query
from satquery.models.captioning import generate_caption
from satquery.models.rs_vlm import RSVLM
from satquery.models.vqa import answer_vqa
from satquery.preprocessing.image_loader import load_image, validate_single_image
from satquery.preprocessing.multispectral import to_model_rgb


# Module-level singleton so the model is only loaded once per process
_model_instances: dict[str, RSVLM] = {}


def _get_model(lora_path: str = "") -> RSVLM:
    resolved_lora = str(Path(lora_path).resolve()) if lora_path and Path(lora_path).exists() else ""
    if resolved_lora not in _model_instances:
        if resolved_lora:
            _model_instances[resolved_lora] = RSVLM.load_with_lora(resolved_lora)
        else:
            _model_instances[resolved_lora] = RSVLM(model_name="BLIP-2 (base)", adapted=False)
    return _model_instances[resolved_lora]


def _as_pil_image(rgb: np.ndarray) -> Image.Image:
    """Convert preprocessed RGB data into the exact image sent to the VLM."""
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype("uint8"))


def analyze_single_image(
    image_path: str,
    query: str,
    lora_path: str = "",
) -> Dict[str, Any]:
    """Analyze a single satellite image with a natural-language query.

    Args:
        image_path: Path to .png / .jpg / .tif image.
        query: Natural-language query (VQA question or captioning request).
        lora_path: Optional path to LoRA adapter checkpoint directory.

    Returns:
        Structured result dict with task, answer/caption, model, trace.
    """
    trace: list[str] = []

    # ── 1. Input validation ──────────────────────────────────────────────
    validation = validate_single_image(image_path)
    if not validation.get("ok"):
        return {
            "task": "invalid",
            "error": validation.get("reason", "validation_failed"),
            "execution_trace": ["input_validation_failed"],
        }
    trace.append("input_validation_ok")

    # ── 2. Load & preprocess ─────────────────────────────────────────────
    loaded = load_image(image_path)
    rgb, preprocess_meta = to_model_rgb(loaded.array, loaded.metadata)
    model_image = _as_pil_image(rgb)
    trace.append(f"preprocess_bands:{preprocess_meta['bands_used']}")

    # ── 3. Query classification ──────────────────────────────────────────
    route = classify_query(query)
    trace.append(f"query_classified:{route.task}:{route.rule}")

    # ── 4. Model selection ───────────────────────────────────────────────
    model = _get_model(lora_path)
    trace.append(f"model_selected:{model.model_name}:adapted={model.adapted}")

    # ── 5. Inference ─────────────────────────────────────────────────────
    if route.task == "captioning":
        out = generate_caption(model, model_image, image_path)
        trace.append("captioning_inference_complete")
    else:
        out = answer_vqa(model, model_image, query, image_path)
        trace.append("vqa_inference_complete")

    # ── 6. Assemble output ───────────────────────────────────────────────
    out["adapted"] = model.adapted
    out["preprocessing"] = preprocess_meta
    out["source_metadata"] = loaded.metadata
    out["execution_trace"] = trace
    return out
