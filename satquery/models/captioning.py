"""Captioning pipeline using real BLIP-2 RS-VLM."""
from __future__ import annotations

from PIL import Image

from .rs_vlm import RSVLM


def generate_caption(model: RSVLM, image_path: str) -> dict:
    """Generate a remote-sensing scene description for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {
            "task": "captioning",
            "error": f"image_load_failed: {exc}",
            "model": model.model_name,
            "evidence": {"image": image_path},
        }

    result = model.caption(image=image)
    return {
        "task": "captioning",
        "caption": result.text,
        "model": model.model_name,
        "evidence": {"image": image_path},
    }
