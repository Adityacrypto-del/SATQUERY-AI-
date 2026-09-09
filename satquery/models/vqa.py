"""VQA pipeline using real BLIP-2 RS-VLM."""
from __future__ import annotations

from PIL import Image

from .rs_vlm import RSVLM


def answer_vqa(model: RSVLM, image_path: str, question: str) -> dict:
    """Run VQA on a single image with a natural-language question."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {
            "task": "vqa",
            "error": f"image_load_failed: {exc}",
            "model": model.model_name,
            "evidence": {"image": image_path},
        }

    result = model.answer(question, image=image)
    return {
        "task": "vqa",
        "answer": result.text,
        "model": model.model_name,
        "confidence": result.confidence,
        "confidence_method": result.confidence_method,
        "evidence": {"image": image_path},
    }
