"""VQA pipeline using real BLIP-2 RS-VLM."""
from __future__ import annotations

from PIL import Image

from .rs_vlm import RSVLM


def answer_vqa(model: RSVLM, image: Image.Image, question: str, image_path: str = "") -> dict:
    """Run VQA on a single image with a natural-language question."""
    result = model.answer(question, image=image)
    return {
        "task": "vqa",
        "answer": result.text,
        "model": model.model_name,
        "confidence": result.confidence,
        "confidence_method": result.confidence_method,
        "evidence": {"image": image_path},
    }
