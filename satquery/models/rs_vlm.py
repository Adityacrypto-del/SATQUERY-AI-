"""Real RS-VLM loader using BLIP-2 (Salesforce/blip2-opt-2.7b).

Falls back gracefully if model weights are unavailable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class ModelResult:
    text: str
    confidence: Optional[float] = None
    confidence_method: Optional[str] = None


def _heuristic_confidence(text: str) -> tuple[float | None, str]:
    """Estimate a simple heuristic confidence from the generated text.

    Heuristics:
      - Empty / very short text -> low confidence.
      - Text contains hedging phrases ("maybe", "not sure", "unclear") -> medium.
      - Longer, decisive text -> higher confidence.

    This is explicitly *not* calibrated probability.  The method name is
    recorded so callers know the confidence is heuristic.
    """
    if not text or not text.strip():
        return 0.0, "heuristic_empty"
    length = len(text.strip())
    hedge_words = {"maybe", "perhaps", "unclear", "not sure", "possibly", "unknown"}
    tokens = set(text.lower().split())
    hedge_overlap = tokens & hedge_words
    if length < 5:
        return 0.2, "heuristic_short"
    if hedge_overlap:
        return 0.35, "heuristic_hedging"
    if length > 60:
        return 0.7, "heuristic_lengthy"
    return 0.55, "heuristic_default"


class RSVLM:
    """Remote-sensing vision-language model wrapper.

    Uses BLIP-2 (blip2-opt-2.7b) as the backbone, adapted for RS tasks.
    On Apple Silicon / CPU, runs in float32 mode.
    On CUDA, uses float16 for speed.
    """

    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    _processor = None
    _model = None
    _device = None

    def __init__(self, model_name: str = "BLIP-2-RS", adapted: bool = False) -> None:
        self.model_name = model_name
        self.adapted = adapted
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"

        dtype = torch.float16 if device_str == "cuda" else torch.float32

        print(f"[RSVLM] Loading {self.MODEL_ID} on {device_str} with {dtype}...")
        token = os.environ.get("HF_TOKEN")

        processor = Blip2Processor.from_pretrained(
            self.MODEL_ID,
            token=token,
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            device_map=device_str if device_str != "mps" else None,
            token=token,
        )
        if device_str == "mps":
            model = model.to(device_str)

        model.eval()

        RSVLM._processor = processor
        RSVLM._model = model
        RSVLM._device = device_str
        self._loaded = True
        print(f"[RSVLM] Model loaded successfully.")

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        import torch

        self._load()

        inputs = RSVLM._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        # Move tensors to device
        inputs = {k: v.to(RSVLM._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = RSVLM._model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=3,
                temperature=1.0,
            )

        generated_text = RSVLM._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return generated_text

    def answer(self, question: str, image: Optional[Image.Image] = None) -> ModelResult:
        if image is None:
            return ModelResult(
                text="[no image provided]",
                confidence=None,
                confidence_method="not_available",
            )
        # RS-style prompt for VQA
        prompt = (
            f"You are analyzing a remote-sensing satellite image.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        text = self._run_inference(image, prompt)
        conf, method = _heuristic_confidence(text)
        return ModelResult(text=text, confidence=conf, confidence_method=method)

    def caption(self, image: Optional[Image.Image] = None) -> ModelResult:
        if image is None:
            return ModelResult(text="[no image provided]")
        prompt = (
            "Describe this satellite remote-sensing image in detail. "
            "Include land cover, major visible objects, spatial context, and scene characteristics:"
        )
        text = self._run_inference(image, prompt)
        conf, method = _heuristic_confidence(text)
        return ModelResult(text=text, confidence=conf, confidence_method=method)

    @classmethod
    def load_with_lora(cls, lora_path: str) -> "RSVLM":
        """Load BLIP-2 with a LoRA adapter checkpoint."""
        from peft import PeftModel

        instance = cls(model_name="BLIP-2-RS-LoRA", adapted=True)
        instance._load()  # Load base first

        print(f"[RSVLM] Applying LoRA adapter from {lora_path} ...")
        RSVLM._model = PeftModel.from_pretrained(RSVLM._model, lora_path)
        RSVLM._model.eval()
        print(f"[RSVLM] LoRA adapter applied.")
        return instance
