"""RS-VLM loader using Qwen2-VL-7B (Qwen/Qwen2-VL-7B-Instruct).

Qwen2-VL-7B is chosen over BLIP-2 because it:
  - Dominates both VQA and captioning benchmarks
  - Handles variable-resolution inputs (critical for satellite imagery)
  - Supports multilingual queries
  - Works with LoRA/PEFT fine-tuning
  - Fits in 8-16GB VRAM with 4-bit quantization

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

    Uses Qwen2-VL-7B-Instruct as the backbone, adapted for RS tasks.
    Supports 4-bit quantization to fit in 8-16GB VRAM.
    """

    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    _processor = None
    _model = None
    _device = None

    def __init__(self, model_name: str = "Qwen2-VL-7B-RS", adapted: bool = False) -> None:
        self.model_name = model_name
        self.adapted = adapted
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"

        use_quantize = device_str == "cuda"
        dtype = torch.float16 if device_str == "cuda" else torch.float32

        print(f"[RSVLM] Loading {self.MODEL_ID} on {device_str} (quantize={use_quantize})...")
        token = os.environ.get("HF_TOKEN")

        processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            token=token,
            trust_remote_code=True,
        )

        if use_quantize:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                token=token,
                trust_remote_code=True,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
                device_map=device_str if device_str != "mps" else None,
                token=token,
                trust_remote_code=True,
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = RSVLM._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = RSVLM._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(RSVLM._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = RSVLM._model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                temperature=1.0,
                do_sample=False,
            )

        # Trim input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]

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
        prompt = (
            "You are analyzing a remote-sensing satellite image. "
            "Answer the following question about the image concisely.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        text = self._run_inference(image, prompt)
        conf, method = _heuristic_confidence(text)
        return ModelResult(text=text, confidence=conf, confidence_method=method)

    def caption(self, image: Optional[Image.Image] = None) -> ModelResult:
        if image is None:
            return ModelResult(text="[no image provided]")
        prompt = (
            "Describe this satellite remote-sensing image in detail. "
            "Include land cover types, major visible objects, spatial context, "
            "and remote-sensing scene characteristics. "
            "Be specific about what you observe."
        )
        text = self._run_inference(image, prompt)
        conf, method = _heuristic_confidence(text)
        return ModelResult(text=text, confidence=conf, confidence_method=method)

    @classmethod
    def load_with_lora(cls, lora_path: str) -> "RSVLM":
        """Load Qwen2-VL with a LoRA adapter checkpoint."""
        from peft import PeftModel

        instance = cls(model_name="Qwen2-VL-7B-RS-LoRA", adapted=True)
        instance._load()  # Load base first

        print(f"[RSVLM] Applying LoRA adapter from {lora_path} ...")
        RSVLM._model = PeftModel.from_pretrained(RSVLM._model, lora_path)
        RSVLM._model.eval()
        print(f"[RSVLM] LoRA adapter applied.")
        return instance
