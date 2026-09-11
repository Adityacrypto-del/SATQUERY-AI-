"""Qwen2.5-VL model wrapper for SATQUERY-AI.

Handles model loading, device selection, and raw generation.
The model is loaded from the HuggingFace cache — weights are
never committed to the repository.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model identifier on HuggingFace Hub.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenVL:
    """Wrapper around Qwen2.5-VL for structured inference.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.

    device:
        Target device. ``None`` triggers auto-detection
        (MPS on Apple Silicon, CUDA if available, else CPU).

    dtype:
        Model dtype string. Defaults to ``"float16"``.

    max_new_tokens:
        Maximum tokens to generate per call.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        dtype: str = "float16",
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self._device_request = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._processor = None

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model and processor are loaded."""
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> str:
        """Resolved device string."""
        if self._device_request is not None:
            return self._device_request
        return _auto_detect_device()

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def load(self) -> None:
        """Load model and processor. Idempotent.

        Raises
        ------
        ImportError
            If ``torch`` or ``transformers`` is not installed.
        """
        if self.is_loaded:
            logger.info("Qwen model already loaded, skipping.")
            return

        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        logger.info(
            "Loading %s on %s (%s)…",
            self.model_name,
            self.device,
            self.dtype,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
        )

        self._model = self._model.to(self.device)

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
        )

        logger.info("Qwen model loaded successfully.")

    # ----------------------------------------------------------
    # Generation
    # ----------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Run inference on a list of chat messages.

        Parameters
        ----------
        messages:
            Chat-format messages compatible with
            ``processor.apply_chat_template``.

        Returns
        -------
        str
            Generated text response.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load() first."
            )

        import torch

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Extract image inputs if present.
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Trim the input tokens from the output.
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]

        text_output = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return text_output[0] if text_output else ""


def _auto_detect_device() -> str:
    """Pick the best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
