"""TerraQ-VL backend: a THIRD-PARTY, VRSBench-adapted model used as Branch 1's VLM.

This is not Branch 1's own adaptation. Any number it produces must be labelled
"TerraQ-VL (third-party, VRSBench-adapted)". Measured results and licence terms:
``outputs/reference_eval/eval_log.md`` and ``reference/terraq_vl/VENDORED.md``.
Licence: research / non-commercial (Qwen Research License + VRSBench CC-BY-NC-4.0).

Loading is lazy, so importing this module never pulls 8 GB of weights. Set the weights location
with ``TERRAQ_WEIGHTS_DIR`` (default ``_ext/terraq-vl-weights`` beside the repo).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO.parent / "_ext" / "terraq-vl-weights"
LABEL = "TerraQ-VL stage-2 checkpoint-2180 (third-party, VRSBench-adapted)"


class TerraQVLBackend:
    """Generation backend: ``generate(image_path, prompt) -> str``."""

    name = LABEL

    def __init__(self, weights_dir: Optional[str] = None, stage: str = "stage2"):
        self.weights_dir = Path(weights_dir or os.environ.get("TERRAQ_WEIGHTS_DIR", DEFAULT_WEIGHTS))
        self.stage = stage
        self._model = None
        self._load_info: dict = {}
        self._unavailable_reason: Optional[str] = None

    # -- availability -----------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.weights_dir / ("stage-2/config/finetune_vrsbench_stage2.yaml" if self.stage == "stage2"
                                   else "stage-1/config/pretrain_vrsbench.yaml")

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / ("stage-2/checkpoints/checkpoint-2180" if self.stage == "stage2"
                                   else "stage-1/checkpoints/checkpoint-3270")

    def is_available(self) -> bool:
        """True when the weights are present and a CUDA device exists."""
        if not self.config_path.exists() or not self.checkpoint_path.exists():
            self._unavailable_reason = (
                f"TerraQ-VL weights not found at {self.weights_dir}. Download with: "
                "huggingface-cli download grKnight/terraq-vl --local-dir <dir> "
                "(or set TERRAQ_WEIGHTS_DIR)."
            )
            return False
        try:
            import torch
        except ImportError:
            self._unavailable_reason = "torch is not installed"
            return False
        if not torch.cuda.is_available():
            self._unavailable_reason = (
                "no CUDA device; the 4-bit (bitsandbytes) path needs a GPU. Measured peak: 3.35 GiB."
            )
            return False
        return True

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    # -- loading / generation --------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(self._unavailable_reason)
        import sys

        vendored = REPO / "reference" / "terraq_vl"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        from load_4bit import load_terraq  # vendored thin wrapper

        self._model, self._load_info = load_terraq(str(self.config_path), str(self.checkpoint_path))

    @property
    def load_info(self) -> dict:
        return dict(self._load_info)

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        self.load()
        from load_4bit import run_inference  # TerraQ-VL's own generation, unchanged

        return run_inference(self._model, image_path, prompt=prompt,
                             max_new_tokens=max_new_tokens, temperature=0.0, device="cuda")
