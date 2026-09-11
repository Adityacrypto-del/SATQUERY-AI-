"""
inference.py — Public inference API for the OpticalSAR specialist model.

This module is the API boundary described in the SatQuery AI design:

    Optical + SAR images
            ↓
    OpticalSARSpecialist.encode_pair(optical, sar)
            ↓
    structured result / fused embedding
            ↓
    Common Reasoning Model (downstream)

Usage:
    from inference import OpticalSARSpecialist

    specialist = OpticalSARSpecialist.load_checkpoint("checkpoints/best.pt")

    result = specialist.encode_pair(
        optical=optical_tensor,   # (1, 13, H, W) or (13, H, W)
        sar=sar_tensor,           # (1, 2,  H, W) or (2,  H, W)
    )

    fused = result["fused_embedding"]  # (512,) numpy array — feed to reasoning model

No LLM, no natural language, no GUI in this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig
from models.contrastive_model import OpticalSARContrastiveModel
from preprocessing import build_preprocessors
from utils import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: ensure tensor is (B, C, H, W) on device
# ---------------------------------------------------------------------------

def _ensure_4d(
    x: Union[torch.Tensor, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)   # (C, H, W) → (1, C, H, W)
    if x.ndim != 4:
        raise ValueError(f"Expected 3-D or 4-D tensor, got shape {tuple(x.shape)}")
    return x.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class OpticalSARSpecialist:
    """
    Inference wrapper for the trained OpticalSARContrastiveModel.

    This is the only class that downstream SatQuery AI modules need to import.
    It is intentionally thin: load a checkpoint, call encode_pair.
    """

    def __init__(
        self,
        model:  OpticalSARContrastiveModel,
        config: ModelConfig,
        device: torch.device,
    ) -> None:
        self.model  = model.eval()
        self.config = config
        self.device = device
        self._s2_pre, self._s1_pre = build_preprocessors(config)

    # ---- Construction ----

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "OpticalSARSpecialist":
        """
        Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: Path to a .pt checkpoint saved by train.py.
            device:          Target device.  Defaults to best available.

        Returns:
            Ready-to-use OpticalSARSpecialist instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = get_device()

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Re-build config from checkpoint
        if "config" in state:
            cfg = ModelConfig.from_dict(state["config"])
        else:
            logger.warning("No config in checkpoint; using default ModelConfig.")
            cfg = ModelConfig()

        # Build model with saved architecture settings
        model = OpticalSARContrastiveModel(
            embedding_dim              = cfg.embedding_dim,
            projection_dim             = cfg.projection_dim,
            encoder_type               = cfg.encoder_type,
            use_cross_attention_fusion = cfg.use_cross_attention_fusion,
            optical_channels           = cfg.optical_channels,
            sar_channels               = cfg.sar_channels,
            pretrained_encoders        = False,   # weights come from checkpoint
        ).to(device)

        model.load_state_dict(state["model_state"], strict=True)
        logger.info(
            "Loaded OpticalSARSpecialist from %s (epoch %d)",
            checkpoint_path, state.get("epoch", -1),
        )

        return cls(model=model, config=cfg, device=device)

    # ---- Core inference ----

    @torch.no_grad()
    def encode_pair(
        self,
        optical: Union[torch.Tensor, np.ndarray],   # (13, H, W) or (B, 13, H, W)
        sar:     Union[torch.Tensor, np.ndarray],   # (2, H, W)  or (B, 2,  H, W)
        preprocess: bool = False,                   # True if raw (un-normalised) arrays
    ) -> Dict[str, Any]:
        """
        Encode an optical + SAR pair and return structured result.

        Args:
            optical:    Sentinel-2 tensor/array.  If preprocess=True, pass raw DN
                        integers; otherwise pass already-normalised float tensors.
            sar:        Sentinel-1 tensor/array.
            preprocess: Apply S2/S1 preprocessing (scale + z-score) to raw arrays.

        Returns:
            {
                "optical_embedding":      np.ndarray (B, embedding_dim)
                "sar_embedding":          np.ndarray (B, embedding_dim)
                "fused_embedding":        np.ndarray (B, embedding_dim)  ← primary output
                "cross_modal_similarity": np.ndarray (B,)
                "confidence":             np.ndarray (B,)
            }

        The `fused_embedding` is the primary output for the downstream reasoning model.
        """
        # ---- Pre-process raw arrays if requested ----
        if preprocess:
            if isinstance(optical, np.ndarray) and optical.ndim == 3:
                optical = self._s2_pre(optical)   # → (13, H, W) tensor
            if isinstance(sar, np.ndarray) and sar.ndim == 3:
                sar = self._s1_pre(sar)           # → (2, H, W) tensor

        # ---- Ensure 4-D tensors ----
        optical_t = _ensure_4d(optical, self.device)  # (B, 13, H, W)
        sar_t     = _ensure_4d(sar,     self.device)  # (B, 2,  H, W)

        # ---- Validate shapes ----
        B = optical_t.shape[0]
        if sar_t.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: optical has {B}, sar has {sar_t.shape[0]}"
            )
        if optical_t.shape[1] != self.config.optical_channels:
            raise ValueError(
                f"Expected {self.config.optical_channels} optical channels, "
                f"got {optical_t.shape[1]}"
            )
        if sar_t.shape[1] != self.config.sar_channels:
            raise ValueError(
                f"Expected {self.config.sar_channels} SAR channels, "
                f"got {sar_t.shape[1]}"
            )

        # ---- Forward pass ----
        output = self.model(optical_t, sar_t)

        # ---- Cosine similarity between optical and SAR embeddings ----
        cross_sim = (output.optical_embedding * output.sar_embedding).sum(dim=-1)  # (B,)

        # ---- Confidence: map cosine similarity from [-1, 1] to [0, 1] ----
        confidence = (cross_sim + 1.0) / 2.0  # (B,)

        # ---- Move to CPU + numpy ----
        def to_np(t: torch.Tensor) -> np.ndarray:
            return t.cpu().float().numpy()

        return {
            "optical_embedding":      to_np(output.optical_embedding),
            "sar_embedding":          to_np(output.sar_embedding),
            "fused_embedding":        to_np(output.fused_embedding),     # primary output
            "cross_modal_similarity": to_np(cross_sim),
            "confidence":             to_np(confidence),
        }

    @torch.no_grad()
    def get_structured_output(
        self,
        optical: Union[torch.Tensor, np.ndarray],
        sar:     Union[torch.Tensor, np.ndarray],
        preprocess: bool = False,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        JSON-serialisable metadata output.

        The fused_embedding is NOT included by default (it's a large float array);
        pass include_embeddings=True to get it as a list.

        Returns:
            {
                "model": {...}                     — model configuration
                "cross_modal_similarity": [float]  — per-sample cosine sim
                "confidence": [float]              — per-sample [0,1]
                "optical_embedding": [[...]]       — only if include_embeddings=True
                "sar_embedding": [[...]]           — only if include_embeddings=True
                "fused_embedding": [[...]]         — only if include_embeddings=True
            }

        NOTE: This module does NOT generate natural-language reasoning.
              That is the responsibility of the Common Reasoning Model.
        """
        result = self.encode_pair(optical, sar, preprocess=preprocess)

        output: Dict[str, Any] = {
            "model": {
                "embedding_dim":  self.config.embedding_dim,
                "encoder_type":   self.config.encoder_type,
                "fusion":         "cross_attention" if self.config.use_cross_attention_fusion
                                  else "concat_mlp",
            },
            "cross_modal_similarity": result["cross_modal_similarity"].tolist(),
            "confidence":             result["confidence"].tolist(),
        }

        if include_embeddings:
            output["optical_embedding"] = result["optical_embedding"].tolist()
            output["sar_embedding"]     = result["sar_embedding"].tolist()
            output["fused_embedding"]   = result["fused_embedding"].tolist()

        return output

    # ---- Convenience: encode single modalities ----

    @torch.no_grad()
    def encode_optical_only(
        self,
        optical: Union[torch.Tensor, np.ndarray],
        preprocess: bool = False,
    ) -> np.ndarray:
        """Encode optical image only.  Returns (B, embedding_dim) numpy array."""
        if preprocess and isinstance(optical, np.ndarray) and optical.ndim == 3:
            optical = self._s2_pre(optical)
        optical_t = _ensure_4d(optical, self.device)
        return self.model.encode_optical(optical_t).cpu().numpy()

    @torch.no_grad()
    def encode_sar_only(
        self,
        sar: Union[torch.Tensor, np.ndarray],
        preprocess: bool = False,
    ) -> np.ndarray:
        """Encode SAR image only.  Returns (B, embedding_dim) numpy array."""
        if preprocess and isinstance(sar, np.ndarray) and sar.ndim == 3:
            sar = self._s1_pre(sar)
        sar_t = _ensure_4d(sar, self.device)
        return self.model.encode_sar(sar_t).cpu().numpy()
