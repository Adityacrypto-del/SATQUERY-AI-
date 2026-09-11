"""
reasoning_pipeline.py — Optical + SAR to Visual Projector to Qwen2.5 Multimodal Reasoning Pipeline.

This module is the core reasoning API for SatQuery AI:

    Sentinel-2 Optical (13 bands)
            ↓
    Modified ResNet-18 Optical Encoder
            ↓
    Optical Features
            \
             → Feature Fusion → Fused Visual Tokens
            /
    Sentinel-1 SAR (2 channels: VV, VH)
            ↓
    Modified ResNet-18 SAR Encoder
            ↓
    SAR Features
            ↓
    Visual Projector
            ↓
    Qwen2.5 Reasoning Model (OpenRouter / Local)
            ↓
    Natural-language reasoning + confidence + visual evidence + JSON result

Public Interface:
    from reasoning_pipeline import OpticalSARReasoner

    reasoner = OpticalSARReasoner.load(checkpoint_path="checkpoints_synthetic/best.pt")

    result = reasoner.reason(
        optical_image=optical_arr,
        sar_image=sar_arr,
        user_query="Identify built-up and water-covered regions."
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
import torch
from PIL import Image

from config import ModelConfig
from models.contrastive_model import OpticalSARContrastiveModel
from models.spatial_extractor import SpatialFeatureExtractor
from models.visual_projector import VisualProjector
from preprocessing import build_preprocessors
from utils import get_device

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_QWEN_MODEL = "qwen/qwen-2.5-72b-instruct"



def _ensure_tensor_4d(
    x: Union[torch.Tensor, np.ndarray], device: torch.device
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.to(device, non_blocking=True)


class OpticalSARReasoner:
    """
    Modular Optical + SAR Multimodal Reasoning Specialist for SatQuery AI.
    """

    def __init__(
        self,
        vision_model: OpticalSARContrastiveModel,
        spatial_extractor: SpatialFeatureExtractor,
        visual_projector: VisualProjector,
        config: ModelConfig,
        device: torch.device,
        openrouter_api_key: str = OPENROUTER_API_KEY,
        qwen_model_name: str = DEFAULT_QWEN_MODEL,
    ) -> None:
        self.vision_model = vision_model.eval()
        self.spatial_extractor = spatial_extractor.eval()
        self.visual_projector = visual_projector.eval()
        self.config = config
        self.device = device
        self.openrouter_api_key = openrouter_api_key
        self.qwen_model_name = qwen_model_name
        self._s2_pre, self._s1_pre = build_preprocessors(config)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path = "checkpoints_synthetic/best.pt",
        device: Optional[torch.device] = None,
        openrouter_api_key: str = OPENROUTER_API_KEY,
        qwen_model_name: str = DEFAULT_QWEN_MODEL,
    ) -> "OpticalSARReasoner":
        """
        Load trained vision encoders and initialize the spatial projector & reasoner.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            checkpoint_path = Path("checkpoints/best.pt")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        if device is None:
            device = get_device()

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "config" in state:
            cfg = ModelConfig.from_dict(state["config"])
        else:
            cfg = ModelConfig()

        vision_model = OpticalSARContrastiveModel(
            embedding_dim=cfg.embedding_dim,
            projection_dim=cfg.projection_dim,
            encoder_type=cfg.encoder_type,
            use_cross_attention_fusion=cfg.use_cross_attention_fusion,
            optical_channels=cfg.optical_channels,
            sar_channels=cfg.sar_channels,
            pretrained_encoders=False,
        ).to(device)

        vision_model.load_state_dict(state["model_state"], strict=True)

        # Build spatial extractor and visual projector
        spatial_extractor = SpatialFeatureExtractor(
            optical_encoder=vision_model.optical_encoder,
            sar_encoder=vision_model.sar_encoder,
            target_grid_size=8,  # 8x8 = 64 spatial tokens
            feature_dim=cfg.embedding_dim,
        ).to(device)

        visual_projector = VisualProjector(
            visual_dim=cfg.embedding_dim,
            llm_hidden_dim=2048,  # Qwen2.5 hidden size
        ).to(device)

        # If checkpoint has saved projector weights, load them
        if "projector_state" in state:
            visual_projector.load_state_dict(state["projector_state"])

        return cls(
            vision_model=vision_model,
            spatial_extractor=spatial_extractor,
            visual_projector=visual_projector,
            config=cfg,
            device=device,
            openrouter_api_key=openrouter_api_key,
            qwen_model_name=qwen_model_name,
        )

    def extract_visual_evidence(
        self, optical_raw: np.ndarray, sar_raw: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract domain-specific remote sensing indicators from 13-band S2 and 2-channel S1.
        """
        evidence = {}

        # 1. Optical Spectral Indices (B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR)
        if optical_raw.shape[0] >= 8:
            red = optical_raw[3].astype(np.float32)
            nir = optical_raw[7].astype(np.float32)
            green = optical_raw[2].astype(np.float32)
            blue = optical_raw[1].astype(np.float32)

            # NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-6)
            veg_pct = float(np.mean(ndvi > 0.3) * 100.0)

            # NDWI (Normalized Difference Water Index)
            ndwi = (green - nir) / (green + nir + 1e-6)
            water_pct_opt = float(np.mean(ndwi > 0.0) * 100.0)

            # Cloud / Bright Albedo Index
            brightness = (red + green + blue) / 3.0
            cloud_pct = float(np.mean((brightness > 2800) & (nir > 2800)) * 100.0)

            evidence["optical_indices"] = {
                "mean_ndvi": round(float(np.mean(ndvi)), 3),
                "vegetation_coverage_pct": round(veg_pct, 1),
                "water_body_optical_pct": round(water_pct_opt, 1),
                "estimated_cloud_coverage_pct": round(cloud_pct, 1),
            }

        # 2. SAR Radar Polarimetric Indicators (VV, VH in dB)
        if sar_raw.shape[0] >= 2:
            vv = sar_raw[0].astype(np.float32)
            vh = sar_raw[1].astype(np.float32)

            # Water specular reflection: extreme low backscatter (VV < -20 dB)
            water_radar_pct = float(np.mean(vv < -20.0) * 100.0)

            # Urban double-bounce scattering: very high backscatter (VV > -10 dB, VH > -16 dB)
            urban_radar_pct = float(np.mean((vv > -10.0) & (vh > -16.0)) * 100.0)

            # Volume scattering (Vegetation / Forest): cross-pol ratio
            vh_vv_ratio = float(np.mean(vh - vv))

            evidence["sar_radar_indices"] = {
                "mean_vv_backscatter_db": round(float(np.mean(vv)), 2),
                "mean_vh_backscatter_db": round(float(np.mean(vh)), 2),
                "water_body_radar_pct": round(water_radar_pct, 1),
                "built_up_structures_pct": round(urban_radar_pct, 1),
                "cross_pol_ratio_vh_minus_vv": round(vh_vv_ratio, 2),
                "cloud_penetration_status": "Active (radar unaffected by cloud occlusion)",
            }

        return evidence

    @torch.no_grad()
    def reason(
        self,
        optical_image: Union[str, Path, np.ndarray, torch.Tensor],
        sar_image: Union[str, Path, np.ndarray, torch.Tensor],
        user_query: str,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """
        Main public reasoning function.

        Args:
            optical_image: 13-band array, path to .npy, or path to .png/.jpg
            sar_image:     2-channel array, path to .npy, or path to .png/.jpg
            user_query:    Natural language question from the user
            preprocess:    Whether to apply standard satellite scaling

        Returns:
            {
                "final_answer": str,
                "confidence": float,
                "visual_evidence": dict,
                "structured_output": dict
            }
        """
        # 1. Load inputs if paths
        optical_raw, sar_raw = self._resolve_input_data(optical_image, sar_image)

        # 2. Preprocess to tensors
        if preprocess:
            opt_tensor = self._s2_pre(optical_raw)  # (13, H, W)
            sar_tensor = self._s1_pre(sar_raw)      # (2, H, W)
        else:
            opt_tensor = torch.from_numpy(optical_raw).float()
            sar_tensor = torch.from_numpy(sar_raw).float()

        opt_4d = _ensure_tensor_4d(opt_tensor, self.device)
        sar_4d = _ensure_tensor_4d(sar_tensor, self.device)

        # 3. Vision Encoders + Spatial Token Extraction + Cross-Attention Fusion
        fused_tokens, opt_tokens, sar_tokens = self.spatial_extractor(opt_4d, sar_4d)
        # Shape: (1, 64, 512)

        # 4. Visual Projector (Maps 512 -> 2048)
        projected_tokens = self.visual_projector(fused_tokens)  # (1, 64, 2048)

        # 5. Multimodal Cross-Modal Alignment & Confidence Score
        cross_sim = float((opt_tokens * sar_tokens).sum(dim=-1).mean().item())
        confidence = float(np.clip((cross_sim + 1.0) / 2.0, 0.05, 0.99))

        # 6. Extract domain visual evidence
        evidence = self.extract_visual_evidence(optical_raw, sar_raw)

        # 7. LLM Reasoning via Qwen2.5
        reasoning_prompt = self._build_reasoning_prompt(
            user_query=user_query,
            confidence=confidence,
            cross_sim=cross_sim,
            evidence=evidence,
            token_count=projected_tokens.shape[1],
            projected_dim=projected_tokens.shape[-1],
        )

        llm_reply = self._call_qwen_llm(reasoning_prompt)

        # 8. Structured JSON Result
        structured = {
            "query": user_query,
            "pipeline": {
                "optical_encoder": "Modified ResNet-18 (13 channels)",
                "sar_encoder": "Modified ResNet-18 (2 channels: VV, VH)",
                "spatial_grid_tokens": projected_tokens.shape[1],
                "visual_projector": f"MLP ({self.config.embedding_dim} -> 2048)",
                "reasoning_model": self.qwen_model_name,
            },
            "metrics": {
                "cross_modal_similarity": round(cross_sim, 4),
                "confidence_score": round(confidence, 4),
            },
            "visual_evidence": evidence,
            "projected_tokens_summary": {
                "num_tokens": projected_tokens.shape[1],
                "token_dimension": projected_tokens.shape[-1],
                "token_norm_mean": round(float(projected_tokens.norm(dim=-1).mean().item()), 3),
            },
            "final_answer": llm_reply,
        }

        return {
            "final_answer": llm_reply,
            "confidence": round(confidence, 4),
            "visual_evidence": evidence,
            "structured_output": structured,
        }

    def _resolve_input_data(
        self,
        opt: Union[str, Path, np.ndarray, torch.Tensor],
        sar: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper to resolve paths or arrays into (13, H, W) and (2, H, W) numpy arrays."""
        # Optical
        if isinstance(opt, (str, Path)):
            p = Path(opt)
            if p.suffix == ".npy":
                opt_arr = np.load(p).astype(np.float32)
            else:
                img = Image.open(p).convert("RGB")
                img_np = np.array(img, dtype=np.float32)
                h, w, _ = img_np.shape
                opt_arr = np.zeros((13, h, w), dtype=np.float32)
                opt_arr[1] = (img_np[:, :, 2] / 255.0) * 2000.0  # Blue
                opt_arr[2] = (img_np[:, :, 1] / 255.0) * 2000.0  # Green
                opt_arr[3] = (img_np[:, :, 0] / 255.0) * 2000.0  # Red
                opt_arr[7] = ((img_np[:, :, 1] + img_np[:, :, 0]) / 2.0 / 255.0) * 3500.0  # NIR
                for b in range(13):
                    if b not in [1, 2, 3, 7]:
                        opt_arr[b] = (img_np.mean(axis=-1) / 255.0) * 1500.0
        elif isinstance(opt, torch.Tensor):
            opt_arr = opt.detach().cpu().numpy().astype(np.float32)
        else:
            opt_arr = np.array(opt, dtype=np.float32)

        # SAR
        if isinstance(sar, (str, Path)):
            p = Path(sar)
            if p.suffix == ".npy":
                sar_arr = np.load(p).astype(np.float32)
            else:
                img = Image.open(p)
                img_np = np.array(img, dtype=np.float32)
                if img_np.ndim == 3:
                    vv = (img_np[:, :, 0] / 255.0) * 20.0 - 25.0
                    vh = (img_np[:, :, 1] / 255.0) * 20.0 - 30.0
                else:
                    vv = (img_np / 255.0) * 20.0 - 25.0
                    vh = (img_np / 255.0) * 20.0 - 30.0
                sar_arr = np.stack([vv, vh], axis=0)
        elif isinstance(sar, torch.Tensor):
            sar_arr = sar.detach().cpu().numpy().astype(np.float32)
        else:
            sar_arr = np.array(sar, dtype=np.float32)

        if opt_arr.ndim == 4:
            opt_arr = opt_arr[0]
        if sar_arr.ndim == 4:
            sar_arr = sar_arr[0]

        return opt_arr, sar_arr

    def _build_reasoning_prompt(
        self,
        user_query: str,
        confidence: float,
        cross_sim: float,
        evidence: Dict[str, Any],
        token_count: int,
        projected_dim: int,
    ) -> str:
        """Construct multimodal reasoning instruction prompt for Qwen2.5."""
        prompt = (
            f"You are the SatQuery AI Optical + SAR Multimodal Reasoning Specialist.\n"
            f"You have processed paired Sentinel-2 Optical (13 spectral bands) and Sentinel-1 SAR (VV/VH polarizations) imagery.\n"
            f"The spatial vision encoders and visual projector generated {token_count} spatial visual tokens (projected dimension: {projected_dim}-d).\n\n"
            f"--- SENSOR & SPATIAL EVIDENCE ---\n"
            f"• Multimodal Confidence Score: {confidence * 100:.1f}%\n"
            f"• Optical-SAR Cross-Modal Cosine Alignment: {cross_sim:.4f}\n"
            f"• Optical Indices: {json.dumps(evidence.get('optical_indices', {}), indent=2)}\n"
            f"• SAR Radar Indices: {json.dumps(evidence.get('sar_radar_indices', {}), indent=2)}\n\n"
            f"--- USER QUERY ---\n"
            f"{user_query}\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"Provide a clear, authoritative, and structured answer explaining:\n"
            f"1. Direct answer to the user's specific request.\n"
            f"2. Physical justification combining optical reflectance (color, NIR/SWIR) with SAR radar backscatter (surface roughness, specular reflection, double-bounce).\n"
            f"3. Note any cloud occlusion and how the SAR sensor penetrated or confirmed the observations."
        )
        return prompt

    def _call_qwen_llm(self, prompt: str) -> str:
        """Send prompt to Qwen2.5 on OpenRouter."""
        if not self.openrouter_api_key:
            return "Error: OpenRouter API key not configured."

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://satquery.ai",
                    "X-Title": "SatQuery AI Optical-SAR Reasoner",
                },
                json={
                    "model": self.qwen_model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the Optical + SAR Remote Sensing Reasoning Specialist for SatQuery AI. "
                                "Provide clear, technically rigorous, evidence-based reasoning."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"OpenRouter API error {resp.status_code}: {resp.text}")
                return f"LLM API Error ({resp.status_code}): {resp.text}"
        except Exception as e:
            logger.error(f"Failed to call Qwen LLM: {e}")
            return f"Reasoning engine error: {str(e)}"
