"""
models/qwen_reasoner.py — Optical + SAR to Qwen2.5 Multimodal Reasoning Model with LoRA / PEFT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from models.optical_encoder import OpticalEncoder
from models.sar_encoder import SAREncoder
from models.spatial_extractor import SpatialFeatureExtractor
from models.visual_projector import VisualProjector

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are the Optical + SAR Remote Sensing Reasoning Specialist for the SatQuery AI system. "
    "You receive aligned Sentinel-2 multispectral optical imagery and Sentinel-1 synthetic aperture radar (SAR) data. "
    "Analyze the visual feature tokens provided, reason over terrain, land cover, water bodies, built-up areas, "
    "vegetation, and cloud penetration, and provide a clear, evidence-based natural language answer to the user query."
)

VISION_START_TAG = "<|vision_start|>"
VISION_PAD_TAG = "<|vision_pad|>"
VISION_END_TAG = "<|vision_end|>"


class OpticalSARQwenReasoner(nn.Module):
    """
    Multimodal Remote Sensing Reasoning Model combining:
    - ResNet-18 Optical & SAR Encoders (frozen)
    - Spatial Feature Extractor & Cross-Attention Token Fusion
    - Trainable Visual Projector (MLP: 512 -> 2048)
    - Qwen2.5-3B-Instruct with PEFT / LoRA adapters
    """

    def __init__(
        self,
        optical_encoder: OpticalEncoder,
        sar_encoder: SAREncoder,
        llm_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        llm_hidden_dim: int = 2048,
        visual_dim: int = 512,
        grid_size: int = 8,  # 8x8 = 64 visual tokens
        freeze_vision_backbones: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        self.llm_hidden_dim = llm_hidden_dim
        self.visual_dim = visual_dim
        self.grid_size = grid_size
        self.num_visual_tokens = grid_size * grid_size

        # 1. Vision Backbones & Spatial Extractor
        self.spatial_extractor = SpatialFeatureExtractor(
            optical_encoder=optical_encoder,
            sar_encoder=sar_encoder,
            target_grid_size=grid_size,
            feature_dim=visual_dim,
        )

        # 2. Trainable Visual Projector
        self.visual_projector = VisualProjector(
            visual_dim=visual_dim,
            llm_hidden_dim=llm_hidden_dim,
        )

        # 3. Freeze vision backbones initially if requested
        if freeze_vision_backbones:
            self.freeze_vision_encoders()

        # 4. LLM & Tokenizer
        self.llm = llm_model
        self.tokenizer = tokenizer

        # Apply LoRA if requested and LLM is provided
        if self.llm is not None and use_lora and PEFT_AVAILABLE:
            self._apply_lora(r=lora_r, alpha=lora_alpha)

    def freeze_vision_encoders(self) -> None:
        """Freeze optical and SAR vision backbones."""
        for param in self.spatial_extractor.optical_encoder.parameters():
            param.requires_grad = False
        for param in self.spatial_extractor.sar_encoder.parameters():
            param.requires_grad = False
        logger.info("Vision backbones (ResNet-18 Optical & SAR) frozen.")

    def unfreeze_vision_encoders(self) -> None:
        """Unfreeze vision backbones for end-to-end fine-tuning."""
        for param in self.spatial_extractor.optical_encoder.parameters():
            param.requires_grad = True
        for param in self.spatial_extractor.sar_encoder.parameters():
            param.requires_grad = True
        logger.info("Vision backbones unfrozen.")

    def _apply_lora(self, r: int = 16, alpha: int = 32) -> None:
        """Attach PEFT LoRA adapters to Qwen2.5 attention projections."""
        if not isinstance(self.llm, PeftModel):
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            logger.info("Attached LoRA adapters to Qwen2.5 attention layers.")

    def extract_and_project_tokens(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract fused spatial visual tokens and project into LLM embedding space.
        Returns:
            projected_tokens: (B, N, llm_hidden_dim) e.g. (B, 64, 2048)
            fused_tokens:     (B, N, 512)
            cross_sim:        (B,)
        """
        fused_tokens, opt_tokens, sar_tokens = self.spatial_extractor(optical, sar)
        projected_tokens = self.visual_projector(fused_tokens)  # (B, N, 2048)

        # Cross-modal alignment metric across tokens
        cross_sim = (opt_tokens * sar_tokens).sum(dim=-1).mean(dim=-1)  # (B,)

        return projected_tokens, fused_tokens, cross_sim

    def prepare_multimodal_embeddings(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        user_queries: List[str],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds `inputs_embeds` by inserting projected visual tokens into the token sequence.
        """
        device = optical.device
        B = optical.shape[0]

        # 1. Project visual tokens
        projected_visual, _, _ = self.extract_and_project_tokens(optical, sar)  # (B, 64, 2048)

        if self.tokenizer is None or self.llm is None:
            # Fallback for offline token representation
            dummy_embeds = projected_visual
            dummy_mask = torch.ones((B, self.num_visual_tokens), device=device, dtype=torch.long)
            return dummy_embeds, dummy_mask

        # 2. Build text sequences with visual placeholders
        pad_sequence = VISION_PAD_TAG * self.num_visual_tokens
        prompt_texts = []
        for q in user_queries:
            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_START_TAG}{pad_sequence}{VISION_END_TAG}\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompt_texts.append(text)

        # 3. Tokenize
        encoded = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

        # 4. Get text embeddings from LLM word embedding layer
        embed_tokens_fn = (
            self.llm.get_input_embeddings()
            if hasattr(self.llm, "get_input_embeddings")
            else self.llm.base_model.model.embed_tokens
        )
        text_embeds = embed_tokens_fn(input_ids)  # (B, SeqLen, 2048)

        # 5. Locate pad token IDs and splice in projected visual tokens
        pad_token_id = self.tokenizer.convert_tokens_to_ids(VISION_PAD_TAG)
        if pad_token_id is None or pad_token_id == self.tokenizer.unk_token_id:
            # Fallback: find vision_start and splice directly
            start_token_id = self.tokenizer.convert_tokens_to_ids(VISION_START_TAG)

        # In-place splice visual tokens into text embedding tensor
        for b in range(B):
            pad_indices = (input_ids[b] == pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_indices) == self.num_visual_tokens:
                text_embeds[b, pad_indices, :] = projected_visual[b]
            elif len(pad_indices) > 0:
                k = min(len(pad_indices), self.num_visual_tokens)
                text_embeds[b, pad_indices[:k], :] = projected_visual[b, :k]

        return text_embeds, attention_mask

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        user_queries: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass for training. Computes language modeling loss over multimodal inputs.
        """
        inputs_embeds, attention_mask = self.prepare_multimodal_embeddings(optical, sar, user_queries)
        if self.llm is None:
            return {"loss": torch.tensor(0.0, requires_grad=True, device=optical.device)}

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs
