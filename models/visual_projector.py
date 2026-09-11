"""
models/visual_projector.py — Trainable Visual Projector mapping visual tokens to LLM embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """
    Trainable Multi-Layer Perceptron (MLP) Projector.
    
    Projects multimodal spatial feature tokens from the satellite vision space (e.g. 512-dim)
    into the Qwen2.5-3B-Instruct hidden embedding space (e.g. 2048-dim).
    """

    def __init__(
        self,
        visual_dim: int = 512,
        llm_hidden_dim: int = 2048,  # Default for Qwen2.5-3B
        intermediate_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.llm_hidden_dim = llm_hidden_dim

        self.projector = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, N, visual_dim) e.g. (B, 64, 512)
        Returns:
            projected_tokens: (B, N, llm_hidden_dim) e.g. (B, 64, 2048)
        """
        return self.projector(visual_tokens)
