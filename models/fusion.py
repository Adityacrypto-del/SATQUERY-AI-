"""
fusion.py — Projection heads and fusion modules for Optical + SAR embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    2-layer MLP for mapping encoder outputs to contrastive loss space.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If batch size is 1 during evaluation/single inference, avoid BatchNorm1d crash
        if x.shape[0] == 1:
            # Bypass batchnorm eval or use training mode check
            self.eval()
        return self.net(x)


class ConcatMLPFusion(nn.Module):
    """
    Default fusion strategy: Concatenates optical and SAR features and projects via 2-layer MLP.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opt_feat: (B, embedding_dim)
            sar_feat: (B, embedding_dim)
        Returns:
            (B, embedding_dim) fused representation
        """
        concat = torch.cat([opt_feat, sar_feat], dim=-1)
        if concat.shape[0] == 1:
            self.eval()
        return self.net(concat)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion:
    Optical attends to SAR, SAR attends to Optical, followed by MLP fusion.
    """

    def __init__(self, embedding_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.opt_to_sar_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.sar_to_opt_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_opt = nn.LayerNorm(embedding_dim)
        self.norm_sar = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opt_feat: (B, embedding_dim)
            sar_feat: (B, embedding_dim)
        Returns:
            (B, embedding_dim) fused representation
        """
        opt_seq = opt_feat.unsqueeze(1)  # (B, 1, D)
        sar_seq = sar_feat.unsqueeze(1)  # (B, 1, D)

        # Optical queries SAR
        opt_attended, _ = self.opt_to_sar_attn(query=opt_seq, key=sar_seq, value=sar_seq)
        opt_out = self.norm_opt(opt_seq + opt_attended).squeeze(1)

        # SAR queries Optical
        sar_attended, _ = self.sar_to_opt_attn(query=sar_seq, key=opt_seq, value=opt_seq)
        sar_out = self.norm_sar(sar_seq + sar_attended).squeeze(1)

        fused = torch.cat([opt_out, sar_out], dim=-1)
        return self.mlp(fused)
