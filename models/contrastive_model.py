"""
contrastive_model.py — The top-level OpticalSARContrastiveModel.

Wires together:
  OpticalEncoder → ProjectionHead → (contrastive loss)
  SAREncoder     → ProjectionHead → (contrastive loss)
  optical_features + sar_features → Fusion → fused_embedding

Forward contract:
  optical_emb, sar_emb, fused_emb = model(optical, sar)

  optical_emb  : (B, embedding_dim) — L2-normalised, for retrieval
  sar_emb      : (B, embedding_dim) — L2-normalised, for retrieval
  fused_emb    : (B, embedding_dim) — for downstream reasoning
  optical_proj : (B, projection_dim) — projected, for InfoNCE loss
  sar_proj     : (B, projection_dim) — projected, for InfoNCE loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optical_encoder import OpticalEncoder
from .sar_encoder import SAREncoder
from .fusion import ProjectionHead, ConcatMLPFusion, CrossAttentionFusion


@dataclass
class ModelOutput:
    """Structured output from the model's forward pass."""
    optical_embedding: torch.Tensor   # (B, embedding_dim) L2-normalised
    sar_embedding:     torch.Tensor   # (B, embedding_dim) L2-normalised
    fused_embedding:   torch.Tensor   # (B, embedding_dim) — primary output
    optical_proj:      torch.Tensor   # (B, projection_dim) for loss
    sar_proj:          torch.Tensor   # (B, projection_dim) for loss


class OpticalSARContrastiveModel(nn.Module):
    """
    Full Optical+SAR joint representation model.

    Args:
        embedding_dim:              Output dimension for all embeddings (default 512).
        projection_dim:             Internal projection head output dim (default 256).
        encoder_type:               "resnet18" | "resnet34".
        use_cross_attention_fusion: Use CrossAttentionFusion instead of ConcatMLP.
        optical_channels:           Number of S2 bands (13).
        sar_channels:               Number of S1 channels (2).
        pretrained_encoders:        Load ImageNet weights for backbone (minus conv1).
        freeze_backbones:           Freeze backbone layers after init.
    """

    def __init__(
        self,
        embedding_dim:              int  = 512,
        projection_dim:             int  = 256,
        encoder_type:               str  = "resnet18",
        use_cross_attention_fusion: bool = False,
        optical_channels:           int  = 13,
        sar_channels:               int  = 2,
        pretrained_encoders:        bool = True,
        freeze_backbones:           bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim  = embedding_dim
        self.projection_dim = projection_dim

        # --- Encoders ---
        self.optical_encoder = OpticalEncoder(
            in_channels    = optical_channels,
            encoder_type   = encoder_type,
            pretrained     = pretrained_encoders,
            freeze_backbone= freeze_backbones,
            out_dim        = embedding_dim,
        )
        self.sar_encoder = SAREncoder(
            in_channels    = sar_channels,
            encoder_type   = encoder_type,
            pretrained     = pretrained_encoders,
            freeze_backbone= freeze_backbones,
            out_dim        = embedding_dim,
        )

        # --- Projection heads (for contrastive loss) ---
        self.optical_proj_head = ProjectionHead(
            in_dim=embedding_dim, hidden_dim=embedding_dim, out_dim=projection_dim
        )
        self.sar_proj_head = ProjectionHead(
            in_dim=embedding_dim, hidden_dim=embedding_dim, out_dim=projection_dim
        )

        # --- Fusion ---
        if use_cross_attention_fusion:
            self.fusion = CrossAttentionFusion(
                embedding_dim=embedding_dim,
                num_heads=max(1, embedding_dim // 64),
            )
        else:
            self.fusion = ConcatMLPFusion(
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim * 2,
            )

    def forward(
        self,
        optical: torch.Tensor,   # (B, 13, H, W)
        sar:     torch.Tensor,   # (B, 2, H, W)
    ) -> ModelOutput:
        """
        Returns a ModelOutput namedtuple-like dataclass.
        """
        # 1. Encode each modality
        opt_feat = self.optical_encoder(optical)   # (B, embedding_dim)
        sar_feat = self.sar_encoder(sar)           # (B, embedding_dim)

        # 2. L2-normalise the encoder outputs for retrieval / cosine similarity
        opt_emb = F.normalize(opt_feat, dim=-1)   # (B, embedding_dim)
        sar_emb = F.normalize(sar_feat, dim=-1)   # (B, embedding_dim)

        # 3. Project for contrastive loss (SimCLR style: project → loss, not project → downstream)
        opt_proj = self.optical_proj_head(opt_feat)   # (B, projection_dim)
        sar_proj = self.sar_proj_head(sar_feat)       # (B, projection_dim)
        opt_proj = F.normalize(opt_proj, dim=-1)
        sar_proj = F.normalize(sar_proj, dim=-1)

        # 4. Fuse encoder features (before normalisation, to preserve magnitude info)
        fused_emb = self.fusion(opt_feat, sar_feat)   # (B, embedding_dim)
        fused_emb = F.normalize(fused_emb, dim=-1)

        return ModelOutput(
            optical_embedding = opt_emb,
            sar_embedding     = sar_emb,
            fused_embedding   = fused_emb,
            optical_proj      = opt_proj,
            sar_proj          = sar_proj,
        )

    def encode_optical(self, optical: torch.Tensor) -> torch.Tensor:
        """Encode optical only. Returns (B, embedding_dim) normalised."""
        feat = self.optical_encoder(optical)
        return F.normalize(feat, dim=-1)

    def encode_sar(self, sar: torch.Tensor) -> torch.Tensor:
        """Encode SAR only. Returns (B, embedding_dim) normalised."""
        feat = self.sar_encoder(sar)
        return F.normalize(feat, dim=-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
