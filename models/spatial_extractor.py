"""
models/spatial_extractor.py — Spatial feature token extractor & token fusion.

Extracts spatial feature maps before Global Average Pooling from the ResNet-18
Optical & SAR backbones, pools to standard grid (e.g. 8x8 = 64 spatial tokens),
and fuses optical and SAR tokens to preserve localized land cover features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optical_encoder import OpticalEncoder
from models.sar_encoder import SAREncoder


class SpatialFeatureExtractor(nn.Module):
    """
    Extracts multi-token spatial feature representations from ResNet backbones
    instead of collapsing to a single 1D vector.
    """

    def __init__(
        self,
        optical_encoder: OpticalEncoder,
        sar_encoder: SAREncoder,
        target_grid_size: int = 8,  # 8x8 = 64 spatial tokens
        feature_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.optical_encoder = optical_encoder
        self.sar_encoder = sar_encoder
        self.target_grid_size = target_grid_size
        self.feature_dim = feature_dim
        self.num_tokens = target_grid_size * target_grid_size

        # Adaptive spatial pooling to standardize token count regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_grid_size, target_grid_size))

        # Spatial Cross-Attention Token Fusion
        self.opt_to_sar_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.sar_to_opt_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_opt = nn.LayerNorm(feature_dim)
        self.norm_sar = nn.LayerNorm(feature_dim)

        # Token-wise fusion MLP
        self.token_fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def _extract_backbone_spatial_map(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet conv layers up to layer4 (before avgpool & fc).
        Returns: (B, 512, H', W')
        """
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)  # (B, 512, H/32, W/32)
        return x

    def forward(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            optical: (B, 13, H, W)
            sar:     (B, 2,  H, W)
        Returns:
            fused_tokens:   (B, N, 512) where N = target_grid_size^2 (e.g. 64)
            optical_tokens: (B, N, 512)
            sar_tokens:     (B, N, 512)
        """
        # 1. Extract raw 2D spatial feature maps
        opt_feat_map = self._extract_backbone_spatial_map(self.optical_encoder.backbone, optical)  # (B, 512, H', W')
        sar_feat_map = self._extract_backbone_spatial_map(self.sar_encoder.backbone, sar)          # (B, 512, H', W')

        # 2. Pool to standard spatial grid (e.g. 8x8)
        opt_grid = self.adaptive_pool(opt_feat_map)  # (B, 512, 8, 8)
        sar_grid = self.adaptive_pool(sar_feat_map)  # (B, 512, 8, 8)

        B, C, H_g, W_g = opt_grid.shape
        N = H_g * W_g

        # 3. Flatten spatial dimensions into token sequence (B, N, 512)
        opt_tokens = opt_grid.view(B, C, N).permute(0, 2, 1)  # (B, N, 512)
        sar_tokens = sar_grid.view(B, C, N).permute(0, 2, 1)  # (B, N, 512)

        # 4. Spatial Multi-Head Cross-Attention Token Fusion
        # Optical tokens query SAR tokens (cloud-penetrating radar details)
        opt_attended, _ = self.opt_to_sar_cross_attn(
            query=opt_tokens, key=sar_tokens, value=sar_tokens
        )
        opt_refined = self.norm_opt(opt_tokens + opt_attended)

        # SAR tokens query Optical tokens (multispectral spectral reflectance details)
        sar_attended, _ = self.sar_to_opt_cross_attn(
            query=sar_tokens, key=opt_tokens, value=opt_tokens
        )
        sar_refined = self.norm_sar(sar_tokens + sar_attended)

        # 5. Concatenate and pass through token-wise fusion MLP
        fused_tokens = torch.cat([opt_refined, sar_refined], dim=-1)  # (B, N, 1024)
        fused_tokens = self.token_fusion_mlp(fused_tokens)            # (B, N, 512)

        return fused_tokens, opt_refined, sar_refined
