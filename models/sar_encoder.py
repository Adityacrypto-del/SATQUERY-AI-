"""
sar_encoder.py — ResNet encoder for 2-channel Sentinel-1 SAR (VV, VH) imagery.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SAREncoder(nn.Module):
    """
    ResNet backbone adapted for Sentinel-1 SAR 2-channel input (VV, VH).
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_type: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder_type = encoder_type

        # Load backbone
        if encoder_type == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif encoder_type == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Patch conv1 for 2 channels (trained from scratch for SAR)
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        if new_conv1.bias is not None:
            nn.init.constant_(new_conv1.bias, 0.0)

        backbone.conv1 = new_conv1

        # Replace final fc layer
        in_features = backbone.fc.in_features
        if in_features != out_dim:
            backbone.fc = nn.Linear(in_features, out_dim)
        else:
            backbone.fc = nn.Identity()

        if freeze_backbone:
            for name, param in backbone.named_parameters():
                if not ("conv1" in name or "fc" in name):
                    param.requires_grad = False

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W)
        Returns:
            (B, out_dim) un-normalised feature vectors
        """
        return self.backbone(x)
