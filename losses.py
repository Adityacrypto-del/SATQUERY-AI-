"""
losses.py — InfoNCE contrastive loss and auxiliary consistency loss for Optical-SAR fusion.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) loss between Optical and SAR projected embeddings.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        optical_proj: torch.Tensor,   # (B, projection_dim) L2-normalised
        sar_proj:     torch.Tensor,   # (B, projection_dim) L2-normalised
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = optical_proj.shape[0]
        device = optical_proj.device

        # Cosine similarity matrix scaled by temperature
        # (B, B) where [i, j] is sim(opt_i, sar_j)
        logits_opt2sar = torch.matmul(optical_proj, sar_proj.T) / self.temperature
        logits_sar2opt = logits_opt2sar.T

        labels = torch.arange(batch_size, device=device)

        loss_opt2sar = F.cross_entropy(logits_opt2sar, labels)
        loss_sar2opt = F.cross_entropy(logits_sar2opt, labels)

        infonce_loss = 0.5 * (loss_opt2sar + loss_sar2opt)

        with torch.no_grad():
            acc_opt_to_sar = (logits_opt2sar.argmax(dim=-1) == labels).float().mean().item()
            acc_sar_to_opt = (logits_sar2opt.argmax(dim=-1) == labels).float().mean().item()

        metrics = {
            "loss_infonce": infonce_loss.item(),
            "acc_opt_to_sar": acc_opt_to_sar,
            "acc_sar_to_opt": acc_sar_to_opt,
        }

        return infonce_loss, metrics


class AuxiliaryConsistencyLoss(nn.Module):
    """
    Consistency loss preventing fusion collapse by encouraging the fused embedding
    to remain close in cosine space to both optical and SAR modality embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        fused_emb:   torch.Tensor,  # (B, embedding_dim) L2-normalised
        optical_emb: torch.Tensor,  # (B, embedding_dim) L2-normalised
        sar_emb:     torch.Tensor,  # (B, embedding_dim) L2-normalised
    ) -> torch.Tensor:
        cos_opt = F.cosine_similarity(fused_emb, optical_emb, dim=-1).mean()
        cos_sar = F.cosine_similarity(fused_emb, sar_emb, dim=-1).mean()
        # Loss is 1 - average cosine similarity
        return 1.0 - 0.5 * (cos_opt + cos_sar)


class CombinedLoss(nn.Module):
    """
    Combines InfoNCE contrastive loss with auxiliary consistency loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        aux_weight: float = 0.1,
        use_aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)
        self.aux_loss = AuxiliaryConsistencyLoss()
        self.aux_weight = aux_weight
        self.use_aux_loss = use_aux_loss

    def forward(
        self,
        optical_proj: torch.Tensor,
        sar_proj:     torch.Tensor,
        fused_emb:    torch.Tensor,
        optical_emb:  torch.Tensor,
        sar_emb:      torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_infonce, metrics = self.infonce(optical_proj, sar_proj)

        if self.use_aux_loss and self.aux_weight > 0:
            loss_aux = self.aux_loss(fused_emb, optical_emb, sar_emb)
            total_loss = loss_infonce + self.aux_weight * loss_aux
            metrics["loss_aux"] = loss_aux.item()
        else:
            total_loss = loss_infonce
            metrics["loss_aux"] = 0.0

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
