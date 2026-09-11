"""
losses.py — InfoNCE contrastive loss and auxiliary consistency loss for Optical-SAR fusion.

Update: adds cloud-coverage-aware weighting to handle the information asymmetry
between optical (degraded by clouds) and SAR (cloud-penetrating) modalities.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) loss between Optical and SAR projected embeddings.

    Supports optional per-sample confidence weighting of the optical->SAR direction,
    so that heavily cloud-covered optical patches contribute less to the alignment
    objective (their optical embedding is uninformative).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        optical_proj: torch.Tensor,                       # (B, projection_dim) L2-normalised
        sar_proj:     torch.Tensor,                       # (B, projection_dim) L2-normalised
        optical_confidence: Optional[torch.Tensor] = None # (B,) in [0, 1], 1 = clear
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = optical_proj.shape[0]
        device = optical_proj.device

        # Cosine similarity matrix scaled by temperature
        # (B, B) where [i, j] is sim(opt_i, sar_j)
        logits_opt2sar = torch.matmul(optical_proj, sar_proj.T) / self.temperature
        logits_sar2opt = logits_opt2sar.T

        labels = torch.arange(batch_size, device=device)

        # Per-sample cross-entropy for the optical->SAR direction
        loss_opt2sar_per_sample = F.cross_entropy(
            logits_opt2sar, labels, reduction="none"
        )  # (B,)

        # SAR->optical is left unweighted: SAR is informative regardless of clouds
        loss_sar2opt = F.cross_entropy(logits_sar2opt, labels)

        if optical_confidence is not None:
            # Clamp for safety; ensure shape (B,)
            w = optical_confidence.to(device=device, dtype=logits_opt2sar.dtype).clamp(0.0, 1.0)
            # Normalise so the effective batch size (sum of weights) is preserved.
            # Avoid div-by-zero when the whole batch is fully clouded.
            w_sum = w.sum().clamp_min(1e-6)
            loss_opt2sar = (loss_opt2sar_per_sample * w).sum() / w_sum
        else:
            loss_opt2sar = loss_opt2sar_per_sample.mean()

        infonce_loss = 0.5 * (loss_opt2sar + loss_sar2opt)

        with torch.no_grad():
            acc_opt_to_sar = (logits_opt2sar.argmax(dim=-1) == labels).float().mean().item()
            acc_sar_to_opt = (logits_sar2opt.argmax(dim=-1) == labels).float().mean().item()
            mean_conf = (
                optical_confidence.mean().item()
                if optical_confidence is not None else 1.0
            )

        metrics = {
            "loss_infonce": infonce_loss.item(),
            "loss_opt2sar": loss_opt2sar.item(),
            "loss_sar2opt": loss_sar2opt.item(),
            "acc_opt_to_sar": acc_opt_to_sar,
            "acc_sar_to_opt": acc_sar_to_opt,
            "mean_optical_confidence": mean_conf,
        }

        return infonce_loss, metrics


class AuxiliaryConsistencyLoss(nn.Module):
    """
    Consistency loss preventing fusion collapse by encouraging the fused embedding
    to remain close in cosine space to both optical and SAR modality embeddings.

    Also supports confidence weighting: for cloudy samples, the optical term is
    down-weighted so the fused embedding is not pulled toward an uninformative
    optical representation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        fused_emb:   torch.Tensor,                        # (B, embedding_dim) L2-normalised
        optical_emb: torch.Tensor,                        # (B, embedding_dim) L2-normalised
        sar_emb:     torch.Tensor,                        # (B, embedding_dim) L2-normalised
        optical_confidence: Optional[torch.Tensor] = None # (B,) in [0, 1]
    ) -> torch.Tensor:
        cos_opt = F.cosine_similarity(fused_emb, optical_emb, dim=-1)  # (B,)
        cos_sar = F.cosine_similarity(fused_emb, sar_emb, dim=-1)      # (B,)

        if optical_confidence is not None:
            w = optical_confidence.to(device=cos_opt.device, dtype=cos_opt.dtype).clamp(0.0, 1.0)
            w_sum = w.sum().clamp_min(1e-6)
            cos_opt_w = (cos_opt * w).sum() / w_sum
        else:
            cos_opt_w = cos_opt.mean()

        cos_sar_w = cos_sar.mean()
        # Loss is 1 - average weighted cosine similarity
        return 1.0 - 0.5 * (cos_opt_w + cos_sar_w)


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
        optical_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_infonce, metrics = self.infonce(
            optical_proj, sar_proj, optical_confidence=optical_confidence
        )

        if self.use_aux_loss and self.aux_weight > 0:
            loss_aux = self.aux_loss(
                fused_emb, optical_emb, sar_emb,
                optical_confidence=optical_confidence,
            )
            total_loss = loss_infonce + self.aux_weight * loss_aux
            metrics["loss_aux"] = loss_aux.item()
        else:
            total_loss = loss_infonce
            metrics["loss_aux"] = 0.0

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
