"""
train.py — Complete training loop for OpticalSARContrastiveModel.

Usage:
    python train.py
    python train.py --resume checkpoints/last.pt
    python train.py --max_samples 5000 --epochs 10   # quick experiment
    python train.py --streaming                       # stream dataset

Design notes:
- Mixed precision (torch.amp) enabled by default on CUDA.
- AdamW + cosine LR schedule with linear warmup.
- Best checkpoint selected by val_loss (lower = better).
- Last checkpoint always saved so training can resume.
- Metrics logged to checkpoints/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import ModelConfig
from dataset import build_dataloaders
from losses import CombinedLoss
from models import OpticalSARContrastiveModel
from evaluate import evaluate_retrieval
from utils import (
    setup_logging,
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    make_scaler,
    MetricsLogger,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_one_step(
    model:    OpticalSARContrastiveModel,
    batch:    Dict[str, torch.Tensor],
    loss_fn:  CombinedLoss,
    optimizer: AdamW,
    scaler:   Optional[torch.cuda.amp.GradScaler],
    cfg:      ModelConfig,
    device:   torch.device,
) -> Dict[str, float]:
    """
    Forward + backward + optimizer step.  Returns metrics dict.
    """
    optical = batch["optical"].to(device, non_blocking=True)  # (B, 13, H, W)
    sar     = batch["sar"].to(device,     non_blocking=True)  # (B, 2, H, W)

    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        output = model(optical, sar)
        loss, metrics = loss_fn(
            optical_proj=output.optical_proj,
            sar_proj=output.sar_proj,
            fused_emb=output.fused_embedding,
            optical_emb=output.optical_embedding,
            sar_emb=output.sar_embedding,
        )

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: ModelConfig, resume_path: Optional[str] = None) -> None:
    setup_logging()
    set_seed(cfg.seed)

    device = get_device()
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoints
    cfg.save(ckpt_dir / "config.json")

    # ---- Data ----
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        "Train batches: %d  |  Val batches: %d",
        len(train_loader), len(val_loader),
    )

    # ---- Model ----
    model = OpticalSARContrastiveModel(
        embedding_dim              = cfg.embedding_dim,
        projection_dim             = cfg.projection_dim,
        encoder_type               = cfg.encoder_type,
        use_cross_attention_fusion = cfg.use_cross_attention_fusion,
        optical_channels           = cfg.optical_channels,
        sar_channels               = cfg.sar_channels,
        pretrained_encoders        = True,
        freeze_backbones           = False,
    ).to(device)

    logger.info("Model parameters: %s", f"{model.n_parameters:,}")

    # ---- Loss, Optimiser, Scheduler ----
    loss_fn = CombinedLoss(
        temperature  = cfg.temperature,
        aux_weight   = cfg.aux_consistency_weight,
        use_aux_loss = True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * cfg.num_epochs
    warmup_steps    = steps_per_epoch * cfg.warmup_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = make_scaler(cfg.use_amp, device)

    metrics_logger = MetricsLogger(ckpt_dir / "metrics.jsonl")

    # ---- Resume ----
    start_epoch  = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        state = load_checkpoint(
            resume_path, model, optimizer, scheduler, device=device
        )
        start_epoch   = state.get("epoch", 0) + 1
        best_val_loss = state.get("metrics", {}).get("val_loss", float("inf"))
        logger.info("Resuming from epoch %d (best_val_loss=%.4f)", start_epoch, best_val_loss)

    # ---- Epoch loop ----
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_metrics: Dict[str, float] = {}
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=True)
        for step, batch in enumerate(progress):
            step_metrics = train_one_step(
                model, batch, loss_fn, optimizer, scaler, cfg, device
            )
            scheduler.step()

            train_loss += step_metrics["total_loss"]
            avg_loss = train_loss / (step + 1)

            progress.set_postfix({
                "loss":    f"{avg_loss:.4f}",
                "lr":      f"{scheduler.get_last_lr()[0]:.2e}",
                "acc_o2s": f"{step_metrics.get('acc_opt_to_sar', 0):.2f}",
            })

        # Per-epoch validation
        val_metrics = evaluate_retrieval(model, val_loader, device)

        epoch_metrics = {
            "epoch":      epoch,
            "train_loss": train_loss / len(train_loader),
            "lr":         scheduler.get_last_lr()[0],
            **val_metrics,
        }
        metrics_logger.log(epoch_metrics)

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | opt→sar R@1=%.3f | sar→opt R@1=%.3f",
            epoch + 1,
            epoch_metrics["train_loss"],
            epoch_metrics.get("val_loss", 0),
            epoch_metrics.get("opt_to_sar/R@1", 0),
            epoch_metrics.get("sar_to_opt/R@1", 0),
        )

        # Save last checkpoint every epoch
        save_checkpoint(
            ckpt_dir / "last.pt",
            model, optimizer, scheduler, epoch, epoch_metrics, cfg,
        )

        # Save best checkpoint
        val_loss = epoch_metrics.get("val_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler, epoch, epoch_metrics, cfg,
            )
            logger.info("New best model saved (val_loss=%.4f)", best_val_loss)

        # Periodic checkpoint every N epochs
        if (epoch + 1) % cfg.save_every_n_epochs == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
                model, optimizer, scheduler, epoch, epoch_metrics, cfg,
            )

    metrics_logger.close()
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpticalSAR contrastive model")
    parser.add_argument("--resume",      type=str,  default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--epochs",      type=int,  default=None,   help="Override num_epochs")
    parser.add_argument("--batch_size",  type=int,  default=None,   help="Override batch_size")
    parser.add_argument("--lr",          type=float,default=None,   help="Override learning_rate")
    parser.add_argument("--max_samples", type=int,  default=None,   help="Max training samples")
    parser.add_argument("--streaming",   action="store_true",        help="Use HF streaming mode")
    parser.add_argument("--cross_attn",  action="store_true",        help="Use cross-attention fusion")
    parser.add_argument("--no_amp",      action="store_true",        help="Disable AMP")
    parser.add_argument("--seed",        type=int,  default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = ModelConfig()
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.max_samples:
        cfg.max_train_samples = args.max_samples
    if args.streaming:
        cfg.streaming = True
    if args.cross_attn:
        cfg.use_cross_attention_fusion = True
    if args.no_amp:
        cfg.use_amp = False
    cfg.seed = args.seed

    train(cfg, resume_path=args.resume)
