"""
utils.py — Logging, seeds, device selection, checkpointing, and metrics utilities.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from config import ModelConfig

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configures root logger with formatted output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_scaler(use_amp: bool, device: torch.device) -> Optional[torch.amp.GradScaler]:
    """Builds GradScaler if AMP is enabled on CUDA device."""
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[ModelConfig] = None,
) -> None:
    """Saves complete training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": metrics or {},
        "config": config.to_dict() if config is not None else {},
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()

    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads checkpoint weights and state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    if device is None:
        device = get_device()

    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=False)

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])

    return state


class MetricsLogger:
    """Appends dictionary records as lines of JSON into a .jsonl file."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8")

    def log(self, metrics: Dict[str, Any]) -> None:
        # Convert non-serializable objects (e.g. tensors/numpy floats)
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.generic)):
                clean_metrics[k] = v.item()
            else:
                clean_metrics[k] = v
        self.file.write(json.dumps(clean_metrics) + "\n")
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
