"""
dataset.py — HuggingFace dataset wrapper for SEN12MS-CR optical-SAR pairs with synthetic fallback.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig
from preprocessing import build_preprocessors

logger = logging.getLogger(__name__)

_S2_CANDIDATES = ["s2", "optical", "s2_cloudy", "cloudy", "image_s2", "s2_img"]
_SAR_CANDIDATES = ["s1", "sar", "s1_sar", "image_s1", "s1_img"]


def _find_field(sample: dict, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in sample:
            return key
    return None


class SyntheticOpticalSARDataset(Dataset):
    """
    Synthetic dataset for offline testing, CI, and schema validation.
    """

    def __init__(
        self,
        num_samples: int = 128,
        image_size: int = 256,
        s2_channels: int = 13,
        sar_channels: int = 2,
        s2_pre=None,
        s1_pre=None,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.s2_channels = s2_channels
        self.sar_channels = sar_channels
        self.s2_pre = s2_pre
        self.s1_pre = s1_pre

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate realistic random Sentinel-2 DN values in [0, 8000]
        s2_raw = np.random.uniform(200.0, 4000.0, size=(self.s2_channels, self.image_size, self.image_size)).astype(np.float32)
        # Generate realistic random Sentinel-1 backscatter values in [0.001, 0.5]
        s1_raw = np.random.uniform(0.005, 0.2, size=(self.sar_channels, self.image_size, self.image_size)).astype(np.float32)

        if self.s2_pre is not None:
            s2_tensor = self.s2_pre(s2_raw)
        else:
            s2_tensor = torch.from_numpy(s2_raw)

        if self.s1_pre is not None:
            s1_tensor = self.s1_pre(s1_raw)
        else:
            s1_tensor = torch.from_numpy(s1_raw)

        return {"optical": s2_tensor, "sar": s1_tensor}


class OpticalSARDataset(Dataset):
    """
    In-memory PyTorch Dataset wrapping loaded HuggingFace items.
    """

    def __init__(self, data: list, s2_pre, s1_pre, s2_key: str = "s2", s1_key: str = "s1") -> None:
        self.data = data
        self.s2_pre = s2_pre
        self.s1_pre = s1_pre
        self.s2_key = s2_key
        self.s1_key = s1_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        s2_val = np.array(item[self.s2_key], dtype=np.float32)
        s1_val = np.array(item[self.s1_key], dtype=np.float32)

        # Transpose if (H, W, C) to (C, H, W)
        if s2_val.ndim == 3 and s2_val.shape[-1] == 13:
            s2_val = np.transpose(s2_val, (2, 0, 1))
        if s1_val.ndim == 3 and s1_val.shape[-1] == 2:
            s1_val = np.transpose(s1_val, (2, 0, 1))

        optical = self.s2_pre(s2_val)
        sar = self.s1_pre(s1_val)

        return {"optical": optical, "sar": sar}


def build_dataloaders(config: ModelConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and validation DataLoader objects.
    Attempts to load HuggingFace dataset, falling back gracefully to synthetic data if network/dataset is unavailable.
    """
    s2_pre, s1_pre = build_preprocessors(config)

    use_synthetic = False
    try:
        from datasets import load_dataset

        logger.info("Loading dataset %s...", config.dataset_name)
        if config.streaming:
            hf_ds = load_dataset(config.dataset_name, split="train", streaming=True)
            samples = []
            max_s = config.max_train_samples or 1000
            for i, sample in enumerate(hf_ds):
                if i >= max_s:
                    break
                samples.append(sample)
        else:
            hf_ds = load_dataset(config.dataset_name, split="train")
            if config.max_train_samples is not None:
                hf_ds = hf_ds.select(range(min(len(hf_ds), config.max_train_samples)))
            samples = list(hf_ds)

        if len(samples) == 0:
            raise ValueError("No samples loaded from HuggingFace dataset.")

        s2_key = _find_field(samples[0], _S2_CANDIDATES) or "s2"
        s1_key = _find_field(samples[0], _SAR_CANDIDATES) or "s1"

        # Split into train / val
        val_size = max(1, int(len(samples) * config.val_fraction))
        train_samples = samples[:-val_size] if len(samples) > val_size else samples
        val_samples = samples[-val_size:]

        train_ds = OpticalSARDataset(train_samples, s2_pre, s1_pre, s2_key, s1_key)
        val_ds = OpticalSARDataset(val_samples, s2_pre, s1_pre, s2_key, s1_key)

    except Exception as err:
        logger.warning(
            "Could not load HuggingFace dataset (%s). Falling back to synthetic dataset. Error: %s",
            config.dataset_name, err,
        )
        use_synthetic = True

    if use_synthetic:
        total_samples = config.max_train_samples or 256
        val_samples_count = max(16, int(total_samples * config.val_fraction))
        train_samples_count = max(32, total_samples - val_samples_count)

        train_ds = SyntheticOpticalSARDataset(
            num_samples=train_samples_count,
            image_size=config.image_size,
            s2_channels=config.optical_channels,
            sar_channels=config.sar_channels,
            s2_pre=s2_pre,
            s1_pre=s1_pre,
        )
        val_ds = SyntheticOpticalSARDataset(
            num_samples=val_samples_count,
            image_size=config.image_size,
            s2_channels=config.optical_channels,
            sar_channels=config.sar_channels,
            s2_pre=s2_pre,
            s1_pre=s1_pre,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0 if use_synthetic else config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=len(train_ds) > config.batch_size,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0 if use_synthetic else config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
