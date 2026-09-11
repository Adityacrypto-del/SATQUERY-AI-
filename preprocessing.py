"""
preprocessing.py — Sentinel-2 (optical 13-band) and Sentinel-1 (SAR 2-channel) input pipelines.
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np
import torch

from config import ModelConfig


class S2Preprocessor:
    """
    Sentinel-2 13-band preprocessing:
    1. Divide by scale_factor (default 10,000)
    2. Clip to [0, 1.5]
    3. Replace NaN/Inf with 0.0
    4. Per-band Z-score normalisation: (x - mean_b) / std_b
    """

    def __init__(self, means: list[float], stds: list[float], scale_factor: float = 10000.0) -> None:
        self.means = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(-1, 1, 1)
        self.scale_factor = float(scale_factor)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.astype(np.float32)

        # 1. Scale DN to reflectance
        x = x / self.scale_factor

        # 2. Clip range
        x = np.clip(x, 0.0, 1.5)

        # 3. Clean NaNs/Infs
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Z-score normalise
        x = (x - self.means) / (self.stds + 1e-7)

        return torch.from_numpy(x).float()


class S1Preprocessor:
    """
    Sentinel-1 SAR 2-channel preprocessing:
    1. Replace NaN/Inf with noise floor (1e-6)
    2. Clip to [1e-6, 1.0]
    3. Per-channel Z-score normalisation: (x - mean_c) / std_c
    """

    def __init__(self, means: list[float], stds: list[float]) -> None:
        self.means = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.astype(np.float32)

        # 1. Replace NaNs/Infs
        x = np.nan_to_num(x, nan=1e-6, posinf=1.0, neginf=1e-6)

        # 2. Clip range
        x = np.clip(x, 1e-6, 1.0)

        # 3. Z-score normalise
        x = (x - self.means) / (self.stds + 1e-7)

        return torch.from_numpy(x).float()


def build_preprocessors(config: ModelConfig) -> Tuple[S2Preprocessor, S1Preprocessor]:
    """Factory helper returning (S2Preprocessor, S1Preprocessor) initialized from config."""
    s2_pre = S2Preprocessor(
        means=config.s2_means,
        stds=config.s2_stds,
        scale_factor=config.s2_scale_factor,
    )
    s1_pre = S1Preprocessor(
        means=config.s1_means,
        stds=config.s1_stds,
    )
    return s2_pre, s1_pre
