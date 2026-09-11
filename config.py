"""
config.py — Central configuration for OpticalSAR specialist model.

All tuneable knobs live here. Import ModelConfig from this module;
do NOT scatter magic numbers through the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Sentinel band statistics (approximate, computed over SEN12MS variants)
# Sentinel-2 L1C top-of-atmosphere reflectance values are in [0, 10000]
# (integer DN scaled by 10000). After /10000 they are in [0, 1].
# Sentinel-1 SAR backscatter in linear scale is stored in the dataset as
# float32 already converted from dB. Typical range: 1e-5 to ~1.0.
# These defaults will be overridden after the first dataset inspection pass.
# ---------------------------------------------------------------------------

# Sentinel-2 per-band means and stds (13 bands, B01–B12 + B8A).
# Order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
S2_BAND_MEANS = [
    0.1215, 0.1094, 0.1014, 0.0939, 0.0990, 0.1317, 0.1399,
    0.1371, 0.1403, 0.0532, 0.0008, 0.0979, 0.0679,
]
S2_BAND_STDS = [
    0.0390, 0.0472, 0.0549, 0.0672, 0.0616, 0.0668, 0.0720,
    0.0761, 0.0735, 0.0413, 0.0009, 0.0769, 0.0624,
]

# Sentinel-1 per-channel means and stds (VV, VH) in linear scale
S1_BAND_MEANS = [0.0600, 0.0200]
S1_BAND_STDS  = [0.1000, 0.0450]


@dataclass
class ModelConfig:
    # ---- Architecture ----
    embedding_dim: int = 512          # output embedding size (both modalities + fused)
    projection_dim: int = 256         # internal projection head hidden dim
    encoder_type: str = "resnet18"    # "resnet18" | "resnet34"
    use_cross_attention_fusion: bool = False   # False → concat+MLP; True → cross-attn

    # ---- Input channels ----
    optical_channels: int = 13        # Sentinel-2 bands
    sar_channels: int = 2             # Sentinel-1 VV + VH

    # ---- Contrastive loss ----
    temperature: float = 0.07        # InfoNCE temperature (CLIP default)

    # ---- Training ----
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True              # automatic mixed precision

    # ---- Auxiliary loss ----
    aux_consistency_weight: float = 0.1  # weight for fused↔modality consistency loss

    # ---- Data ----
    dataset_name: str = "Hermanni/sen12mscr"
    image_size: int = 256             # expected spatial size from dataset
    val_fraction: float = 0.1
    max_train_samples: Optional[int] = None   # None = full; set e.g. 10000 for subset
    streaming: bool = False           # True for initial exploration

    # ---- Normalization ----
    s2_means: list = field(default_factory=lambda: S2_BAND_MEANS)
    s2_stds:  list = field(default_factory=lambda: S2_BAND_STDS)
    s1_means: list = field(default_factory=lambda: S1_BAND_MEANS)
    s1_stds:  list = field(default_factory=lambda: S1_BAND_STDS)
    s2_scale_factor: float = 10000.0  # divide raw S2 ints by this → [0,1] reflectance

    # ---- Checkpointing ----
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


# Convenience singleton for scripts that import directly
DEFAULT_CONFIG = ModelConfig()
