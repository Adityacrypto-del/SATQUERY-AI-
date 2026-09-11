"""Segmentation inference: two images in, two semantic change maps out.

This is the bridge between the trained model and the deterministic rules.
It is deliberately checkpoint-agnostic -- the architecture is read from the
checkpoint, so the caller never has to know whether the two-decoder or the
shared-change-head model won.

**Class 0 means "unchanged", not "background land cover".** The maps this
produces are semantic *change* maps: a pixel carries a land-cover class only
where it changed, giving what it was (s_t1) and what it became (s_t2).

Two things this refuses to do, both for the same reason -- a confident wrong
answer is worse than an honest refusal:

*   Guess at non-RGB input. The model was trained on 3-band RGB; replicating
    a single band or silently taking the first three of a multispectral
    stack would produce a fluent, meaningless segmentation.
*   Hide its value-scaling assumption. Whether input is 0-255 or 0-1
    reflectance changes the answer, so the assumption made is recorded in
    :meth:`metadata` and travels into the execution trace.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .io import RSImage

__all__ = ["SemanticSegmenter"]

# Matches segmentation.dataset -- inference must normalise exactly as
# training did or the model sees a different distribution than it learned.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The encoder downsamples by 32, so inputs are padded up to a multiple and
# the prediction cropped back to the true size.
_STRIDE = 32

_RGB_ALIASES = (
    {"red", "r", "b04", "b4"},
    {"green", "g", "b03", "b3"},
    {"blue", "b", "b02", "b2"},
)


class SemanticSegmenter:
    """Runs a trained checkpoint over an image pair."""

    def __init__(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        import torch

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint!r}")

        self.checkpoint = checkpoint
        self._torch = torch
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        config = payload.get("config") or {}
        # Run 1 and Run 2 predate the `arch` config field, so it is absent
        # from their checkpoints. Falling back to a default would pick the
        # architecture by luck; the weights themselves say which it is, and
        # only the shared model has a change_head.
        state = payload["model"]
        inferred = "shared" if any(
            k.startswith("change_head") for k in state
        ) else "siamese"
        declared = config.get("arch")
        self.arch = declared or inferred
        self.arch_source = "declared" if declared else "inferred_from_weights"
        if declared and declared != inferred:
            raise ValueError(
                f"checkpoint declares arch={declared!r} but its weights look "
                f"like {inferred!r}; refusing to load a mismatched model"
            )
        self.epoch = payload.get("epoch")
        self.val_report = payload.get("val_report") or {}
        self._explicit_range = value_range
        self._last_scaling: Optional[str] = None
        self._last_band_selection: Optional[str] = None

        from segmentation.train import SharedChangeNet, SiameseChangeNet

        if self.arch == "shared":
            # Run 5 checkpoints carry a seventh "unchanged" logit on each
            # class head. Read the width from the weights rather than assume
            # it -- the same reason arch itself is inferred above.
            head = state.get("class_heads.0.weight")
            width = int(head.shape[0]) if head is not None else 6
            model = SharedChangeNet(pretrained=False, unchanged_logit=width > 6)
        elif self.arch == "siamese":
            model = SiameseChangeNet(pretrained=False)
        else:
            raise ValueError(f"unknown architecture in checkpoint: {self.arch!r}")
        model.load_state_dict(payload["model"])
        self.model = model.to(self.device).eval()

    # -- metadata ---------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """Provenance for the execution trace."""
        return {
            "checkpoint": self.checkpoint,
            "arch": self.arch,
            "arch_source": self.arch_source,
            "epoch": self.epoch,
            "device": str(self.device),
            "val_average_accuracy": self.val_report.get("average_accuracy"),
            "value_scaling": self._last_scaling,
            "band_selection": self._last_band_selection,
        }

    # -- band handling ----------------------------------------------------

    def _rgb_planes(self, image: RSImage) -> np.ndarray:
        """Return an ``(3, H, W)`` RGB stack, or raise.

        Never replicates a single band and never silently takes the first
        three of a multispectral stack when the bands are unnamed -- both
        would yield a confident answer the model has no basis for.
        """
        if image.modality == "sar":
            raise ValueError(
                "segmentation is the optical path; SAR input must go through "
                "the ratio-operator detector instead"
            )
        if image.n_bands < 3:
            raise ValueError(
                f"the segmentation model requires 3-band RGB input; this image "
                f"has {image.n_bands} band(s). Replicating a single band would "
                f"produce a confident but meaningless segmentation."
            )
        if image.band_names:
            indices = []
            for aliases in _RGB_ALIASES:
                found = next(
                    (i for i, name in enumerate(image.band_names)
                     if str(name).strip().lower() in aliases),
                    None,
                )
                indices.append(found)
            if all(i is not None for i in indices):
                self._last_band_selection = "declared_rgb_bands"
                return image.array[[int(i) for i in indices]]
        if image.n_bands == 3:
            self._last_band_selection = "three_band_assumed_rgb"
            return image.array[:3]
        raise ValueError(
            f"image has {image.n_bands} bands with names {image.band_names!r}; "
            "cannot identify which are red/green/blue. Name the bands or pass "
            "a 3-band RGB image."
        )

    def _scale(self, planes: np.ndarray) -> np.ndarray:
        """Bring values into 0-1, recording which assumption was used."""
        if self._explicit_range is not None:
            low, high = self._explicit_range
            self._last_scaling = f"explicit_{low}_{high}"
            return (planes - low) / max(high - low, 1e-6)
        finite = planes[np.isfinite(planes)]
        peak = float(np.nanmax(finite)) if finite.size else 0.0
        if peak > 1.5:
            self._last_scaling = "assumed_0_255"
            return planes / 255.0
        self._last_scaling = "assumed_0_1"
        return planes

    # -- inference --------------------------------------------------------

    def predict(self, t1: RSImage, t2: RSImage) -> Tuple[np.ndarray, np.ndarray]:
        """Predict ``(s_t1, s_t2)`` as ``(H, W)`` int class maps in 0..6."""
        if t1.shape != t2.shape:
            raise ValueError(
                f"t1 and t2 must share a spatial shape; got {t1.shape} and {t2.shape}"
            )
        torch = self._torch

        stacks = []
        for image in (t1, t2):
            planes = self._scale(np.nan_to_num(self._rgb_planes(image), nan=0.0))
            planes = np.clip(planes, 0.0, 1.0)
            planes = (planes - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
            stacks.append(planes.astype(np.float32))

        height, width = t1.shape
        pad_h = (-height) % _STRIDE
        pad_w = (-width) % _STRIDE
        x = np.concatenate(stacks, axis=0)[None]
        if pad_h or pad_w:
            # Reflect rather than zero-pad: a black border is a strong,
            # false edge signal for a change detector.
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            use_amp = self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                p1, p2 = self.model.predict(tensor)

        s1 = p1[0].to(torch.int64).cpu().numpy()[:height, :width]
        s2 = p2[0].to(torch.int64).cpu().numpy()[:height, :width]
        return s1, s2
