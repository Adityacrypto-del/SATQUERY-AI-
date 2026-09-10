"""Tests for the SECOND change-segmentation dataset.

Weighted toward the two things that would silently invalidate a training
run: a held-out scene reaching the training set, and augmentation being
applied to images but not to their labels.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from segmentation.dataset import (
    SECONDChangeDataset,
    assert_split_disjoint,
    cdvqa_split_scenes,
)
from tools.change_analysis.cdvqa import PALETTE

_COLOURS = {index: colour for colour, index in PALETTE.items()}


def _make_scene(root, name, class1=1, class2=4, size=8):
    """Write one synthetic SECOND scene: two images and two label maps."""
    for folder in ("im1", "im2", "label1", "label2"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    for folder in ("im1", "im2"):
        rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(root / folder / name)

    # Left half changed, right half unchanged.
    for folder, cls in (("label1", class1), ("label2", class2)):
        label = np.zeros((size, size, 3), dtype=np.uint8)
        label[:, :] = _COLOURS[0]
        label[:, : size // 2] = _COLOURS[cls]
        Image.fromarray(label).save(root / folder / name)
    return name


def _make_manifest(root, split, scenes):
    entries = []
    for index, scene in enumerate(scenes):
        # Two entries per scene, mirroring CDVQA's question groups.
        entries.extend([
            {"id": index * 2, "file_name": scene},
            {"id": index * 2 + 1, "file_name": scene},
        ])
    (root / f"{split}_images.json").write_text(
        json.dumps({"images": entries}), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Split handling
# --------------------------------------------------------------------------


def test_split_scenes_are_unique_and_sorted(tmp_path):
    _make_manifest(tmp_path, "Train", ["02.png", "01.png", "02.png"])

    assert cdvqa_split_scenes(str(tmp_path), "Train") == ["01.png", "02.png"]


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cdvqa_split_scenes(str(tmp_path), "Nope")


def test_held_out_scene_in_training_set_fails_and_names_the_split():
    with pytest.raises(AssertionError) as excinfo:
        assert_split_disjoint(["a.png", "b.png"], test=["b.png"], val=["c.png"])

    message = str(excinfo.value)
    assert "test" in message and "b.png" in message


def test_disjoint_splits_pass():
    assert assert_split_disjoint(
        ["a.png"], val=["b.png"], test=["c.png"]
    ) is None


# --------------------------------------------------------------------------
# Sample shape and content
# --------------------------------------------------------------------------


def test_sample_stacks_both_dates_on_the_channel_axis(tmp_path):
    _make_scene(tmp_path, "01.png")
    dataset = SECONDChangeDataset(str(tmp_path), ["01.png"])

    x, y1, y2 = dataset[0]

    assert x.shape == (6, 8, 8)
    assert x.dtype == torch.float32
    assert y1.shape == (8, 8) and y2.shape == (8, 8)
    assert y1.dtype == torch.int64


def test_labels_decode_to_class_indices_not_colours(tmp_path):
    _make_scene(tmp_path, "01.png", class1=1, class2=4)
    dataset = SECONDChangeDataset(str(tmp_path), ["01.png"])

    _, y1, y2 = dataset[0]

    assert set(y1.unique().tolist()) == {0, 1}
    assert set(y2.unique().tolist()) == {0, 4}


def test_unchanged_pixels_agree_between_the_two_label_maps(tmp_path):
    """Class 0 is identical in both by construction; the loader must not
    break that invariant."""
    _make_scene(tmp_path, "01.png")
    dataset = SECONDChangeDataset(str(tmp_path), ["01.png"])

    _, y1, y2 = dataset[0]

    assert torch.equal(y1 == 0, y2 == 0)


def test_images_are_normalised_not_raw_bytes(tmp_path):
    _make_scene(tmp_path, "01.png")
    dataset = SECONDChangeDataset(str(tmp_path), ["01.png"])

    x, _, _ = dataset[0]

    # ImageNet-normalised values sit near zero, nowhere near 0..255.
    assert x.abs().max() < 10.0


def test_empty_scene_list_raises(tmp_path):
    with pytest.raises(ValueError):
        SECONDChangeDataset(str(tmp_path), [])


# --------------------------------------------------------------------------
# Augmentation must move images and labels together
# --------------------------------------------------------------------------


def test_augmentation_keeps_labels_aligned_with_images(tmp_path):
    """A flip applied to the image but not the label destroys supervision
    while still training happily, so this is checked structurally."""
    _make_scene(tmp_path, "01.png", class1=1, class2=4, size=8)
    dataset = SECONDChangeDataset(str(tmp_path), ["01.png"], augment=True, seed=3)

    for _ in range(12):
        x, y1, y2 = dataset[0]
        changed = y1 != 0
        # The synthetic scene has exactly half its pixels changed, whatever
        # the orientation, and the two maps must agree on which half.
        assert changed.sum().item() == 32
        assert torch.equal(y1 != 0, y2 != 0)
        # Class identity must survive the geometric transform.
        assert set(y1.unique().tolist()) == {0, 1}
        assert set(y2.unique().tolist()) == {0, 4}
