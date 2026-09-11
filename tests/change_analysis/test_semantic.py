"""Tests for segmentation inference (build-order step 6).

Checkpoint-agnostic on purpose: the wrapper reads the architecture from the
checkpoint it is handed, so it works with either the two-decoder or the
shared-change-head model without the caller needing to know which won.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.change_analysis.io import RSImage
from tools.change_analysis.semantic import SemanticSegmenter


def _rsimage(bands=3, height=64, width=64, scale=255.0, band_names=None):
    rng = np.random.default_rng(0)
    array = rng.uniform(0, scale, (bands, height, width)).astype(np.float32)
    return RSImage(
        array=array, crs=None, transform=None, modality="optical",
        band_names=band_names, gsd_m=None,
    )


def _write_checkpoint(path, arch="shared"):
    """Save an untrained model of the given architecture as a checkpoint."""
    from segmentation.train import SharedChangeNet, SiameseChangeNet

    torch.manual_seed(0)
    model = SharedChangeNet(pretrained=False) if arch == "shared" \
        else SiameseChangeNet(pretrained=False)
    torch.save(
        {"model": model.state_dict(), "config": {"arch": arch},
         "epoch": 7, "val_report": {"average_accuracy": 0.5}},
        path,
    )
    return str(path)


@pytest.fixture(scope="module")
def shared_ckpt(tmp_path_factory):
    return _write_checkpoint(tmp_path_factory.mktemp("ck") / "shared.pt", "shared")


@pytest.fixture(scope="module")
def siamese_ckpt(tmp_path_factory):
    return _write_checkpoint(tmp_path_factory.mktemp("ck") / "siamese.pt", "siamese")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SemanticSegmenter(str(tmp_path / "nope.pt"), device="cpu")


def test_architecture_is_read_from_the_checkpoint(shared_ckpt, siamese_ckpt):
    """The caller should not have to know which architecture won."""
    assert SemanticSegmenter(shared_ckpt, device="cpu").arch == "shared"
    assert SemanticSegmenter(siamese_ckpt, device="cpu").arch == "siamese"


def test_metadata_is_traceable(shared_ckpt):
    meta = SemanticSegmenter(shared_ckpt, device="cpu").metadata()

    assert meta["arch"] == "shared"
    assert meta["epoch"] == 7
    assert meta["checkpoint"].endswith("shared.pt")


# --------------------------------------------------------------------------
# Prediction contract
# --------------------------------------------------------------------------


def test_predict_returns_two_class_maps(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    s1, s2 = seg.predict(_rsimage(), _rsimage())

    for s in (s1, s2):
        assert s.shape == (64, 64)
        assert s.dtype == np.int64 or s.dtype == np.int32
        assert s.min() >= 0 and s.max() <= 6


def test_output_shape_matches_input_even_when_not_multiple_of_32(shared_ckpt):
    """A 70x50 tile must come back 70x50, not padded to 96x64."""
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    s1, s2 = seg.predict(_rsimage(height=70, width=50),
                         _rsimage(height=70, width=50))

    assert s1.shape == (70, 50)
    assert s2.shape == (70, 50)


def test_shared_checkpoint_yields_consistent_change_masks(shared_ckpt):
    """The architecture's guarantee must survive the inference wrapper."""
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    s1, s2 = seg.predict(_rsimage(), _rsimage())

    assert np.array_equal(s1 == 0, s2 == 0)


def test_mismatched_shapes_raise(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu")

    with pytest.raises(ValueError):
        seg.predict(_rsimage(height=64), _rsimage(height=32))


def test_single_band_input_is_rejected_not_guessed(shared_ckpt):
    """The model was trained on RGB. Feeding it one band and replicating it
    would produce a confident, meaningless answer."""
    seg = SemanticSegmenter(shared_ckpt, device="cpu")

    with pytest.raises(ValueError) as excinfo:
        seg.predict(_rsimage(bands=1), _rsimage(bands=1))

    assert "rgb" in str(excinfo.value).lower()


def test_sar_modality_is_rejected(shared_ckpt):
    """Segmentation is the optical path; SAR goes through the detector."""
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    sar = _rsimage(bands=1)
    sar.modality = "sar"

    with pytest.raises(ValueError):
        seg.predict(sar, sar)


# --------------------------------------------------------------------------
# Value scaling -- the assumption must be recorded, not hidden
# --------------------------------------------------------------------------


def test_byte_range_input_is_detected_and_recorded(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    seg.predict(_rsimage(scale=255.0), _rsimage(scale=255.0))

    assert seg.metadata()["value_scaling"] == "assumed_0_255"


def test_reflectance_range_input_is_detected_and_recorded(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    seg.predict(_rsimage(scale=1.0), _rsimage(scale=1.0))

    assert seg.metadata()["value_scaling"] == "assumed_0_1"


def test_explicit_scaling_overrides_the_heuristic(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu", value_range=(0.0, 255.0))
    seg.predict(_rsimage(scale=1.0), _rsimage(scale=1.0))

    assert seg.metadata()["value_scaling"] == "explicit_0.0_255.0"


def test_more_than_three_bands_uses_named_rgb_when_available(shared_ckpt):
    seg = SemanticSegmenter(shared_ckpt, device="cpu")
    image = _rsimage(bands=4, band_names=["B08", "B04", "B03", "B02"])

    seg.predict(image, image)

    assert seg.metadata()["band_selection"] == "declared_rgb_bands"
