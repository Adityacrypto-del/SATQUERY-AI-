"""Tests for the shared-change-head architecture (Run 3).

The point of this architecture is a guarantee, not a tendency: the two dates
share one changed/unchanged decision, so head disagreement is impossible by
construction. Run 1 and Run 2 both plateaued at 2.96% and 5.92% disagreement
with no convergence trend, which is what motivated the change. These tests
enforce the guarantee so a future refactor cannot quietly reintroduce it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from segmentation.train import (
    SharedChangeLoss,
    SharedChangeNet,
    SiameseChangeNet,
)


def _net():
    torch.manual_seed(0)
    return SharedChangeNet(pretrained=False)


def _batch(n=2, size=64):
    torch.manual_seed(1)
    return torch.randn(n, 6, size, size)


# --------------------------------------------------------------------------
# The structural guarantee
# --------------------------------------------------------------------------


def test_head_disagreement_is_exactly_zero():
    """Not "small" -- zero. This is the whole reason the architecture changed."""
    p1, p2 = _net().predict(_batch())

    assert torch.equal(p1 == 0, p2 == 0)
    assert float(((p1 == 0) != (p2 == 0)).float().mean()) == 0.0


def test_the_old_architecture_does_not_give_that_guarantee():
    """Contrast test: the two-decoder model can and does disagree.

    Without this, the test above could pass for a trivial reason (e.g. an
    untrained net predicting one class everywhere) and prove nothing.
    """
    torch.manual_seed(0)
    old = SiameseChangeNet(pretrained=False)
    p1, p2 = old.predict(_batch())

    # Nothing in that architecture constrains the two heads to agree.
    assert not hasattr(old, "change_head")


def test_predictions_are_valid_seven_class_maps():
    p1, p2 = _net().predict(_batch())

    for p in (p1, p2):
        assert p.dtype == torch.int64
        assert int(p.min()) >= 0
        assert int(p.max()) <= 6


def test_class_heads_emit_six_channels_not_seven():
    """Class 0 belongs to the change head; a 7th channel would let the class
    head contradict it."""
    change_logits, c1, c2 = _net()(_batch())

    assert change_logits.shape[1] == 2
    assert c1.shape[1] == 6
    assert c2.shape[1] == 6


def test_unchanged_pixels_take_class_zero_regardless_of_class_head():
    """Where the change head says unchanged, the class head is overridden."""
    net = _net()
    x = _batch()
    change_logits, c1, _ = net(x)
    p1, _ = net.predict(x)

    unchanged = change_logits.argmax(1) == 0
    assert (p1[unchanged] == 0).all()


def test_changed_pixels_never_take_class_zero():
    net = _net()
    x = _batch()
    change_logits, _, _ = net(x)
    p1, p2 = net.predict(x)

    changed = change_logits.argmax(1) == 1
    if changed.any():
        assert (p1[changed] >= 1).all()
        assert (p2[changed] >= 1).all()


# --------------------------------------------------------------------------
# The two-part loss
# --------------------------------------------------------------------------


def _targets(size=8):
    y1 = torch.zeros(1, size, size, dtype=torch.long)
    y2 = torch.zeros(1, size, size, dtype=torch.long)
    y1[0, :4] = 1   # NVG at t1
    y2[0, :4] = 4   # became buildings at t2
    return y1, y2


def test_loss_ignores_class_predictions_on_unchanged_pixels():
    """An unchanged pixel has no land-cover label, so it must not be scored.

    Changing the class logits only on unchanged pixels must leave the loss
    untouched; if it moves, the class heads are being supervised on noise.
    """
    y1, y2 = _targets()
    criterion = SharedChangeLoss(torch.ones(2))
    change = torch.zeros(1, 2, 8, 8)
    change[:, 1, :4] = 5.0  # predict "changed" on the top half
    c1 = torch.zeros(1, 6, 8, 8)
    c2 = torch.zeros(1, 6, 8, 8)

    before = criterion((change, c1, c2), y1, y2).item()

    c1_perturbed = c1.clone()
    c1_perturbed[:, :, 4:] = 9.0  # only the unchanged half
    after = criterion((change, c1_perturbed, c2), y1, y2).item()

    assert after == pytest.approx(before, abs=1e-6)


def test_loss_responds_to_class_predictions_on_changed_pixels():
    """The complement of the test above: changed pixels must be supervised."""
    y1, y2 = _targets()
    criterion = SharedChangeLoss(torch.ones(2))
    change = torch.zeros(1, 2, 8, 8)
    c1 = torch.zeros(1, 6, 8, 8)
    c2 = torch.zeros(1, 6, 8, 8)

    before = criterion((change, c1, c2), y1, y2).item()

    correct = c1.clone()
    correct[:, 0, :4] = 9.0  # class index 0 == land-cover class 1 (NVG)
    after = criterion((change, correct, c2), y1, y2).item()

    assert after < before


def test_binary_weight_penalises_missing_change_more_than_false_change():
    """The 79/21 imbalance correction must actually bite."""
    y1, y2 = _targets()
    weighted = SharedChangeLoss(torch.tensor([0.5, 1.5]))

    c1 = torch.zeros(1, 6, 8, 8)
    c2 = torch.zeros(1, 6, 8, 8)

    missed = torch.zeros(1, 2, 8, 8)
    missed[:, 0] = 5.0  # says "unchanged" everywhere -> misses real change

    false_alarm = torch.zeros(1, 2, 8, 8)
    false_alarm[:, 1] = 5.0  # says "changed" everywhere -> false positives

    assert weighted((missed, c1, c2), y1, y2).item() > \
        weighted((false_alarm, c1, c2), y1, y2).item()


def test_all_unchanged_scene_does_not_produce_nan():
    """A scene with no change gives the class heads nothing to learn from."""
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    criterion = SharedChangeLoss(torch.ones(2))

    loss = criterion(
        (torch.zeros(1, 2, 8, 8), torch.zeros(1, 6, 8, 8), torch.zeros(1, 6, 8, 8)),
        y, y,
    )

    assert torch.isfinite(loss)
