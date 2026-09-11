"""Tests for Run 5's single change: unchanged-pixel supervision of the class heads.

Run 4 masked the class loss to changed pixels via ``ignore_index``. That was
correct in that SECOND carries no land-cover label for an unchanged pixel, but
it had two unintended effects, both visible in the Run 1 vs Run 4 comparison:

*   the class heads saw 18.65% of pixels instead of 100%, roughly 5x less
    supervision per epoch for the six-way decision;
*   removing class 0 from the class loss implicitly rebalanced the remaining
    six classes, lifting rare ones (water +6.79 IoU, playgrounds +8.33) and
    depressing the common ones (buildings -1.50, low_veg -1.01) that the area
    arithmetic behind increase/decrease/change_ratio_types depends on.

Run 5 restores the supervision without inventing a label: a seventh
"unchanged" logit on each class head, supervised on a *sampled* fraction of
unchanged pixels. ``predict`` reads only the six land-cover channels, so the
change head still owns the changed/unchanged decision outright.

Exactly one knob moves relative to Run 4. The binary change weight stays at
0.4, which measured -4.2% against ground truth and is settled.
"""

from __future__ import annotations

import math

import pytest
import torch

from segmentation.train import (
    SharedChangeLoss,
    SharedChangeNet,
    unchanged_sample_rate,
)


def _net(**kwargs):
    torch.manual_seed(0)
    return SharedChangeNet(pretrained=False, **kwargs)


def _batch(n=2, size=64):
    torch.manual_seed(1)
    return torch.randn(n, 6, size, size)


# --------------------------------------------------------------------------
# The structural guarantee must survive the extra logit
# --------------------------------------------------------------------------


def test_disagreement_is_still_exactly_zero_with_the_unchanged_logit():
    """The whole point of the architecture; the new channel must not cost it."""
    p1, p2 = _net(unchanged_logit=True).predict(_batch())

    assert torch.equal(p1 == 0, p2 == 0)
    assert float(((p1 == 0) != (p2 == 0)).float().mean()) == 0.0


def test_predict_never_emits_the_unchanged_logit_as_a_land_cover_class():
    """argmax runs over the six land-cover channels only. If it ran over all
    seven, the seventh would decode to class 7 -- outside the palette and
    outside every rule in cdvqa.py."""
    net = _net(unchanged_logit=True)
    # Make the unchanged logit dominate everywhere, the worst case.
    with torch.no_grad():
        for head in net.class_heads:
            head.bias.zero_()
            head.bias[6] = 50.0

    p1, p2 = net.predict(_batch())

    assert int(p1.max()) <= 6 and int(p2.max()) <= 6
    assert int(p1.min()) >= 0 and int(p2.min()) >= 0


def test_head_width_follows_the_flag():
    assert _net(unchanged_logit=False).class_heads[0].out_channels == 6
    assert _net(unchanged_logit=True).class_heads[0].out_channels == 7


# --------------------------------------------------------------------------
# The sampling rate is derived from measured counts, never chosen
# --------------------------------------------------------------------------


def test_sample_rate_balances_class_zero_against_the_mean_changed_class():
    """p = (changed / 6) / unchanged, so the sampled unchanged pixels arrive
    at the same expected frequency as an average land-cover class."""
    counts = {0: 8135, 1: 1000, 2: 1000, 3: 1000, 4: 500, 5: 300, 6: 200}
    changed = sum(v for k, v in counts.items() if k)

    rate = unchanged_sample_rate(counts)

    assert rate == pytest.approx((changed / 6) / counts[0])
    assert 0.0 < rate < 1.0


def test_sample_rate_is_clamped_and_safe_on_degenerate_counts():
    assert unchanged_sample_rate({0: 0, 1: 10}) == 0.0
    assert unchanged_sample_rate({0: 10, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}) == 0.0
    # More changed than unchanged would ask for a rate above 1; a probability
    # cannot exceed 1, and clamping is honest where scaling would not be.
    assert unchanged_sample_rate({0: 1, 1: 600}) == 1.0


# --------------------------------------------------------------------------
# The loss actually supervises those pixels
# --------------------------------------------------------------------------


def _targets(size=32, changed_rows=8):
    y = torch.zeros(2, size, size, dtype=torch.long)
    y[:, :changed_rows, :] = 3
    return y


def test_rate_zero_reproduces_run4_masking_exactly():
    """The default path must be bit-identical to Run 4, so a null knob is a
    null result rather than a silent second change."""
    torch.manual_seed(2)
    weight = torch.tensor([0.5, 1.5])
    outputs = (torch.randn(2, 2, 32, 32), torch.randn(2, 7, 32, 32),
               torch.randn(2, 7, 32, 32))
    y = _targets()

    masked = SharedChangeLoss(weight, unchanged_sample_rate=0.0)
    reference = SharedChangeLoss(weight)

    assert torch.equal(masked(outputs, y, y), reference(outputs, y, y))


def test_unchanged_pixels_reach_the_class_heads_when_the_rate_is_positive():
    """Gradient reaching *unchanged* pixels is the measurable form of "these
    pixels are now supervised". Under Run 4's ignore_index it is exactly zero
    there. Measured at the unchanged pixels rather than on channel 6 alone:
    the softmax denominator gives that channel gradient at changed pixels in
    either case, so channel 6 on its own would not isolate the knob."""
    y = _targets()
    weight = torch.tensor([0.5, 1.5])
    unchanged = (y == 0)

    def unchanged_pixel_grad(rate):
        torch.manual_seed(3)
        logits = torch.randn(2, 7, 32, 32, requires_grad=True)
        change = torch.randn(2, 2, 32, 32)
        loss = SharedChangeLoss(weight, unchanged_sample_rate=rate)(
            (change, logits, logits), y, y
        )
        loss.backward()
        return float(logits.grad.abs().sum(1)[unchanged].sum())

    assert unchanged_pixel_grad(0.0) == 0.0
    assert unchanged_pixel_grad(1.0) > 0.0


def test_changed_pixel_targets_are_untouched_by_sampling():
    """Sampling may only add unchanged pixels. If it perturbed the changed
    ones it would be two knobs, not one."""
    torch.manual_seed(4)
    y = _targets()
    loss = SharedChangeLoss(torch.tensor([0.5, 1.5]), unchanged_sample_rate=1.0)

    shifted = loss.class_targets(y)

    changed = y != 0
    assert torch.equal(shifted[changed], y[changed] - 1)
    # Every unchanged pixel is labelled with the unchanged index at rate 1.0.
    assert torch.all(shifted[~changed] == 6)


def test_sampling_is_stochastic_and_hits_roughly_the_requested_rate():
    torch.manual_seed(5)
    y = torch.zeros(4, 64, 64, dtype=torch.long)  # all unchanged
    loss = SharedChangeLoss(torch.tensor([0.5, 1.5]), unchanged_sample_rate=0.25)

    hit = (loss.class_targets(y) == 6).float().mean().item()

    assert hit == pytest.approx(0.25, abs=0.02)


def test_a_six_wide_head_rejects_a_positive_rate():
    """Asking for unchanged supervision from a head with nowhere to put it is
    a configuration error, not something to silently ignore."""
    loss = SharedChangeLoss(torch.tensor([0.5, 1.5]), unchanged_sample_rate=0.25)
    outputs = (torch.randn(2, 2, 32, 32), torch.randn(2, 6, 32, 32),
               torch.randn(2, 6, 32, 32))

    with pytest.raises(ValueError, match="unchanged"):
        loss(outputs, _targets(), _targets())
