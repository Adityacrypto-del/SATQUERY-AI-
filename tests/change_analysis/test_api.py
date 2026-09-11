"""Tests for the branch-2 public interface.

The controller sees only this. What it can do here, and what it is protected
from, is the whole of the integration contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.api import BiTemporalSpecialist, describe_tool, _coerce
from tools.change_analysis.io import RSImage
from tools.change_analysis.overlay import DISPLAY_PALETTE, focus_comparison, focus_legend


def _image(bands=3, size=32, names=("red", "green", "blue"), modality="optical"):
    rng = np.random.default_rng(0)
    return RSImage(
        array=rng.uniform(0, 1, (bands, size, size)).astype(np.float32),
        crs=None, transform=None, modality=modality,
        band_names=list(names) if names else None, gsd_m=None,
    )


# -- the registry descriptor ------------------------------------------------


def test_descriptor_declares_limitations_not_just_capabilities():
    """A controller that reads the limitations can route around them instead
    of discovering them at runtime."""
    tool = describe_tool()

    assert tool["name"]
    assert tool["accepts"] and tool["returns"]
    text = " ".join(tool["limitations"]).lower()
    assert "sar" in text, "the SAR refusal must be declared"
    assert "rare" in text or "water" in text, "rare-class blindness must be declared"


def test_descriptor_accuracy_carries_its_caveat():
    """Quoting 68.14% without the near-duplicate bound would overstate it."""
    accuracy = describe_tool()["measured_accuracy"]

    assert accuracy["average_accuracy"] == pytest.approx(0.6814)
    assert "caveat" in accuracy and "0.677" in accuracy["caveat"]


# -- input coercion ---------------------------------------------------------


def test_accepts_an_rsimage_unchanged():
    image = _image()
    assert _coerce(image) is image


def test_accepts_a_channels_last_array_as_a_controller_would_have():
    coerced = _coerce(np.zeros((64, 64, 3), dtype=np.float32))

    assert coerced.array.shape == (3, 64, 64)


def test_refuses_input_it_cannot_interpret():
    with pytest.raises(TypeError, match="cannot interpret"):
        _coerce({"not": "an image"})


# -- the one call -----------------------------------------------------------


def test_analyze_returns_evidence_with_everything_the_ps_grades():
    evidence = BiTemporalSpecialist(overlay_dir=None).analyze(
        _image(), _image(), "What changed and where?"
    )

    for field in ("changed", "summary", "confidence", "confidence_basis",
                  "trace", "overlays", "route"):
        assert hasattr(evidence, field)
    assert evidence.trace, "the trace is a graded artefact and must be populated"
    assert evidence.confidence_basis, "a bare confidence is unreadable"


def test_a_broken_pair_is_reported_not_raised():
    """A controller must get an explanation, never a stack trace."""
    evidence = BiTemporalSpecialist(overlay_dir=None).analyze(
        _image(size=32), _image(size=16), "did buildings change?"
    )

    assert evidence.changed is False or evidence.answer is None
    assert "reject" in evidence.summary.lower() or "differ" in evidence.summary.lower()


# -- the query-scoped figure ------------------------------------------------


def test_focus_comparison_is_saveable_uint8_and_side_by_side():
    t1, t2 = _image(size=16), _image(size=16)
    s1 = np.zeros((16, 16), dtype=np.int64); s1[:8] = 5   # water
    s2 = np.zeros((16, 16), dtype=np.int64); s2[:8] = 4   # buildings

    figure = focus_comparison(t1, t2, s1, s2, target_class="water")

    assert figure.dtype == np.uint8, "save_png expects uint8; floats render black"
    assert figure.shape[0] == 16 and figure.shape[1] > 32


def test_focus_scopes_to_the_queried_class_only():
    """Asking about water must not highlight an unrelated change."""
    t1, t2 = _image(size=16), _image(size=16)
    s1 = np.zeros((16, 16), dtype=np.int64); s1[:4] = 5; s1[12:] = 3
    s2 = np.zeros((16, 16), dtype=np.int64); s2[:4] = 4; s2[12:] = 1

    water = focus_comparison(t1, t2, s1, s2, target_class="water")
    everything = focus_comparison(t1, t2, s1, s2, target_class=None)

    assert not np.array_equal(water, everything)


def test_nvg_surface_is_not_rendered_grey_on_grey():
    """The defect this palette exists for: SECOND paints NVG mid-grey, which
    vanished against asphalt and made a correct figure read as its opposite."""
    from tools.change_analysis.cdvqa import PALETTE

    analysis_grey = [rgb for rgb, index in PALETTE.items() if index == 1][0]
    display, name = DISPLAY_PALETTE[1]

    assert name == "NVG_surface"
    assert display != analysis_grey
    assert max(display) - min(display) > 60, "must be saturated, not another grey"


def test_legend_names_only_the_classes_present():
    legend = focus_legend(np.array([[0, 5]]), np.array([[0, 4]]))

    assert set(legend) == {"water", "buildings"}
    assert all(v.startswith("#") for v in legend.values())


def test_an_unknown_class_is_refused_rather_than_guessed():
    t1, t2 = _image(size=16), _image(size=16)
    zeros = np.zeros((16, 16), dtype=np.int64)

    with pytest.raises(ValueError, match="unknown land-cover class"):
        focus_comparison(t1, t2, zeros, zeros, target_class="parking_lot")
