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


# -- concurrency ------------------------------------------------------------


def test_concurrent_calls_do_not_swap_confidences():
    """One specialist serves many calls and a controller may make them at
    once. The softmax margin used to live on the segmenter instance and was
    read back after the fact: measured across six threads, four calls
    returned another call's margin, and so another call's confidence. Wrong
    numbers rather than a crash, which is the harder kind to notice.
    """
    import threading

    class _Segmenter:
        """Returns a margin keyed to the input, so a swap is detectable."""

        val_report = {"per_type": {"change_ratio": {"accuracy": 0.5}}}
        calibration = {
            "per_type_bins": {
                "change_ratio": [
                    {"low": -1e9, "high": 0.5, "n": 500, "accuracy": 0.2,
                     "usable": True},
                    {"low": 0.5, "high": 1e9, "n": 500, "accuracy": 0.9,
                     "usable": True},
                ]
            }
        }
        last_margin = None

        def metadata(self):
            return {"arch": "fake", "value_scaling": "none"}

        def predict_with_margin(self, t1, t2):
            import time

            # The margin is decided by the input; any crossing shows up as a
            # confidence that does not match the image that produced it.
            margin = 0.9 if float(t1.array.mean()) > 0.5 else 0.1
            self.last_margin = margin
            time.sleep(0.01)  # widen the window a real forward pass creates
            shape = t1.array.shape[1:]
            maps = np.zeros(shape, dtype=np.int64)
            maps[:2] = 4
            return maps, maps.copy(), margin

    from tools.change_analysis.pipeline import BiTemporalPipeline, PipelineConfig

    pipeline = BiTemporalPipeline(PipelineConfig(), segmenter=_Segmenter())
    question = "What is the percentage of changed areas?"

    def run(level):
        image = RSImage(
            array=np.full((3, 16, 16), level, dtype=np.float32), crs=None,
            transform=None, modality="optical",
            band_names=["red", "green", "blue"], gsd_m=None,
        )
        return pipeline.run(image, image, question).confidence

    expected = {0.9: run(0.9), 0.1: run(0.1)}
    assert expected[0.9] != expected[0.1], "the fixture must be discriminating"

    seen = {}
    levels = [0.9, 0.1] * 4

    def work(index, level):
        seen[index] = run(level)

    threads = [threading.Thread(target=work, args=(i, l))
               for i, l in enumerate(levels)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for index, level in enumerate(levels):
        assert seen[index] == expected[level], (
            f"call {index} at level {level} returned {seen[index]}, "
            f"expected {expected[level]} -- a margin crossed between calls"
        )


# -- out-of-distribution input ---------------------------------------------


def test_confidence_collapses_when_the_two_producers_flatly_contradict():
    """Measured on a real Sentinel-2 GeoTIFF pair: the SECOND-trained model
    detected no change at all while the index producer detected change across
    34% of the scene, and the system answered "0" to the change-ratio
    question at 0.83 confidence. SECOND is sub-metre aerial imagery and
    Sentinel-2 is 10 m, so the model does not transfer.

    The calibration was measured on inputs where the model does detect
    change; it says nothing about one where the model detects none and
    physics disagrees. Measured-or-zero therefore means zero, and the basis
    has to say which.
    """
    from tools.change_analysis.pipeline import BiTemporalPipeline, PipelineConfig

    class _BlindSegmenter:
        """Detects nothing, and is confident about it."""

        val_report = {"per_type": {"change_ratio": {"accuracy": 0.83}}}
        calibration = None
        last_margin = None

        def metadata(self):
            return {"arch": "fake", "value_scaling": "none"}

        def predict_with_margin(self, t1, t2):
            shape = t1.array.shape[1:]
            blank = np.zeros(shape, dtype=np.int64)
            return blank, blank.copy(), 0.95

    # Multispectral, so the index producer can supply a second opinion and
    # will find change where the model found none.
    rng = np.random.default_rng(0)
    names = ["red", "green", "blue", "nir", "swir"]

    def frame(nir_level):
        array = rng.uniform(0.05, 0.15, (5, 32, 32)).astype(np.float32)
        array[3, :16] = nir_level      # vegetation on the top half at t1 only
        return RSImage(array=array, crs=None, transform=None,
                       modality="optical", band_names=names, gsd_m=None)

    pipeline = BiTemporalPipeline(PipelineConfig(), segmenter=_BlindSegmenter())
    result = pipeline.run(frame(0.9), frame(0.05),
                          "What is the percentage of changed areas?")
    extras = result.evidence_extras

    assert extras["cross_check_contradiction"] is True
    assert result.confidence == 0.0, "a contradicted model must not sound calibrated"
    basis = extras["confidence_basis"].lower()
    assert "outside the distribution" in basis
