"""Tests for the staged orchestrator and the template description."""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from tools.change_analysis.describe import describe_change, describe_transitions
from tools.change_analysis.io import RSImage
from tools.change_analysis.pipeline import (
    BiTemporalPipeline,
    PipelineConfig,
    export_report,
)
from tools.change_analysis.regions import Region, extract_regions, summarise_regions

TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
CRS_UTM = CRS.from_epsg(32643)


def _pair(georeferenced=True, bands=3, size=64, shift=0, modality="optical"):
    """A pair whose second date has a bright square inserted."""
    rng = np.random.default_rng(0)
    base = rng.uniform(20, 60, (bands, size, size)).astype(np.float32)
    a = base.copy()
    b = base.copy()
    b[:, 20 + shift:40 + shift, 20:40] = 200.0
    kwargs = dict(
        crs=CRS_UTM if georeferenced else None,
        transform=TRANSFORM if georeferenced else None,
        modality=modality,
        band_names=["red", "green", "blue"][:bands] if bands == 3 else None,
        gsd_m=10.0 if georeferenced else None,
        _pixel_size_m=(10.0, 10.0) if georeferenced else None,
        _gsd_source="projected_crs_exact" if georeferenced else "unavailable",
    )
    return RSImage(array=a, **kwargs), RSImage(array=b, **kwargs)


class _FakeSegmenter:
    """Stands in for a trained checkpoint: change where t2 is bright."""

    arch = "shared"
    val_report = {
        "average_accuracy": 0.68,
        "per_type": {"change_or_not": {"accuracy": 0.845}},
    }

    def predict(self, t1, t2):
        changed = t2.array[0] > 120.0
        s1 = np.where(changed, 2, 0).astype(np.int64)   # low_vegetation
        s2 = np.where(changed, 4, 0).astype(np.int64)   # buildings
        return s1, s2

    def metadata(self):
        return {"arch": self.arch, "value_scaling": "assumed_0_255",
                "checkpoint": "fake.pt", "epoch": 1,
                "val_average_accuracy": 0.68, "band_selection": "declared_rgb_bands"}


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


def test_every_stage_records_the_required_trace_fields():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "Have the areas of buildings changed?"
    )

    assert result.trace
    for entry in result.trace:
        for key in ("stage", "name", "tool", "params", "observation",
                    "duration_ms", "why"):
            assert key in entry, f"trace entry missing {key}"
        assert isinstance(entry["duration_ms"], float)
        assert entry["why"], "every stage must record why it ran"


def test_stages_are_numbered_in_order():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(t1, t2)

    assert [e["stage"] for e in result.trace] == list(
        range(1, len(result.trace) + 1)
    )


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_rgb_with_checkpoint_routes_to_semantic():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(t1, t2)

    assert result.evidence_extras["route"] == "semantic"
    assert result.semantic_t1 is not None


def test_no_checkpoint_routes_to_the_index_detector():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig()).run(t1, t2)

    assert result.evidence_extras["route"] == "index"
    assert result.semantic_t1 is None
    names = [e["name"] for e in result.trace]
    assert "index_detection" in names


def test_sar_never_routes_to_segmentation():
    t1, t2 = _pair(bands=1, modality="sar")
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(t1, t2)

    assert result.evidence_extras["route"] == "index"


# --------------------------------------------------------------------------
# Answering and honesty
# --------------------------------------------------------------------------


def test_cdvqa_question_gets_a_rule_answer():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "Have the areas of buildings changed?"
    )

    assert result.answer == "yes"
    assert result.evidence_extras["question_type"] == "change_or_not"


def test_open_ended_query_gets_a_description_not_a_fabricated_label():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "What changed and where?"
    )

    assert result.answer is None
    assert result.evidence_extras["question_type"] is None
    assert "region" in result.summary.lower()


def test_absent_class_is_answered_honestly():
    """Asking about a class the scene does not contain must not invent one."""
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "Have the areas of water changed?"
    )

    assert result.answer == "no"


def test_ungeoreferenced_input_reports_no_area_in_metres():
    t1, t2 = _pair(georeferenced=False)
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(t1, t2)

    assert result.changed_area_m2 is None
    assert result.changed_area_pixels is not None
    assert "m2" not in result.summary


def test_failed_validation_stops_and_explains():
    t1, _ = _pair()
    _, t2 = _pair(size=32)
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(t1, t2)

    assert result.changed is False
    assert "rejected" in result.summary.lower()
    assert result.answer is None


# --------------------------------------------------------------------------
# Confidence must never be an invented number
# --------------------------------------------------------------------------


def test_confidence_uses_measured_per_type_accuracy():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "Have the areas of buildings changed?"
    )

    assert result.confidence == pytest.approx(0.845, abs=1e-6)
    assert "measured val accuracy" in result.evidence_extras["confidence_basis"]


def test_detector_route_reports_zero_confidence_with_an_explicit_basis():
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig()).run(t1, t2, "Did buildings increase?")

    assert result.confidence == 0.0
    assert result.evidence_extras["confidence_basis"].startswith("none")


# --------------------------------------------------------------------------
# Export and overlays
# --------------------------------------------------------------------------


def test_report_export_round_trips(tmp_path):
    t1, t2 = _pair()
    result = BiTemporalPipeline(PipelineConfig(), segmenter=_FakeSegmenter()).run(
        t1, t2, "Have the areas of buildings changed?"
    )
    path = export_report(result, str(tmp_path / "r.json"))

    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["answer"] == "yes"
    assert payload["trace"]
    assert "semantic_t1" not in payload  # large arrays stay out


def test_overlays_are_written_when_requested(tmp_path):
    t1, t2 = _pair()
    config = PipelineConfig(overlay_dir=str(tmp_path / "ov"))
    result = BiTemporalPipeline(config, segmenter=_FakeSegmenter()).run(
        t1, t2, scene_id="demo"
    )

    overlays = result.evidence_extras["overlays"]
    assert "mask_overlay" in overlays
    for path in overlays.values():
        assert open(path, "rb").read(4) == b"\x89PNG"


# --------------------------------------------------------------------------
# Template description
# --------------------------------------------------------------------------


def _regions_for(image, mask):
    return extract_regions(mask, image, min_pixels=1), summarise_regions(
        extract_regions(mask, image, min_pixels=1), image
    )


def test_description_says_nothing_changed_when_nothing_did():
    image, _ = _pair()
    mask = np.zeros(image.shape, dtype=bool)
    regions, summary = _regions_for(image, mask)

    assert "no change" in describe_change(regions, summary, image).lower()


def test_description_uses_metres_when_georeferenced():
    image, _ = _pair()
    mask = np.zeros(image.shape, dtype=bool)
    mask[10:20, 10:20] = True
    regions, summary = _regions_for(image, mask)

    text = describe_change(regions, summary, image)
    assert "m2" in text or "km2" in text
    assert "N," in text or "S," in text  # a lat/lon location


def test_description_uses_pixels_when_ungeoreferenced():
    image, _ = _pair(georeferenced=False)
    mask = np.zeros(image.shape, dtype=bool)
    mask[10:20, 10:20] = True
    regions, summary = _regions_for(image, mask)

    text = describe_change(regions, summary, image)
    assert "pixel" in text
    assert "m2" not in text


def test_description_names_the_dominant_transition():
    image, _ = _pair()
    mask = np.zeros(image.shape, dtype=bool)
    mask[10:30, 10:30] = True
    regions, summary = _regions_for(image, mask)
    s1 = np.where(mask, 2, 0)
    s2 = np.where(mask, 4, 0)

    text = describe_change(regions, summary, image, s1, s2)
    assert "low vegetation became buildings" in text


def test_transitions_are_empty_without_class_maps():
    assert describe_transitions(None, None) == []
