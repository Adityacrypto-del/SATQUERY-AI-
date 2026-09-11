"""Deterministic tests for Evidence Fusion.

No model loading required. All tests use hand-constructed
QwenAnalysis and ChangeEvidence instances.
"""

from modules.bi_temporal.schemas import ChangeEvidence
from modules.fusion.fuse import (
    format_measurements,
    format_specialist_summary,
    fuse_evidence,
)
from modules.fusion.schemas import FusedEvidence
from modules.qwen.schemas import QwenAnalysis


def _make_qwen(changes=None, confidence=0.5):
    return QwenAnalysis(
        observation="test observation",
        possible_changes=changes or [],
        objects=["building"],
        confidence=confidence,
    )


def _make_specialist(changed=True, pixels=1000, area=4000.0, confidence=0.7):
    return ChangeEvidence(
        changed=changed,
        summary="test specialist summary",
        change_type="construction",
        confidence=confidence,
        changed_area_pixels=pixels,
        changed_area_m2=area,
    )


# ------------------------------------------------------------------
# FusedEvidence construction
# ------------------------------------------------------------------


def test_fused_evidence_defaults():
    fused = FusedEvidence(
        qwen_analysis=_make_qwen(),
    )

    assert fused.specialist_evidence is None
    assert fused.specialist_available is False
    assert fused.agreements == []
    assert fused.disagreements == []
    assert fused.authoritative_measurements == {}


# ------------------------------------------------------------------
# fuse_evidence — no specialist
# ------------------------------------------------------------------


def test_fuse_without_specialist():
    qwen = _make_qwen(changes=["new building"])
    fused = fuse_evidence(qwen, specialist=None)

    assert fused.qwen_analysis is qwen
    assert fused.specialist_available is False
    assert fused.specialist_evidence is None
    assert fused.authoritative_measurements == {}


# ------------------------------------------------------------------
# fuse_evidence — with specialist
# ------------------------------------------------------------------


def test_fuse_both_detect_change():
    qwen = _make_qwen(changes=["new road"])
    spec = _make_specialist(changed=True)

    fused = fuse_evidence(qwen, spec)

    assert fused.specialist_available is True
    assert any("Both" in a for a in fused.agreements)
    assert fused.authoritative_measurements["changed_pixels"] == 1000
    assert fused.authoritative_measurements["changed_area_m2"] == 4000.0


def test_fuse_both_no_change():
    qwen = _make_qwen(changes=[])
    spec = _make_specialist(changed=False, pixels=0, area=0.0)

    fused = fuse_evidence(qwen, spec)

    assert any("no change" in a.lower() for a in fused.agreements)


def test_fuse_qwen_change_specialist_no_change():
    qwen = _make_qwen(changes=["vegetation loss"])
    spec = _make_specialist(changed=False)

    fused = fuse_evidence(qwen, spec)

    assert any("Qwen reports" in d for d in fused.disagreements)


def test_fuse_specialist_change_qwen_no_change():
    qwen = _make_qwen(changes=[])
    spec = _make_specialist(changed=True)

    fused = fuse_evidence(qwen, spec)

    assert any("Specialist detected" in d for d in fused.disagreements)


def test_fuse_confidence_divergence():
    qwen = _make_qwen(changes=["x"], confidence=0.3)
    spec = _make_specialist(changed=True, confidence=0.9)

    fused = fuse_evidence(qwen, spec)

    assert any("divergence" in d.lower() for d in fused.disagreements)


def test_fuse_confidence_agreement():
    qwen = _make_qwen(changes=["x"], confidence=0.6)
    spec = _make_specialist(changed=True, confidence=0.7)

    fused = fuse_evidence(qwen, spec)

    assert any("consistent" in a.lower() for a in fused.agreements)


def test_authoritative_measurements_include_regions():
    spec = ChangeEvidence(
        changed=True,
        summary="test",
        regions=[{"id": 1}, {"id": 2}],
    )

    fused = fuse_evidence(_make_qwen(), spec)

    assert fused.authoritative_measurements["region_count"] == 2


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------


def test_format_measurements_with_data():
    fused = fuse_evidence(
        _make_qwen(changes=["x"]),
        _make_specialist(),
    )

    text = format_measurements(fused)

    assert "Changed Pixels" in text
    assert "1000" in text


def test_format_measurements_without_specialist():
    fused = fuse_evidence(_make_qwen())

    text = format_measurements(fused)

    assert "No specialist" in text


def test_format_specialist_summary_available():
    fused = fuse_evidence(
        _make_qwen(changes=["x"]),
        _make_specialist(),
    )

    text = format_specialist_summary(fused)

    assert "test specialist summary" in text


def test_format_specialist_summary_unavailable():
    fused = fuse_evidence(_make_qwen())

    text = format_specialist_summary(fused)

    assert "not available" in text
