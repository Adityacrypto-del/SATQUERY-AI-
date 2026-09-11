"""Deterministic tests for the final reasoner's response parsing.

Tests ``_parse_final_response`` directly without model inference.
"""

from modules.bi_temporal.schemas import ChangeEvidence
from modules.fusion.schemas import FusedEvidence
from modules.qwen.reasoner import _parse_final_response
from modules.qwen.schemas import QwenAnalysis


def _make_fused(specialist_available=True, specialist_confidence=0.7):
    qwen = QwenAnalysis(
        observation="Green area in T2",
        possible_changes=["vegetation increase"],
        confidence=0.6,
    )

    specialist = None
    if specialist_available:
        specialist = ChangeEvidence(
            changed=True,
            summary="NDVI increase detected",
            confidence=specialist_confidence,
            changed_area_pixels=5000,
        )

    return FusedEvidence(
        qwen_analysis=qwen,
        specialist_evidence=specialist,
        specialist_available=specialist_available,
        agreements=["Both detect change"],
        disagreements=["Confidence gap"],
        authoritative_measurements={
            "changed_pixels": 5000,
        },
    )


def test_parse_includes_answer():
    fused = _make_fused()
    resp = _parse_final_response("Vegetation increased.", fused)

    assert resp.answer == "Vegetation increased."


def test_parse_confidence_averages_both():
    fused = _make_fused(specialist_confidence=0.8)

    # qwen=0.6, specialist=0.8 → avg=0.7
    resp = _parse_final_response("answer", fused)

    assert resp.confidence == 0.7


def test_parse_confidence_qwen_only_when_no_specialist():
    fused = _make_fused(specialist_available=False)

    resp = _parse_final_response("answer", fused)

    assert resp.confidence == 0.6


def test_parse_carries_disagreements():
    fused = _make_fused()
    resp = _parse_final_response("answer", fused)

    assert "Confidence gap" in resp.disagreements


def test_parse_adds_limitation_when_no_specialist():
    fused = _make_fused(specialist_available=False)

    resp = _parse_final_response("answer", fused)

    assert any("not available" in l for l in resp.limitations)


def test_parse_supporting_evidence_includes_both():
    fused = _make_fused()
    resp = _parse_final_response("answer", fused)

    texts = " ".join(resp.supporting_evidence)

    assert "Qwen" in texts
    assert "Specialist" in texts or "NDVI" in texts
    assert "changed_pixels" in texts


def test_parse_preserves_raw_response():
    fused = _make_fused()
    raw = "This is the full raw model output."

    resp = _parse_final_response(raw, fused)

    assert resp.raw_response == raw
