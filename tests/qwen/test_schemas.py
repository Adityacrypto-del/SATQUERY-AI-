"""Deterministic tests for Qwen schemas.

These tests validate dataclass construction and field defaults.
No model loading or inference is required.
"""

from modules.qwen.schemas import FinalResponse, QwenAnalysis


def test_qwen_analysis_defaults():
    analysis = QwenAnalysis(observation="test observation")

    assert analysis.observation == "test observation"
    assert analysis.possible_changes == []
    assert analysis.objects == []
    assert analysis.confidence == 0.0
    assert analysis.raw_response == ""
    assert analysis.model_name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert analysis.latency_s is None


def test_qwen_analysis_with_all_fields():
    analysis = QwenAnalysis(
        observation="urban area with buildings",
        possible_changes=["new construction", "road widening"],
        objects=["buildings", "roads", "vegetation"],
        confidence=0.85,
        raw_response="raw model output here",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        latency_s=2.5,
    )

    assert analysis.observation == "urban area with buildings"
    assert len(analysis.possible_changes) == 2
    assert len(analysis.objects) == 3
    assert analysis.confidence == 0.85
    assert analysis.latency_s == 2.5


def test_qwen_analysis_confidence_range():
    for conf in [0.0, 0.5, 1.0]:
        analysis = QwenAnalysis(
            observation="test",
            confidence=conf,
        )
        assert 0.0 <= analysis.confidence <= 1.0


def test_final_response_defaults():
    response = FinalResponse(answer="No change detected.")

    assert response.answer == "No change detected."
    assert response.confidence == 0.0
    assert response.supporting_evidence == []
    assert response.disagreements == []
    assert response.limitations == []
    assert response.raw_response == ""


def test_final_response_with_all_fields():
    response = FinalResponse(
        answer="Vegetation increased by 15%.",
        confidence=0.75,
        supporting_evidence=[
            "NDVI increased in region A",
            "Specialist: +11940 changed pixels",
        ],
        disagreements=[
            "Qwen estimated 20% but specialist measured 15%",
        ],
        limitations=[
            "DeltaVLM not available on this device",
        ],
        raw_response="full model output",
    )

    assert response.answer == "Vegetation increased by 15%."
    assert len(response.supporting_evidence) == 2
    assert len(response.disagreements) == 1
    assert len(response.limitations) == 1
