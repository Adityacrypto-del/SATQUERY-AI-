"""Tests for the branch-2 handoff contract.

The controller consumes one object. What it can see there is the whole of
what this branch delivers, so the tests that matter are about substitution
(nothing breaks) and completeness (nothing the problem statement grades is
dropped at the boundary).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from modules.bi_temporal.schemas import ChangeEvidence
from tools.change_analysis.contract import BiTemporalEvidence
from tools.change_analysis.result import ChangeResult


def _result(**overrides):
    base = dict(
        answer="yes",
        trace=[{"stage": 1, "name": "validate_pair", "tool": "t", "params": {},
                "observation": "ok", "duration_ms": 1.0, "why": "because"}],
        changed=True,
        summary="something changed",
        change_type="buildings",
        confidence=0.84,
        changed_area_pixels=1234,
        changed_area_m2=None,
        regions=[{"label": 1, "pixel_count": 10}],
        semantic_t1=np.zeros((4, 4), dtype=np.int64),
        semantic_t2=np.zeros((4, 4), dtype=np.int64),
        georeferencing={"georeferenced": False},
    )
    base.update(overrides)
    return ChangeResult(**base)


# --------------------------------------------------------------------------
# Substitution: the extension must not break the shared contract
# --------------------------------------------------------------------------


def test_is_substitutable_for_the_shared_contract():
    evidence = BiTemporalEvidence(changed=True, summary="x")

    assert isinstance(evidence, ChangeEvidence)


def test_every_added_field_is_defaulted():
    """So the subclass constructs exactly where the parent does. If one were
    required, any existing caller building a ChangeEvidence-shaped object
    would break."""
    shared = {f.name for f in dataclasses.fields(ChangeEvidence)}
    added = [f for f in dataclasses.fields(BiTemporalEvidence)
             if f.name not in shared]

    assert added, "the extension must actually add something"
    for f in added:
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        assert has_default, f"{f.name} must be defaulted"


def test_the_shared_schema_file_is_untouched():
    """HARD RULE 2: schemas.py is a shared contract. The extension exists
    precisely so that file does not need editing."""
    shared = {f.name for f in dataclasses.fields(ChangeEvidence)}

    assert shared == {
        "changed", "summary", "change_type", "confidence",
        "changed_area_pixels", "changed_area_m2", "regions",
    }


# --------------------------------------------------------------------------
# Completeness: what the PS grades must survive the handoff
# --------------------------------------------------------------------------


def test_the_trace_survives_the_handoff():
    """The problem statement grades the observable execution trace. Under the
    bare shared schema it was dropped here."""
    evidence = _result().to_change_evidence()

    assert isinstance(evidence, ChangeEvidence)
    assert evidence.trace and evidence.trace[0]["why"] == "because"


def test_the_answer_token_and_question_type_survive():
    evidence = _result().to_change_evidence()

    assert evidence.answer == "yes"


def test_confidence_basis_survives_so_a_zero_is_readable():
    """A bare 0.0 cannot be told apart from "very unsure". The basis is what
    makes measured-or-zero mean anything to the controller."""
    result = _result(confidence=0.0)
    result.evidence_extras = {"confidence_basis": "none: no measured basis"}

    evidence = result.to_change_evidence()

    assert evidence.confidence == 0.0
    assert "none" in evidence.confidence_basis


def test_route_and_overlays_survive():
    result = _result()
    result.evidence_extras = {
        "route": "semantic",
        "overlays": {"mask": "outputs/smoke/overlays/a.png"},
        "question_type": "change_or_not",
    }

    evidence = result.to_change_evidence()

    assert evidence.route == "semantic"
    assert evidence.overlays["mask"].endswith(".png")
    assert evidence.question_type == "change_or_not"


def test_compound_answers_survive():
    result = _result(answer=None)
    result.evidence_extras = {
        "answer_evidence": {
            "compound": True,
            "per_class": {"low_vegetation": "buildings", "water": None},
        }
    }

    evidence = result.to_change_evidence()

    assert evidence.answer is None
    assert evidence.compound_answers == {
        "low_vegetation": "buildings", "water": None
    }


# --------------------------------------------------------------------------
# The strict path stays available
# --------------------------------------------------------------------------


def test_strict_mode_returns_exactly_the_shared_shape():
    """For a consumer that reconstructs the dataclass from its fields and
    would choke on extras."""
    evidence = _result().to_change_evidence(strict=True)

    assert type(evidence) is ChangeEvidence
    assert not hasattr(evidence, "trace")


def test_no_placeholder_values_are_introduced():
    """An absent field stays absent. The extension must not fill gaps with
    plausible defaults that read as measurements."""
    evidence = _result(changed_area_m2=None, change_type=None).to_change_evidence()

    assert evidence.changed_area_m2 is None
    assert evidence.change_type is None
    assert evidence.overlays == {}
