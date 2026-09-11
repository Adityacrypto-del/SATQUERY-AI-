"""Tests for ChangeResult and its downgrade into the shared contract.

ChangeResult is ours and complete. ChangeEvidence in modules/bi_temporal/
schemas.py is Aditya's shared contract, which we never edit. Integration is
exactly one call: ``to_change_evidence()``.

The behaviour that matters here is forward-compatibility in both directions.
Today ChangeEvidence has no ``answer`` and no ``trace``, so those are dropped.
If Aditya later adds them, the same adapter must carry them through with no
edit on our side. Both directions are tested.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from tools.change_analysis import result as result_module
from tools.change_analysis.result import ChangeResult

SCHEMAS_MODULE = "modules.bi_temporal.schemas"


def _make_result(**overrides) -> ChangeResult:
    kwargs: Dict[str, Any] = dict(
        answer="yes",
        trace=[{"stage": 1, "name": "load", "tool": "io.load_rsimage"}],
        changed=True,
        summary="Buildings replaced low vegetation in the north-east.",
        change_type="buildings",
        confidence=0.82,
        changed_area_pixels=1250,
        changed_area_m2=125000.0,
        regions=[{"area_m2": 125000.0, "centroid_lonlat": [77.5, 13.0]}],
        semantic_t1=np.zeros((4, 4), dtype=np.int16),
        semantic_t2=np.ones((4, 4), dtype=np.int16),
        georeferencing={"gsd_source": "projected_crs_exact", "gsd_m": 10.0},
    )
    kwargs.update(overrides)
    return ChangeResult(**kwargs)


# --------------------------------------------------------------------------
# Downgrade into the shared contract
# --------------------------------------------------------------------------


def test_downgrade_maps_the_shared_fields():
    from modules.bi_temporal.schemas import ChangeEvidence

    evidence = _make_result().to_change_evidence()

    assert isinstance(evidence, ChangeEvidence)
    assert evidence.changed is True
    assert evidence.change_type == "buildings"
    assert evidence.confidence == pytest.approx(0.82)
    assert evidence.changed_area_pixels == 1250
    assert evidence.changed_area_m2 == pytest.approx(125000.0)
    assert evidence.regions == [
        {"area_m2": 125000.0, "centroid_lonlat": [77.5, 13.0]}
    ]


def test_strict_mode_drops_fields_absent_from_the_shared_schema_without_raising():
    """The shared ChangeEvidence declares no answer/trace. A consumer that
    wants exactly that shape asks for it, and the extra fields are dropped
    rather than forced in.

    This previously asserted the default did the dropping. The default now
    returns BiTemporalEvidence, which carries them -- dropping the trace at
    the boundary meant discarding the artefact the problem statement grades.
    The narrow path is still available and still must not raise.
    """
    from modules.bi_temporal.schemas import ChangeEvidence

    schema_fields = {f.name for f in dataclasses.fields(ChangeEvidence)}
    evidence = _make_result().to_change_evidence(strict=True)

    assert type(evidence) is ChangeEvidence
    for absent in ("answer", "trace"):
        if absent not in schema_fields:
            assert not hasattr(evidence, absent)


def test_the_default_handoff_carries_what_the_ps_grades():
    """The counterpart: by default nothing graded is lost at the boundary."""
    evidence = _make_result().to_change_evidence()

    assert evidence.trace, "the execution trace is a graded artefact"
    assert hasattr(evidence, "answer")


def test_extra_schema_fields_are_carried_through_when_they_appear(monkeypatch):
    """If Aditya adds answer/trace later, the adapter must need no edit here."""

    @dataclasses.dataclass
    class ExtendedChangeEvidence:
        changed: bool
        summary: str
        change_type: Optional[str] = None
        confidence: float = 0.0
        changed_area_pixels: Optional[int] = None
        changed_area_m2: Optional[float] = None
        regions: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
        answer: Optional[str] = None
        trace: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    stub = importlib.import_module(SCHEMAS_MODULE)
    monkeypatch.setattr(stub, "ChangeEvidence", ExtendedChangeEvidence)

    evidence = _make_result().to_change_evidence()

    assert evidence.answer == "yes"
    assert evidence.trace[0]["tool"] == "io.load_rsimage"


def test_semantic_maps_never_reach_the_shared_contract():
    """s_t1/s_t2 are large arrays and are not part of Aditya's schema."""
    evidence = _make_result().to_change_evidence()

    assert not hasattr(evidence, "semantic_t1")
    assert not hasattr(evidence, "semantic_t2")


def test_none_values_survive_the_downgrade():
    """HARD RULE 3: an uncomputed field stays None, never a placeholder number."""
    evidence = _make_result(
        change_type=None,
        changed_area_pixels=None,
        changed_area_m2=None,
    ).to_change_evidence()

    assert evidence.change_type is None
    assert evidence.changed_area_pixels is None
    assert evidence.changed_area_m2 is None


# --------------------------------------------------------------------------
# Independence from the shared contract
# --------------------------------------------------------------------------


def test_result_module_imports_without_the_shared_schema(monkeypatch):
    """The whole point of the lazy import: schemas.py moving must not break us.

    Setting the module to None in sys.modules makes ``import`` raise
    ImportError, so a top-level import in result.py would fail this reload.
    """
    monkeypatch.setitem(sys.modules, SCHEMAS_MODULE, None)

    reloaded = importlib.reload(result_module)
    instance = reloaded.ChangeResult(
        answer=None,
        trace=[],
        changed=False,
        summary="no change",
        change_type=None,
        confidence=0.0,
        changed_area_pixels=None,
        changed_area_m2=None,
        regions=[],
        semantic_t1=None,
        semantic_t2=None,
        georeferencing={},
    )

    assert instance.changed is False
    with pytest.raises(ImportError):
        instance.to_change_evidence()


def test_module_is_restored_after_the_reload_test():
    """Guards against the reload above leaking a broken module into later tests."""
    importlib.reload(result_module)

    assert result_module.ChangeResult is not None
    assert _make_result().to_change_evidence() is not None


# --------------------------------------------------------------------------
# Construction invariants
# --------------------------------------------------------------------------


def test_mutable_defaults_are_not_shared_between_instances():
    first = ChangeResult.empty(summary="a")
    second = ChangeResult.empty(summary="b")

    first.trace.append({"stage": 1})
    first.regions.append({"area_m2": 1.0})

    assert second.trace == []
    assert second.regions == []


def test_empty_result_reports_nothing_measured():
    """A result with no computation behind it must not imply measurements."""
    empty = ChangeResult.empty(summary="not analysed")

    assert empty.answer is None
    assert empty.changed is False
    assert empty.confidence == 0.0
    assert empty.changed_area_pixels is None
    assert empty.changed_area_m2 is None
    assert empty.semantic_t1 is None


def test_confidence_outside_zero_to_one_is_rejected():
    """Confidence is reported to the grader; an out-of-range value is a bug."""
    with pytest.raises(ValueError):
        _make_result(confidence=1.4)
