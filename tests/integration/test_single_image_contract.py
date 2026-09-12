"""Handover contract: SingleImageEvidence + SingleImageAdapter.

The adapter is what a controller calls for single-image evidence. It mirrors the shape a
specialist adapter is expected to have (`is_available`, `load_model`, `analyze`) and returns a
structured SingleImageEvidence, never a bare string and never fabricated numbers.

No model is loaded here: the VLM backend is stubbed. What is tested is the contract.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from satquery.contracts import SingleImageEvidence
from satquery.integration.adapter import SingleImageAdapter


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "scene.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype="uint8")).save(p)
    return str(p)


class StubBackend:
    """Stands in for the TerraQ-VL backend."""
    name = "StubVLM (test)"
    available = True

    def __init__(self):
        self.calls = []

    def is_available(self):
        return self.available

    def generate(self, image_path, prompt, max_new_tokens=256):
        self.calls.append((image_path, prompt))
        return "two grey buildings beside a road"


def test_evidence_is_serialisable_and_declares_its_source(png):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(png, "What buildings are visible?")
    assert isinstance(ev, SingleImageEvidence)
    d = ev.to_dict()
    assert d["status"] == "ok" and d["task"] == "vqa"
    assert d["answer"] == "two grey buildings beside a road"
    assert d["is_vlm_answer"] is True
    assert "StubVLM" in d["model"]


def test_captioning_query_routes_to_captioning(png):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(png, "Describe this satellite image.")
    assert ev.task == "captioning"


def test_spectral_query_never_reaches_the_vlm(png):
    backend = StubBackend()
    ev = SingleImageAdapter(backend=backend).analyze(png, "What is the NDVI here?")
    assert ev.task == "spectral_index"
    assert backend.calls == []           # the VLM was not consulted
    assert ev.is_vlm_answer is False


def test_rgb_input_refuses_spectral_index_with_a_reason(png):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(png, "What is the NDVI here?")
    assert ev.status == "refused"
    assert ev.answer is None
    assert ev.reason and "NIR" in ev.reason
    assert ev.measurements == {}          # nothing invented


def test_unavailable_backend_reports_unavailable_not_a_guess(png):
    backend = StubBackend()
    backend.available = False
    adapter = SingleImageAdapter(backend=backend)
    assert adapter.is_available is False
    ev = adapter.analyze(png, "How many buildings?")
    assert ev.status == "unavailable"
    assert ev.answer is None
    assert ev.reason


def test_confidence_is_labelled_heuristic_or_absent(png):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(png, "How many buildings?")
    if ev.confidence is not None:
        assert ev.confidence_method and ev.confidence_method.startswith("heuristic")


def test_no_fabricated_fields_anywhere(png):
    d = SingleImageAdapter(backend=StubBackend()).analyze(png, "How many buildings?").to_dict()
    assert "land_use" not in d            # the old server invented this
    assert d["measurements"] == {}        # only deterministic tools populate measurements
    assert d["source_metadata"]["filename"].endswith("scene.png")


def test_trace_and_limitations_are_populated(png):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(png, "How many buildings?")
    assert ev.execution_trace and all(isinstance(s, str) for s in ev.execution_trace)
    assert any("rgb" in l.lower() or "not measured" in l.lower() or "heuristic" in l.lower()
               for l in ev.limitations)


def test_missing_file_is_rejected(tmp_path):
    ev = SingleImageAdapter(backend=StubBackend()).analyze(str(tmp_path / "nope.png"), "describe")
    assert ev.status in {"error", "refused"}
    assert ev.answer is None
