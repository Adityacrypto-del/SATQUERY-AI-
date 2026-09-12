"""The HTTP contract must never return invented analysis.

The old server answered with hardcoded text ("approximately 48 distinct industrial buildings"),
confidence 0.942, invented land_use percentages and a fake CRS whenever the model was missing or
raised, and /api/analyze-preset never called a model at all. These tests pin that shut.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.backend_server as server

REAL = {
    "task": "vqa",
    "answer": "two buildings",
    "status": "ok",
    "model": "TerraQ-VL stage-2 checkpoint-2180 (third-party, VRSBench-adapted)",
    "confidence": 0.55,
    "confidence_method": "heuristic_default",
    "execution_trace": ["input_validation_ok"],
}


@pytest.fixture
def client():
    return TestClient(server.app)


def _png_bytes():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 3), dtype="uint8")).save(buf, format="PNG")
    return buf.getvalue()


def _post(client, **kw):
    return client.post("/api/analyze", files={"image": ("scene.png", _png_bytes(), "image/png")},
                       data={"query": "how many buildings?"}, **kw)


def test_the_mock_fallback_is_gone():
    assert not hasattr(server, "_mock_analysis_fallback")


def test_analyze_returns_pipeline_output_unembellished(client, monkeypatch):
    monkeypatch.setattr(server, "analyze", lambda *a, **k: dict(REAL))
    body = client.post("/api/analyze", files={"image": ("scene.png", _png_bytes(), "image/png")},
                       data={"query": "how many buildings?"}).json()
    assert body["answer"] == "two buildings"
    assert "land_use" not in body                      # no injected chart data
    assert body["confidence"] == 0.55                  # not overwritten with 0.95
    assert body["confidence_method"] == "heuristic_default"


def test_analyze_failure_is_an_error_not_a_fabrication(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("weights missing")
    monkeypatch.setattr(server, "analyze", boom)
    r = client.post("/api/analyze", files={"image": ("scene.png", _png_bytes(), "image/png")},
                    data={"query": "how many buildings?"})
    assert r.status_code == 503
    body = r.json()
    text = str(body).lower()
    assert "weights missing" in text
    for invented in ("48 distinct", "0.942", "94.2", "land_use"):
        assert invented not in text


def test_analyze_without_pipeline_available_is_503(client, monkeypatch):
    monkeypatch.setattr(server, "analyze", None)
    r = client.post("/api/analyze", files={"image": ("scene.png", _png_bytes(), "image/png")},
                    data={"query": "anything"})
    assert r.status_code == 503
    assert "answer" not in r.json()


def test_preset_endpoint_does_not_fabricate(client):
    r = client.post("/api/analyze-preset", data={"query": "describe", "preset_title": "Urban"})
    assert r.status_code in (501, 503)
    text = str(r.json()).lower()
    assert "answer" not in r.json()
    assert "48 distinct" not in text and "confidence" not in r.json()


def test_presets_carry_no_invented_land_use(client):
    for preset in client.get("/api/presets").json()["presets"]:
        assert "default_land_use" not in preset


def test_status_does_not_claim_an_adapter_that_does_not_exist(client):
    body = client.get("/api/status").json()
    assert body["adapted"] is False
    assert "lora ready" not in body["model_name"].lower()
    assert "backend_available" in body
    assert body["branch"] == "peek/branch1-baseline"
