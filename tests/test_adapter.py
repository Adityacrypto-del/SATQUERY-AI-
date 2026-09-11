"""Tests for the updated DeltaVLM adapter.

Validates graceful degradation on non-CUDA environments.
"""

import pytest

from modules.bi_temporal.adapter import DeltaVLMAdapter


def test_adapter_creation():
    adapter = DeltaVLMAdapter()

    assert adapter.model is None
    assert adapter.model_path is None


def test_adapter_creation_with_path():
    adapter = DeltaVLMAdapter(model_path="/path/to/checkpoint")

    assert adapter.model_path == "/path/to/checkpoint"


def test_is_available_is_bool():
    adapter = DeltaVLMAdapter()

    # On Apple Silicon / no CUDA, this should be False.
    assert isinstance(adapter.is_available, bool)


def test_is_available_caches_result():
    adapter = DeltaVLMAdapter()

    first = adapter.is_available
    second = adapter.is_available

    assert first is second


def test_load_model_raises_on_non_cuda():
    adapter = DeltaVLMAdapter()

    if adapter.is_available:
        pytest.skip("CUDA is available; cannot test graceful failure.")

    with pytest.raises(RuntimeError):
        adapter.load_model()


def test_analyze_raises_on_non_cuda():
    adapter = DeltaVLMAdapter()

    if adapter.is_available:
        pytest.skip("CUDA is available; cannot test graceful failure.")

    with pytest.raises(RuntimeError):
        adapter.analyze("t1", "t2", "What changed?")


def test_analyze_raises_when_model_not_loaded():
    adapter = DeltaVLMAdapter()

    # Force _available = True to test the "model not loaded" path.
    adapter._available = True

    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.analyze("t1", "t2", "What changed?")
