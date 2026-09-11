"""Integration tests for Qwen VLM inference.

These tests require:
- The ``qwenvl`` conda environment (or equivalent)
- Qwen2.5-VL-3B-Instruct cached locally
- torch, transformers, qwen-vl-utils installed

Tests are skipped automatically when the model is not available.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# ------------------------------------------------------------------
# Skip guard: only run when model dependencies are available.
# ------------------------------------------------------------------

_SKIP_REASON = None

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    _SKIP_REASON = "torch or transformers not installed"

try:
    from PIL import Image
except ImportError:
    _SKIP_REASON = "Pillow not installed"

requires_model = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=_SKIP_REASON or "model dependencies unavailable",
)

# Mark all tests in this module as requiring model dependencies.
pytestmark = [
    requires_model,
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _create_synthetic_image(
    width: int = 64,
    height: int = 64,
    color: tuple = (100, 150, 100),
) -> str:
    """Create a synthetic RGB image and return its path.

    The file is written to a temporary directory and the path
    is returned. Caller should clean up or use a fixture.
    """
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    img.save(path)
    return path


@pytest.fixture
def qwen_model():
    """Load the Qwen model once per test session."""
    from modules.qwen.model import QwenVL

    model = QwenVL()

    try:
        model.load()
    except Exception as exc:
        pytest.skip(f"Could not load Qwen model: {exc}")

    return model


@pytest.fixture
def green_image():
    """Synthetic green vegetation-like image."""
    path = _create_synthetic_image(color=(50, 150, 50))
    yield path
    os.unlink(path)


@pytest.fixture
def brown_image():
    """Synthetic brown/bare-soil image."""
    path = _create_synthetic_image(color=(160, 120, 80))
    yield path
    os.unlink(path)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.slow
def test_single_image_returns_qwen_analysis(qwen_model, green_image):
    """Verify single-image inference returns structured output."""
    from modules.qwen.inference import analyze_single

    result = analyze_single(
        model=qwen_model,
        image_path=green_image,
        query="What do you see in this image?",
    )

    assert result.observation
    assert isinstance(result.observation, str)
    assert len(result.observation) > 10
    assert result.model_name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert result.latency_s is not None
    assert result.latency_s > 0


@pytest.mark.slow
def test_bitemporal_returns_qwen_analysis(
    qwen_model,
    green_image,
    brown_image,
):
    """Verify bi-temporal inference returns structured output."""
    from modules.qwen.inference import analyze_bitemporal

    result = analyze_bitemporal(
        model=qwen_model,
        t1_path=green_image,
        t2_path=brown_image,
        query="What changed between these two images?",
    )

    assert result.observation
    assert isinstance(result.observation, str)
    assert len(result.observation) > 10
    assert result.latency_s is not None
    assert result.latency_s > 0


@pytest.mark.slow
def test_confidence_is_extracted(qwen_model, green_image):
    """Verify confidence extraction from model output."""
    from modules.qwen.inference import analyze_single

    result = analyze_single(
        model=qwen_model,
        image_path=green_image,
        query="Describe this satellite image. State your confidence.",
    )

    # Confidence should be a reasonable float.
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_model_not_loaded_raises():
    """Verify error when calling generate before load."""
    from modules.qwen.model import QwenVL

    model = QwenVL()

    with pytest.raises(RuntimeError, match="not loaded"):
        model.generate([{"role": "user", "content": "test"}])
