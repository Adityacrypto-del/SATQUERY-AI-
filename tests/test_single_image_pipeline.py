from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from satquery.agent.single_image_router import classify_query
from satquery.models.captioning import generate_caption
from satquery.models.rs_vlm import RSVLM, ModelResult, _heuristic_confidence
from satquery.models.vqa import answer_vqa
from satquery.preprocessing.image_loader import load_image, validate_single_image
from satquery.preprocessing.multispectral import to_model_rgb


# ── Router tests ────────────────────────────────────────────────────────


def test_router_distinguishes_captioning_from_vqa():
    assert classify_query("Give me a detailed scene description.").task == "captioning"
    assert classify_query("How many buildings are visible?").task == "vqa"


def test_router_routes_describe_to_captioning():
    assert classify_query("Describe this image.").task == "captioning"


def test_router_routes_caption_keyword():
    assert classify_query("caption the scene").task == "captioning"


def test_router_routes_summary_to_captioning():
    assert classify_query("summarize the image").task == "captioning"


def test_router_routes_what_do_you_see_to_captioning():
    assert classify_query("what do you see here?").task == "captioning"


def test_router_default_is_vqa():
    assert classify_query("What is the land cover?").task == "vqa"
    assert classify_query("Is there water present?").task == "vqa"


def test_router_handles_empty_query():
    result = classify_query("")
    assert result.task == "vqa"
    assert result.rule == "default_vqa"


def test_router_handles_none_like_query():
    result = classify_query(None)
    assert result.task == "vqa"


# ── Validation tests ────────────────────────────────────────────────────


def test_validation_accepts_a_decodable_png(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)

    result = validate_single_image(str(image_path))

    assert result["ok"] is True
    assert result["width"] == 8
    assert result["height"] == 6

    loaded = load_image(str(image_path))
    _, trace = to_model_rgb(loaded.array, loaded.metadata)
    assert trace["band_selection"] == "declared_rgb_channels"


def test_validation_rejects_missing_file():
    result = validate_single_image("/nonexistent/file.png")
    assert result["ok"] is False
    assert result["reason"] == "file_not_found"


def test_validation_rejects_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("not an image")
    result = validate_single_image(str(bad_file))
    assert result["ok"] is False
    assert "unsupported_extension" in result["reason"]


def test_validation_rejects_corrupt_image(tmp_path: Path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    result = validate_single_image(str(corrupt))
    assert result["ok"] is False
    assert "decode_error" in result["reason"]


def test_validation_accepts_jpeg(tmp_path: Path):
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(img_path, format="JPEG")
    result = validate_single_image(str(img_path))
    assert result["ok"] is True
    assert result["extension"] == ".jpg"


def test_validation_accepts_jpeg_extension(tmp_path: Path):
    img_path = tmp_path / "photo.jpeg"
    Image.new("RGB", (16, 16)).save(img_path, format="JPEG")
    result = validate_single_image(str(img_path))
    assert result["ok"] is True
    assert result["extension"] == ".jpeg"


def test_validation_accepts_tiff(tmp_path: Path):
    img_path = tmp_path / "image.tiff"
    Image.new("RGB", (20, 20), color=(50, 100, 150)).save(img_path, format="TIFF")
    result = validate_single_image(str(img_path))
    assert result["ok"] is True
    assert result["extension"] == ".tiff"


# ── Multispectral preprocessing tests ──────────────────────────────────


def test_multispectral_trace_records_named_sentinel_rgb_selection():
    source = np.stack([
        np.full((4, 4), fill_value, dtype=np.uint16)
        for fill_value in range(1, 14)
    ], axis=-1)

    rgb, trace = to_model_rgb(
        source,
        {"band_descriptions": [f"B{i:02d}" for i in range(1, 14)]},
    )

    assert rgb.shape == (4, 4, 3)
    assert trace["bands_used"] == ["B04", "B03", "B02"]
    assert trace["band_selection"] == "named_sentinel_bands"
    assert trace["source_band_count"] == 13


def test_multispectral_first_three_bands_fallback():
    arr = np.random.randint(0, 255, (8, 8, 5), dtype=np.uint16)
    rgb, trace = to_model_rgb(arr, {})
    assert rgb.shape == (8, 8, 3)
    assert trace["band_selection"] == "first_three_bands_fallback"
    assert trace["bands_used"] == ["band_1", "band_2", "band_3"]


def test_multispectral_single_band_repeated():
    arr = np.random.randint(0, 255, (6, 6, 1), dtype=np.uint16)
    rgb, trace = to_model_rgb(arr, {})
    assert rgb.shape == (6, 6, 3)
    assert trace["band_selection"] == "single_band_repeated"


def test_multispectral_rejects_2d_array():
    arr = np.zeros((4, 4), dtype=np.float32)
    try:
        to_model_rgb(arr, {})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_multispectral_rgb_declared_channels():
    arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    meta = {"color_interpretations": ["red", "green", "blue"]}
    rgb, trace = to_model_rgb(arr, meta)
    assert trace["band_selection"] == "declared_rgb_channels"
    assert trace["bands_used"] == ["red", "green", "blue"]


def test_multispectral_percentile_normalization():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[:2, :, :] = 10.0
    arr[2:, :, :] = 200.0
    rgb, _ = to_model_rgb(arr, {})
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


# ── Heuristic confidence tests ─────────────────────────────────────────


def test_heuristic_confidence_empty_text():
    conf, method = _heuristic_confidence("")
    assert conf == 0.0
    assert "empty" in method


def test_heuristic_confidence_short_text():
    conf, method = _heuristic_confidence("OK")
    assert conf == 0.2
    assert "short" in method


def test_heuristic_confidence_hedging():
    conf, method = _heuristic_confidence("This is maybe unclear and not sure")
    assert conf == 0.35
    assert "hedging" in method


def test_heuristic_confidence_lengthy_text():
    long_text = "This is a satellite image showing urban area with buildings and roads " * 2
    conf, method = _heuristic_confidence(long_text)
    assert conf == 0.7
    assert "lengthy" in method


def test_heuristic_confidence_default():
    conf, method = _heuristic_confidence("The image shows a forest.")
    assert conf == 0.55
    assert "default" in method


# ── VQA pipeline tests (with mock model) ────────────────────────────────


def test_answer_vqa_returns_structured_output():
    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.answer.return_value = ModelResult(
        text="forest", confidence=0.6, confidence_method="heuristic_default"
    )
    image = Image.new("RGB", (64, 64), color=(0, 128, 0))

    result = answer_vqa(mock_model, image, "What land cover is present?", "/fake/path.jpg")

    assert result["task"] == "vqa"
    assert result["answer"] == "forest"
    assert result["model"] == "BLIP-2 (test)"
    assert result["confidence"] == 0.6
    assert result["confidence_method"] == "heuristic_default"
    assert result["evidence"]["image"] == "/fake/path.jpg"
    mock_model.answer.assert_called_once_with("What land cover is present?", image=image)


def test_answer_vqa_no_image():
    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.answer.return_value = ModelResult(
        text="[no image provided]", confidence=None, confidence_method="not_available"
    )

    result = answer_vqa(mock_model, None, "What is this?", "")

    assert result["task"] == "vqa"
    assert "no image" in result["answer"]


# ── Captioning pipeline tests (with mock model) ────────────────────────


def test_generate_caption_returns_structured_output():
    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.caption.return_value = ModelResult(
        text="An urban area with buildings and roads.",
        confidence=0.7,
        confidence_method="heuristic_lengthy",
    )
    image = Image.new("RGB", (64, 64), color=(100, 100, 100))

    result = generate_caption(mock_model, image, "/fake/satellite.tif")

    assert result["task"] == "captioning"
    assert "urban area" in result["caption"]
    assert result["model"] == "BLIP-2 (test)"
    assert result["confidence"] == 0.7
    assert result["evidence"]["image"] == "/fake/satellite.tif"
    mock_model.caption.assert_called_once_with(image=image)


def test_generate_caption_no_image():
    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.caption.return_value = ModelResult(text="[no image provided]")

    result = generate_caption(mock_model, None, "")

    assert result["task"] == "captioning"
    assert "no image" in result["caption"]


# ── Full pipeline integration test (with mocked model loading) ──────────


def test_analyze_single_image_vqa_flow(tmp_path: Path):
    from satquery.single_image_api import analyze_single_image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32), color=(0, 100, 200)).save(img_path)

    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.adapted = False
    mock_model.answer.return_value = ModelResult(
        text="water", confidence=0.55, confidence_method="heuristic_default"
    )

    with patch("satquery.single_image_api._get_model", return_value=mock_model):
        result = analyze_single_image(str(img_path), "What is the dominant feature?")

    assert result["task"] == "vqa"
    assert result["answer"] == "water"
    assert result["adapted"] is False
    assert "execution_trace" in result
    assert len(result["execution_trace"]) >= 3


def test_analyze_single_image_captioning_flow(tmp_path: Path):
    from satquery.single_image_api import analyze_single_image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32), color=(0, 100, 200)).save(img_path)

    mock_model = MagicMock(spec=RSVLM)
    mock_model.model_name = "BLIP-2 (test)"
    mock_model.adapted = False
    mock_model.caption.return_value = ModelResult(
        text="A satellite image showing water body.",
        confidence=0.55,
        confidence_method="heuristic_default",
    )

    with patch("satquery.single_image_api._get_model", return_value=mock_model):
        result = analyze_single_image(str(img_path), "Describe this image.")

    assert result["task"] == "captioning"
    assert "water body" in result["caption"]
    assert result["adapted"] is False


def test_analyze_single_image_rejects_missing_file():
    from satquery.single_image_api import analyze_single_image

    result = analyze_single_image("/nonexistent/file.png", "What is this?")

    assert result["task"] == "invalid"
    assert "execution_trace" in result
