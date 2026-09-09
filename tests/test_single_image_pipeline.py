from pathlib import Path

import numpy as np
from PIL import Image

from satquery.agent.single_image_router import classify_query
from satquery.preprocessing.image_loader import load_image, validate_single_image
from satquery.preprocessing.multispectral import to_model_rgb


def test_router_distinguishes_captioning_from_vqa():
    assert classify_query("Give me a detailed scene description.").task == "captioning"
    assert classify_query("How many buildings are visible?").task == "vqa"


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
