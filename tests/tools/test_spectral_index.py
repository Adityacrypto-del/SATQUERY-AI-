"""Track C: spectral-index fallback tool + router extension.

The tool is a physics-based computation over real spectral bands. It must
never answer from an RGB render, never guess band positions, and never pass
itself off as a VLM answer. Index maths is imported from the bi-temporal
branch (``tools.change_analysis.indices``), not re-implemented here.

The bi-temporal checkout is located through ``SATQUERY_BITEMPORAL_ROOT``.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from satquery.agent.single_image_router import SPECTRAL_PATTERNS, classify_query

rasterio = pytest.importorskip("rasterio")

requires_bitemporal = pytest.mark.skipif(
    not os.environ.get("SATQUERY_BITEMPORAL_ROOT"),
    reason="SATQUERY_BITEMPORAL_ROOT not set; the spectral tool imports indices.py from the bi-temporal checkout",
)

NIR, RED = 0.5, 0.1  # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 2/3


def _write_geotiff(path: Path, bands: dict[str, np.ndarray], named: bool = True) -> Path:
    first = next(iter(bands.values()))
    with rasterio.open(
        path, "w", driver="GTiff", height=first.shape[0], width=first.shape[1],
        count=len(bands), dtype="float32",
    ) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype("float32"), i)
            if named:
                dst.set_band_description(i, name)
    return path


def _s2_four_band(tmp_path: Path, named: bool = True, zero_pixel: bool = False) -> Path:
    shape = (8, 8)
    red = np.full(shape, RED)
    nir = np.full(shape, NIR)
    if zero_pixel:
        red[0, 0] = nir[0, 0] = 0.0  # undefined NDVI -> must be excluded, not counted
    return _write_geotiff(
        tmp_path / "s2.tif",
        {"B02": np.full(shape, 0.05), "B03": np.full(shape, 0.08), "B04": red, "B08": nir},
        named=named,
    )


# -- router ------------------------------------------------------------------

@pytest.mark.parametrize("pattern,index", [(p, i) for i, ps in SPECTRAL_PATTERNS.items() for p in ps])
def test_router_sends_every_listed_spectral_pattern_to_the_tool(pattern, index):
    route = classify_query(f"What is the {pattern} of this area?")
    assert route.task == "spectral_index"
    assert route.index == index
    assert pattern in route.rule


def test_router_spectral_request_beats_caption_keyword():
    assert classify_query("Describe the NDVI of this scene").task == "spectral_index"


def test_router_leaves_plain_visual_questions_on_the_vlm():
    for q in ("Is there water present?", "How much vegetation is visible?", "Describe this image."):
        assert classify_query(q).task in {"vqa", "captioning"}
        assert classify_query(q).index is None


# -- tool --------------------------------------------------------------------

@requires_bitemporal
def test_ndvi_on_named_multispectral_is_computed_and_labelled_physics_based(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    res = compute_spectral_index(str(_s2_four_band(tmp_path)), index="ndvi", modality="optical")
    assert res.status == "ok"
    assert res.statistics["mean"] == pytest.approx(2 / 3, abs=1e-5)
    assert res.trace["is_vlm_answer"] is False
    assert "physics-based" in res.trace["method"]
    assert res.trace["bands_used"] == {"nir": "B08", "red": "B04"}


@requires_bitemporal
def test_rgb_png_is_refused_not_fabricated(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    png = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(png)
    res = compute_spectral_index(str(png), index="ndvi", modality="optical")
    assert res.status == "refused"
    assert res.statistics is None
    assert "NIR" in res.reason


@requires_bitemporal
def test_unnamed_multiband_is_refused_rather_than_guessing_band_order(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    res = compute_spectral_index(str(_s2_four_band(tmp_path, named=False)), index="ndvi", modality="optical")
    assert res.status == "refused"
    assert res.statistics is None


@requires_bitemporal
def test_sar_input_is_refused(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    res = compute_spectral_index(str(_s2_four_band(tmp_path)), index="ndvi", modality="sar")
    assert res.status == "refused"
    assert "SAR" in res.reason


@requires_bitemporal
def test_missing_swir_refuses_ndbi_but_ndvi_still_works(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    path = str(_s2_four_band(tmp_path))
    assert compute_spectral_index(path, index="ndvi", modality="optical").status == "ok"
    ndbi = compute_spectral_index(path, index="ndbi", modality="optical")
    assert ndbi.status == "refused"
    assert "SWIR" in ndbi.reason


@requires_bitemporal
def test_undefined_pixels_are_excluded_and_reported(tmp_path):
    from satquery.tools.spectral_index import compute_spectral_index

    res = compute_spectral_index(str(_s2_four_band(tmp_path, zero_pixel=True)), index="ndvi", modality="optical")
    assert res.status == "ok"
    assert res.statistics["valid_fraction"] == pytest.approx(63 / 64)
    assert np.isfinite(res.statistics["mean"])


@requires_bitemporal
def test_index_maths_is_imported_from_bitemporal_not_forked():
    import satquery.tools.spectral_index as tool
    from tools.change_analysis import indices

    assert tool.indices is indices


@requires_bitemporal
def test_api_routes_spectral_query_to_tool_without_loading_the_vlm(tmp_path):
    from satquery.single_image_api import analyze_single_image

    png = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(png)
    with patch("satquery.single_image_api._get_model") as get_model:
        out = analyze_single_image(str(png), "What is the NDVI here?")
    get_model.assert_not_called()
    assert out["task"] == "spectral_index"
    assert out["status"] == "refused"
    assert any("spectral_index" in step for step in out["execution_trace"])
