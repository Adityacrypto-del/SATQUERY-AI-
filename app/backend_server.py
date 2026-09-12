"""FastAPI backend for the SatQuery AI frontend (Branch 1: single-image VQA / captioning).

Contract with the React app (it proxies /api -> localhost:8000):
    GET  /api/status          backend + checkpoint state, honestly reported
    GET  /api/presets         demo scene metadata (descriptive only, no analysis numbers)
    POST /api/analyze         image + query -> SingleImageEvidence as JSON
    POST /api/analyze-preset  501: preset imagery is remote, so there is nothing local to analyse

This layer holds NO analysis logic. It writes the upload to a temp file, calls
``satquery.integration.service.analyze``, and returns that dict unchanged.

It never invents a result. An earlier version answered with hardcoded text ("approximately 48
distinct industrial buildings"), a 0.942 confidence, invented land-use percentages and a fake CRS
whenever the model was missing or raised. A demo built on that shows judges fabricated analysis, so
it is gone: if the model cannot run, this returns 503 with the reason.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from satquery.integration.service import analyze, get_adapter
except ImportError:  # the pipeline is not importable in this environment
    analyze = None
    get_adapter = None

app = FastAPI(
    title="SatQuery AI Backend",
    description="Remote-sensing single-image VQA / captioning API (Branch 1)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(__file__).parent.parent / "outputs" / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ACCEPTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Demo scenes for the UI. Descriptive metadata only: no land-use percentages and no analysis
# numbers, because nothing here has been analysed. Analysis comes from /api/analyze on a real upload.
PRESETS = [
    {
        "id": "preset_urban_01",
        "title": "Urban Infrastructure & Port (GeoTIFF)",
        "category": "Urban Density",
        "format": "GeoTIFF / PNG",
        "resolution": "0.5m / px",
        "location": "Rotterdam Harbor, Netherlands",
        "image_url": "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "Count shipping containers and cargo vessels.",
            "Describe the overall land cover and industrial density.",
            "Identify transportation networks and docks.",
        ],
    },
    {
        "id": "preset_agri_02",
        "title": "Circular Pivot Agricultural Fields",
        "category": "Agriculture",
        "format": "Sentinel-2 L2A (10m)",
        "resolution": "10m / px",
        "location": "Kansas, United States",
        "image_url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "What is the NDVI of this area?",
            "Estimate percentage of active agricultural land.",
            "Describe this satellite scene in detail.",
        ],
    },
    {
        "id": "preset_coastal_03",
        "title": "Barrier Reef & Coastal Vegetation",
        "category": "Environment",
        "format": "PlanetScope (3m)",
        "resolution": "3m / px",
        "location": "Cairns, Australia",
        "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "Describe the water clarity and coastal shoreline.",
            "What is the NDWI of this scene?",
            "Are there visible vessels near the shore?",
        ],
    },
]


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Report what is actually loadable, including when nothing is."""
    base: Dict[str, Any] = {
        "status": "online",
        "branch": "peek/branch1-baseline",
        "supported_formats": sorted(ext.lstrip(".").upper() for ext in ACCEPTED),
        "timestamp": time.time(),
        # Branch 1 has no adapter checkpoint of its own yet; the VLM is a third-party model.
        "adapted": False,
        "measured_results": "outputs/reference_eval/eval_log.md",
    }
    if get_adapter is None:
        return {**base, "status": "degraded", "backend_available": False,
                "model_name": "unavailable", "pipeline_importable": False,
                "reason": "satquery.integration is not importable in this environment"}

    adapter = get_adapter()
    available = adapter.is_available
    return {
        **base,
        "pipeline_importable": True,
        "backend_available": available,
        "model_name": adapter.backend.name,
        "tasks": adapter.describe()["tasks"],
        "reason": None if available else adapter.unavailable_reason,
    }


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    """Demo scene metadata. Descriptive only: no analysis has been run on these."""
    return {"presets": PRESETS}


@app.post("/api/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    query: str = Form(...),
    lora_path: str = Form(""),  # accepted for frontend compatibility; unused, no adapter exists yet
) -> Dict[str, Any]:
    """Upload a satellite image (.png/.jpg/.tif/.tiff) with a query and get real evidence back."""
    if analyze is None:
        raise HTTPException(
            status_code=503,
            detail="analysis pipeline unavailable: satquery.integration is not importable",
        )
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="no image file provided")

    suffix = Path(image.filename).suffix.lower() or ".png"
    if suffix not in ACCEPTED:
        raise HTTPException(status_code=400,
                            detail=f"unsupported format {suffix!r}; accepted: {sorted(ACCEPTED)}")

    temp_file = TEMP_DIR / f"upload_{int(time.time() * 1000)}{suffix}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        result = analyze(str(temp_file), query)
    except Exception as exc:  # a bug or a missing backend: say so, do not invent an answer
        raise HTTPException(
            status_code=503,
            detail=f"analysis failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except OSError:
                pass

    # The upload path is temporary and already deleted; do not leak it to the client.
    result.get("source_metadata", {}).pop("path", None)
    result["query"] = query
    return result


@app.post("/api/analyze-preset")
async def analyze_preset_endpoint(
    query: str = Form(...),
    preset_title: str = Form(""),
    lora_path: str = Form(""),
) -> Dict[str, Any]:
    """Not implemented: preset images are remote URLs, so there is no local raster to analyse.

    This used to return canned text as if a model had produced it. Upload the image instead, which
    goes through /api/analyze and is really analysed.
    """
    raise HTTPException(
        status_code=501,
        detail=("preset analysis is not implemented: preset imagery is a remote URL, not a local "
                "raster. Download the scene and upload it to /api/analyze for a real result."),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
