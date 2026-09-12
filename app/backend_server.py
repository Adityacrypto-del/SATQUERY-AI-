"""FastAPI backend server for SatQuery AI frontend application.

Provides complete REST API endpoints implementing the project architecture:
                     USER
                       │
                 Natural Language
                       │
                       ▼
                 QUERY ROUTER
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   SINGLE IMAGE    BI-TEMPORAL    OPTICAL + SAR
        │              │              │
        ▼              ▼              ▼
   Specialist       Specialist      Specialist
     Model             Model           Model
        │              │              │
        ▼              ▼              ▼
    Features        Change tokens    Fused tokens
    Objects         T1/T2 features   Optical tokens
    Regions         Change map       SAR tokens
        │              │              │
        ▼              ▼              ▼
   Task Adapter     Task Adapter    Task Adapter
        │              │              │
        └──────────────┼──────────────┘
                       ↓
               SHARED REASONING LLM
                       ↓
                  FINAL ANSWER
                       ↓
             Evidence + Confidence
                       ↓
                    GUI / Voice
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from satquery.agent.query_router import route_query_and_inputs
from satquery.models.shared_reasoner import execute_unified_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SatQueryBackend")

app = FastAPI(
    title="SatQuery AI Backend",
    description="Multi-Modal Remote Sensing Vision-Language & Spatial Reasoning API Server",
    version="2.0.0",
)

# Enable CORS for local dev (Vite running on 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(__file__).parent.parent / "outputs" / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Curated Presets across all 3 Specialist Modes
PRESETS = [
    # --- MODE 1: SINGLE IMAGE SPECIALIST ---
    {
        "id": "preset_single_urban",
        "mode": "single_image",
        "title": "Rotterdam Harbor & Logistics Terminal",
        "category": "Single Image • Urban & Maritime",
        "format": "GeoTIFF / PNG (0.5m GSD)",
        "resolution": "0.5m / px",
        "location": "Rotterdam, Netherlands",
        "image_url": "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "Count shipping containers, docks, and maritime vessels.",
            "Describe the overall land cover and industrial density.",
            "Identify transportation corridors and port infrastructure."
        ],
        "default_land_use": {"Urban / Built-up": 55, "Water Bodies": 30, "Industrial": 12, "Vegetation": 3}
    },
    {
        "id": "preset_single_agri",
        "mode": "single_image",
        "title": "Kansas Pivot Irrigation Cropland",
        "category": "Single Image • Agriculture",
        "format": "Sentinel-2 L2A (10m)",
        "resolution": "10m / px",
        "location": "Kansas, United States",
        "image_url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "What crop health or irrigation patterns are visible?",
            "Estimate percentage of active agricultural cropland.",
            "Describe the crop vigor and vegetative distribution."
        ],
        "default_land_use": {"Agricultural Cropland": 72, "Bare Soil": 18, "Vegetation": 8, "Water": 2}
    },
    {
        "id": "preset_single_coastal",
        "mode": "single_image",
        "title": "Great Barrier Reef & Coastal Inlets",
        "category": "Single Image • Marine & Coastal",
        "format": "PlanetScope (3m)",
        "resolution": "3m / px",
        "location": "Cairns, Australia",
        "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80",
        "suggested_queries": [
            "Identify coral formations and shoreline stability.",
            "Calculate marine vs terrestrial land coverage.",
            "Describe water clarity and bathymetric boundaries."
        ],
        "default_land_use": {"Water / Marine": 68, "Coastal Forest": 22, "Sand / Beach": 10}
    },

    # --- MODE 2: BI-TEMPORAL SPECIALIST (T1 vs T2) ---
    {
        "id": "preset_bitemporal_flood",
        "mode": "bitemporal",
        "title": "València Flash Flood Inundation (2024)",
        "category": "Bi-Temporal • Flood Progression",
        "format": "Sentinel-2 Temporal Pair (T1 Pre / T2 Post)",
        "resolution": "10m / px",
        "location": "València Basin, Spain",
        "t1_label": "Pre-Flood (October 2024)",
        "t2_label": "Post-Flood Inundated (November 2024)",
        "image_url": "https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=1200&q=80",
        "t1_url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=800&q=80",
        "t2_url": "https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=800&q=80",
        "suggested_queries": [
            "What is the flood inundation extent between T1 and T2?",
            "Detect water surface expansion and submerged infrastructure.",
            "Quantify the percentage of flooded agricultural area."
        ],
        "default_land_use": {"Newly Flooded Land": 34, "Persistent Water": 18, "Unaltered Terrestrial": 48}
    },
    {
        "id": "preset_bitemporal_forest",
        "mode": "bitemporal",
        "title": "Amazon Rainforest Canopy Loss (2020-2024)",
        "category": "Bi-Temporal • Deforestation",
        "format": "Landsat-8 & 9 Surface Reflectance",
        "resolution": "30m / px",
        "location": "Rondônia, Brazil",
        "t1_label": "Intact Canopy (2020)",
        "t2_label": "Cleared Parcels (2024)",
        "image_url": "https://images.unsplash.com/photo-1516214104703-d870798883c5?auto=format&fit=crop&w=1200&q=80",
        "t1_url": "https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=800&q=80",
        "t2_url": "https://images.unsplash.com/photo-1516214104703-d870798883c5?auto=format&fit=crop&w=800&q=80",
        "suggested_queries": [
            "Quantify canopy cover loss and road access cuts.",
            "Compare vegetation density index between 2020 and 2024.",
            "Identify logging corridors and clear-cut boundaries."
        ],
        "default_land_use": {"Deforested Parcels": 26, "Intact Canopy": 64, "Secondary Growth": 10}
    },
    {
        "id": "preset_bitemporal_urban",
        "mode": "bitemporal",
        "title": "Dubai Waterfront & Desert Urban Expansion",
        "category": "Bi-Temporal • Urban Growth",
        "format": "High-Res Optical Temporal Pair",
        "resolution": "1m / px",
        "location": "Dubai, UAE",
        "t1_label": "Desert Baseline (2018)",
        "t2_label": "Constructed Districts (2024)",
        "image_url": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=1200&q=80",
        "t1_url": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=800&q=80",
        "t2_url": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&w=800&q=80",
        "suggested_queries": [
            "How much new built-up area was constructed between T1 and T2?",
            "Identify new artificial islands and coastal seawalls.",
            "Estimate percentage of desert converted to paved infrastructure."
        ],
        "default_land_use": {"New Built Structures": 38, "Existing City": 42, "Desert Sand": 20}
    },

    # --- MODE 3: OPTICAL + SAR SPECIALIST (Multimodal Fusion) ---
    {
        "id": "preset_optical_sar_cloud",
        "mode": "optical_sar",
        "title": "Cloud-Covered Harbor with Penetrating SAR",
        "category": "Optical + SAR • Cloud Penetration",
        "format": "Sentinel-2 (13 Bands) + Sentinel-1 (VV/VH)",
        "resolution": "10m / px (Co-registered)",
        "location": "Singapore Strait",
        "optical_label": "Sentinel-2 Optical (Cloudy RGB)",
        "sar_label": "Sentinel-1 SAR Radar (C-band VV/VH)",
        "image_url": "https://images.unsplash.com/photo-1559827291-72ee739d0d9a?auto=format&fit=crop&w=1200&q=80",
        "optical_url": "https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=800&q=80",
        "sar_url": "https://images.unsplash.com/photo-1559827291-72ee739d0d9a?auto=format&fit=crop&w=800&q=80",
        "suggested_queries": [
            "Use SAR radar to penetrate optical clouds and count ships.",
            "Analyze radar backscatter vs optical reflectance for port facilities.",
            "Detect water surface and coastal outline through cloud cover."
        ],
        "default_land_use": {"SAR-Verified Vessels": 28, "Port Infrastructure": 40, "Water": 32}
    },
    {
        "id": "preset_optical_sar_mangrove",
        "mode": "optical_sar",
        "title": "Sundarbans Coastal Mangrove & S1 Radar",
        "category": "Optical + SAR • Wetland Roughness",
        "format": "Sentinel-2 + Sentinel-1 Dual-Pol",
        "resolution": "10m / px",
        "location": "Sundarbans, India/Bangladesh",
        "optical_label": "Sentinel-2 Optical Multispectral",
        "sar_label": "Sentinel-1 SAR Volumetric Scatter",
        "image_url": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=1200&q=80",
        "optical_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80",
        "sar_url": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=800&q=80",
        "suggested_queries": [
            "Evaluate radar volumetric scattering vs optical chlorophyll NDVI.",
            "Differentiate tidal mudflats from permanent water using SAR dB.",
            "Assess mangrove canopy structure and moisture levels."
        ],
        "default_land_use": {"Mangrove Canopy": 52, "Tidal Water Channels": 36, "Mudflats": 12}
    }
]


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Return comprehensive system status across all 3 specialist branches and Query Router."""
    return {
        "status": "online",
        "system": "SatQuery AI Multimodal Architecture",
        "query_router": {
            "status": "active",
            "modes_supported": ["single_image", "bitemporal", "optical_sar"]
        },
        "branches": {
            "single_image": {
                "specialist": "Single Image Specialist (Features, Objects, Regions)",
                "adapter": "Visual Projector (512 -> 2048)",
                "status": "ready"
            },
            "bitemporal": {
                "specialist": "Bi-Temporal Specialist (Change tokens, T1/T2 features, Change map)",
                "adapter": "Temporal Projector (512 -> 2048)",
                "status": "ready"
            },
            "optical_sar": {
                "specialist": "Optical+SAR Specialist (ResNet18 13-band + ResNet18 SAR 2-channel)",
                "adapter": "Visual Projector (512 -> 2048)",
                "status": "ready"
            }
        },
        "shared_reasoning_llm": "Qwen2.5-72B Multimodal Engine / Remote OpenRouter",
        "timestamp": time.time()
    }


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    """Return preset demo satellite datasets across all specialist modes."""
    return {"presets": PRESETS}


@app.post("/api/route")
async def route_endpoint(
    query: str = Form(...),
    explicit_mode: str = Form("auto"),
    has_t1_t2: bool = Form(False),
    has_optical_sar: bool = Form(False),
) -> Dict[str, Any]:
    """Query Router classification preview."""
    decision = route_query_and_inputs(
        query=query,
        explicit_mode=explicit_mode,
        has_t1_t2=has_t1_t2,
        has_optical_sar=has_optical_sar,
    )
    return {
        "mode": decision.mode,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "rule_matched": decision.rule_matched,
        "suggested_task": decision.suggested_task,
        "tokens_expected": decision.tokens_expected,
    }


@app.post("/api/analyze/unified")
async def analyze_unified_endpoint(
    query: str = Form(...),
    mode: str = Form("auto"),
    preset_id: str = Form(""),
    image: Optional[UploadFile] = File(None),
    image_t1: Optional[UploadFile] = File(None),
    image_t2: Optional[UploadFile] = File(None),
    optical_image: Optional[UploadFile] = File(None),
    sar_image: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Universal Multimodal Endpoint:
    Accepts natural language query + dynamic file uploads or preset selections,
    routes through Query Router -> Specialist Branch -> Task Adapter -> Shared Reasoning LLM.
    """
    preset_data = None
    if preset_id:
        preset_data = next((p for p in PRESETS if p["id"] == preset_id), None)

    # Read byte buffers if files uploaded
    img_bytes = await image.read() if image else None
    t1_bytes = await image_t1.read() if image_t1 else None
    t2_bytes = await image_t2.read() if image_t2 else None
    opt_bytes = await optical_image.read() if optical_image else None
    sar_bytes = await sar_image.read() if sar_image else None

    # Run unified pipeline
    result = execute_unified_pipeline(
        query=query,
        explicit_mode=mode,
        image=img_bytes,
        image_t1=t1_bytes,
        image_t2=t2_bytes,
        optical_image=opt_bytes,
        sar_image=sar_bytes,
        preset_data=preset_data,
    )

    return result


# Backward-compatible endpoints
@app.post("/api/analyze-preset")
async def analyze_preset_endpoint(
    query: str = Form(...),
    preset_title: str = Form(""),
    preset_id: str = Form(""),
    mode: str = Form("auto"),
) -> Dict[str, Any]:
    """Preset analyzer supporting all 3 specialist modes."""
    preset_data = None
    if preset_id:
        preset_data = next((p for p in PRESETS if p["id"] == preset_id), None)
    elif preset_title:
        preset_data = next((p for p in PRESETS if p["title"].lower() == preset_title.lower()), None)

    return execute_unified_pipeline(
        query=query,
        explicit_mode=mode or (preset_data.get("mode") if preset_data else "auto"),
        preset_data=preset_data,
    )


@app.post("/api/analyze")
async def legacy_analyze_endpoint(
    image: UploadFile = File(...),
    query: str = Form(...),
    mode: str = Form("auto"),
) -> Dict[str, Any]:
    """Legacy single image upload endpoint mapped to unified pipeline."""
    img_bytes = await image.read()
    return execute_unified_pipeline(
        query=query,
        explicit_mode=mode,
        image=img_bytes,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend_server:app", host="0.0.0.0", port=8000, reload=True)
