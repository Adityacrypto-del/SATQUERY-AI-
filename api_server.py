"""
api_server.py — FastAPI Backend Endpoint for the SatQuery AI Frontend & Query Router.

Provides:
- POST /api/optical-sar: Accepts uploaded optical + SAR images + user query and returns complete structured response.
- GET /api/health: Health check endpoint.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from optical_sar_branch import OpticalSARBranchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpticalSAR_API")

app = FastAPI(
    title="SatQuery AI — Optical + SAR Specialist Branch API",
    version="1.0.0"
)

# Enable CORS for Next.js / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service once at server startup
logger.info("Loading OpticalSAR Specialist Model and Task Adapter...")
service = OpticalSARBranchService()
logger.info("OpticalSAR Branch Service ready!")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "branch": "OPTICAL + SAR", "device": str(service.device)}


@app.post("/api/optical-sar")
async def process_optical_sar(
    optical_file: UploadFile = File(..., description="Uploaded Sentinel-2 Optical image (.png, .jpg, .npy)"),
    sar_file: UploadFile = File(..., description="Uploaded Sentinel-1 SAR image (.png, .jpg, .npy)"),
    query: str = Form("Analyze the land cover, vegetation health, water bodies, and structural density.", description="User natural language question"),
):
    """
    Endpoint for frontend image upload and query processing.
    """
    try:
        optical_bytes = await optical_file.read()
        sar_bytes = await sar_file.read()

        logger.info(f"Received query: '{query}' | Optical: {optical_file.filename} | SAR: {sar_file.filename}")

        # Run branch pipeline
        result = service.process(
            optical_image=optical_bytes,
            sar_image=sar_bytes,
            user_query=query,
        )

        return {
            "success": True,
            "filename_optical": optical_file.filename,
            "filename_sar": sar_file.filename,
            "data": result,
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
