"""HTTP surface for the branch-2 specialist, for the frontend to call.

The existing frontend backend on ``ayushFRONT`` exposes ``/api/analyze``,
which takes **one** image. Bi-temporal analysis needs two, so there is no
endpoint the frontend can use to reach this branch today. Rather than edit
another team's file, this stands alone: run it on its own port and the
frontend either proxies to it or calls it directly.

Endpoints:

``GET  /api/bitemporal/describe``  the tool-registry descriptor
``GET  /api/bitemporal/health``    whether a checkpoint is loaded
``POST /api/bitemporal/analyze``   two images + a query -> evidence JSON
``GET  /api/bitemporal/evidence/{name}``  a rendered overlay PNG

The response is the ``BiTemporalEvidence`` the controller receives, verbatim,
plus URLs for the overlay images. Nothing is added for presentation and
nothing is filled in when missing: a field the analysis could not compute
comes back null, and the confidence carries the basis that explains it --
including when that basis is "no measured basis applies".

Run::

    python -m app.bitemporal_server --port 8008
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from tools.change_analysis.api import BiTemporalSpecialist, describe_tool

DEFAULT_CHECKPOINT = "outputs/segmentation/run1_unweighted/best.pt"
EVIDENCE_DIR = os.path.abspath("outputs/evidence")
ACCEPTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="SatQuery AI -- bi-temporal change analysis",
    description="Branch-2 specialist: two dates in, evidence-grounded answer out.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_specialist: Optional[BiTemporalSpecialist] = None
_checkpoint: Optional[str] = None


def get_specialist() -> BiTemporalSpecialist:
    """Built once. Loading the checkpoint costs seconds; a query costs ms."""
    global _specialist
    if _specialist is None:
        _specialist = BiTemporalSpecialist(
            checkpoint=_checkpoint, overlay_dir=EVIDENCE_DIR
        )
    return _specialist


@app.get("/api/bitemporal/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "checkpoint": _checkpoint,
        "checkpoint_present": bool(_checkpoint and os.path.exists(_checkpoint)),
        "note": (
            "Without a checkpoint the deterministic index path still runs, but "
            "no CDVQA class questions can be answered."
        ),
    }


@app.get("/api/bitemporal/describe")
def describe() -> Dict[str, Any]:
    """What this specialist accepts, returns, and refuses."""
    return describe_tool(_checkpoint)


@app.get("/api/bitemporal/evidence/{name}")
def evidence_image(name: str):
    """Serve a rendered overlay. Name only -- no path traversal."""
    if os.path.basename(name) != name:
        raise HTTPException(status_code=400, detail="invalid evidence name")
    path = os.path.join(EVIDENCE_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no such evidence image")
    return FileResponse(path, media_type="image/png")


@app.post("/api/bitemporal/analyze")
async def analyze(
    image_t1: UploadFile = File(..., description="earlier image"),
    image_t2: UploadFile = File(..., description="later image"),
    query: str = Form("", description="natural-language question; optional"),
) -> Dict[str, Any]:
    """Analyse a bi-temporal pair.

    Failures are reported, not raised into a 500: an incompatible pair is a
    normal outcome with an explanation, and the frontend should be able to
    show that explanation rather than a stack trace.
    """
    paths = []
    workdir = tempfile.mkdtemp(prefix="bitemporal_")
    try:
        for upload, label in ((image_t1, "t1"), (image_t2, "t2")):
            if not upload or not upload.filename:
                raise HTTPException(status_code=400, detail=f"{label} missing")
            suffix = os.path.splitext(upload.filename)[1].lower() or ".png"
            if suffix not in ACCEPTED:
                raise HTTPException(
                    status_code=400,
                    detail=(f"{label}: unsupported format {suffix!r}; "
                            f"accepted: {sorted(ACCEPTED)}"),
                )
            path = os.path.join(workdir, f"{label}{suffix}")
            with open(path, "wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            paths.append(path)

        scene_id = f"web_{int(time.time() * 1000)}"
        evidence = get_specialist().analyze(paths[0], paths[1], query,
                                            scene_id=scene_id)
        payload = dataclasses.asdict(evidence)
        # Overlays are written to disk; hand back URLs the browser can load.
        payload["overlays"] = {
            name: f"/api/bitemporal/evidence/{os.path.basename(path)}"
            for name, path in (evidence.overlays or {}).items()
        }
        payload["ok"] = True
        return payload
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - report, do not 500
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "answer": None,
            "confidence": 0.0,
            "confidence_basis": "none: analysis did not complete",
            "summary": f"Analysis failed: {exc}",
            "trace": [],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)

    global _checkpoint
    _checkpoint = args.checkpoint if os.path.exists(args.checkpoint) else None
    if _checkpoint is None:
        print(f"note: no checkpoint at {args.checkpoint!r}; the deterministic "
              f"index path will serve instead.", file=sys.stderr)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
