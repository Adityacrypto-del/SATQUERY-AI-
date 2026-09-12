from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class LoadedImage:
    path: str
    extension: str
    width: int
    height: int
    mode: str
    array: np.ndarray
    metadata: Dict[str, Any]


def validate_single_image(image_path: str) -> Dict[str, object]:
    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "reason": "file_not_found"}

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"ok": False, "reason": f"unsupported_extension:{ext}"}

    if ext in {".tif", ".tiff"}:
        if rasterio is None:
            return {"ok": False, "reason": "rasterio_not_installed"}
        try:
            with rasterio.open(path) as src:
                if src.width <= 0 or src.height <= 0:
                    return {"ok": False, "reason": "invalid_dimensions"}
                return {
                    "ok": True,
                    "extension": ext,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                }
        except Exception as exc:
            return {"ok": False, "reason": f"decode_error:{exc}"}

    try:
        with Image.open(path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return {"ok": False, "reason": "invalid_dimensions"}
            return {"ok": True, "extension": ext, "width": width, "height": height, "bands": len(img.getbands())}
    except Exception as exc:
        return {"ok": False, "reason": f"decode_error:{exc}"}


def load_image(image_path: str) -> LoadedImage:
    path = Path(image_path)
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        if rasterio is None:
            raise RuntimeError("rasterio is required for TIFF/GeoTIFF input")
        with rasterio.open(path) as src:
            # Keep every source band here.  The model-preprocessing layer is
            # responsible for selecting an RGB-compatible representation.
            # Dropping bands while loading makes the decision impossible to
            # audit and is particularly harmful for Sentinel-2 inputs.
            arr = src.read()
            arr = np.moveaxis(arr, 0, -1)
            descriptions = list(src.descriptions or ())
            color_interpretations = [getattr(item, "name", str(item)) for item in src.colorinterp]
            metadata = {
                "crs": str(src.crs),
                "transform": str(src.transform),
                "count": src.count,
                "band_descriptions": descriptions,
                "color_interpretations": color_interpretations,
                "nodata": src.nodata,
            }
            return LoadedImage(
                path=str(path),
                extension=ext,
                width=src.width,
                height=src.height,
                mode="TIFF",
                array=arr,
                metadata=metadata,
            )

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.array(rgb)
        return LoadedImage(
            path=str(path),
            extension=ext,
            width=rgb.width,
            height=rgb.height,
            mode="RGB",
            array=arr,
            metadata={"color_interpretations": ["red", "green", "blue"]},
        )
