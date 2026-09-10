from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import rasterio


@dataclass
class RSImage:
    """
    Common representation of a remote-sensing image.

    array:
        Image data in (bands, height, width) format.

    crs:
        Coordinate Reference System metadata.

    transform:
        Geospatial transform mapping pixels to coordinates.

    modality:
        Image modality: optical, sar, or unknown.

    band_names:
        Optional names of the image bands.

    gsd_m:
        Ground Sample Distance in metres, if available.
    """

    array: np.ndarray
    crs: Any
    transform: Any
    modality: str
    band_names: Optional[List[str]]
    gsd_m: Optional[float]

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("array must be a numpy ndarray")

        if self.array.ndim != 3:
            raise ValueError(
                "array must have shape (bands, height, width)"
            )

        if self.array.dtype != np.float32:
            raise ValueError(
                "array must use float32 dtype"
            )

        if self.modality not in {"optical", "sar", "unknown"}:
            raise ValueError(
                "modality must be optical, sar, or unknown"
            )


def load_raster(path, modality="unknown"):
    """
    Load a remote-sensing raster and preserve its geospatial metadata.

    Returns
    -------
    RSImage
        Image data in (bands, height, width), float32.
    """

    with rasterio.open(path) as src:
        array = src.read().astype(np.float32)

        crs = src.crs
        transform = src.transform

        # Pixel dimensions from the affine transform.
        gsd_x = abs(transform.a)
        gsd_y = abs(transform.e)

        # Use a single GSD only when pixels are approximately square.
        if not np.isclose(gsd_x, gsd_y):
            gsd_m = None
        else:
            gsd_m = float((gsd_x + gsd_y) / 2)

        band_names = None

        return RSImage(
            array=array,
            crs=crs,
            transform=transform,
            modality=modality,
            band_names=band_names,
            gsd_m=gsd_m,
        )