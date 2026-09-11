"""Generate SYNTHETIC GeoTIFF fixtures for smoke testing.

These are NOT real satellite data. No Cartosat, RISAT or Sentinel scene
exists anywhere in this repository. Every file this writes is prefixed
SYNTHETIC_ and carries a TIFF metadata tag saying so, precisely so that no
report can later imply a real acquisition was used.

What they are for: exercising code paths that SECOND's RGB PNGs cannot
reach -- multispectral index maths (NDVI/NDWI/NDBI), the SAR ratio operator,
and georeferenced area/coordinate derivation.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

# UTM 43N, 10 m pixels -- a plausible Indian-subcontinent projected grid.
CRS_UTM43N = CRS.from_epsg(32643)
TRANSFORM_10M = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)

TAG = {"SYNTHETIC": "true",
       "ORIGIN": "generated fixture, not real satellite data"}


def _write(path, array, band_names, nodata=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=CRS_UTM43N, transform=TRANSFORM_10M, nodata=nodata,
    ) as dst:
        dst.write(array.astype(np.float32))
        dst.descriptions = tuple(band_names)
        dst.update_tags(**TAG)
    return path


def make_optical(out_dir, size=128, shift=0, seed=0):
    """5-band optical pair. A vegetated block becomes built-up at t2.

    Bands are named so the index resolver finds them: B02/B03/B04 are
    blue/green/red, B08 is NIR, B11 is SWIR -- enough for all three indices.
    """
    rng = np.random.default_rng(seed)
    names = ["B02", "B03", "B04", "B08", "B11"]

    # Shared terrain structure, identical in both dates. Without this the
    # two scenes have no common signal, phase correlation has nothing to
    # lock onto, and the co-registration estimate is pure noise -- which is
    # exactly the false 4.47 px and 37.44 px rejections the first version of
    # these fixtures produced.
    from scipy import ndimage
    terrain = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, (size, size)), 4
    ).astype(np.float32)
    terrain = terrain / (np.abs(terrain).max() + 1e-9)

    def scene(built_slice):
        # Vegetation baseline: low red, high NIR, moderate SWIR.
        base = np.stack([
            np.full((size, size), 0.04),   # blue
            np.full((size, size), 0.07),   # green
            np.full((size, size), 0.05),   # red
            np.full((size, size), 0.42),   # NIR
            np.full((size, size), 0.18),   # SWIR
        ]).astype(np.float32)
        # Terrain modulates every band, so the dates share real structure.
        base = base * (1.0 + 0.85 * terrain[None])
        base += rng.normal(0, 0.004, base.shape).astype(np.float32)
        # A water body, constant across both dates (should NOT be flagged).
        base[:, 8:28, 8:28] = np.array(
            [0.05, 0.08, 0.04, 0.02, 0.01], dtype=np.float32
        )[:, None, None]
        if built_slice is not None:
            # Built-up: NIR collapses, SWIR and red rise.
            base[:, built_slice[0], built_slice[1]] = np.array(
                [0.16, 0.18, 0.22, 0.20, 0.34], dtype=np.float32
            )[:, None, None]
        return np.clip(base, 0.0, 1.0)

    t1 = scene(None)
    built = (slice(60 + shift, 100 + shift), slice(60, 100))
    t2 = scene(built)
    return (
        _write(os.path.join(out_dir, "SYNTHETIC_optical_t1.tif"), t1, names),
        _write(os.path.join(out_dir, "SYNTHETIC_optical_t2.tif"), t2, names),
    )


def make_sar(out_dir, size=128, seed=1):
    """Single-band SAR-like pair with multiplicative speckle.

    Speckle is modelled as gamma-distributed multiplicative noise, which is
    why the ratio operator rather than the difference is correct for this
    modality -- the point of including it in the smoke test.
    """
    rng = np.random.default_rng(seed)

    def speckle(backscatter, looks=4):
        return backscatter * rng.gamma(looks, 1.0 / looks, backscatter.shape)

    from scipy import ndimage
    terrain = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, (size, size)), 4
    ).astype(np.float32)
    terrain = terrain / (np.abs(terrain).max() + 1e-9)
    base = (0.08 * (1.0 + 0.9 * terrain)).astype(np.float32)
    base[20:50, 70:110] = 0.30          # a bright feature present at both dates
    t1 = speckle(base)[None]
    later = base.copy()
    later[70:100, 20:60] = 0.45          # new bright scatterer at t2
    t2 = speckle(later)[None]
    return (
        _write(os.path.join(out_dir, "SYNTHETIC_sar_t1.tif"),
               t1.astype(np.float32), ["VV"]),
        _write(os.path.join(out_dir, "SYNTHETIC_sar_t2.tif"),
               t2.astype(np.float32), ["VV"]),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/smoke/fixtures")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    written = list(make_optical(args.out_dir)) + list(make_sar(args.out_dir))
    written += list(make_optical(args.out_dir, shift=0, seed=2))[:0]
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
