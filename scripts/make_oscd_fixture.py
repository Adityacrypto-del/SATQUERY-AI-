"""Build a realistic GeoTIFF test pair from OSCD Sentinel-2 imagery.

Every georeferenced test in this branch so far used fixtures written by
``make_synthetic_fixtures.py``: clean float32, uncompressed, untiled,
axis-aligned, no nodata, band names always present. Real files are not like
that, and GeoTIFF is both the problem statement's primary declared format and
the format the ISRO/SAC evaluation set arrives in. "Built for it" and
"verified on it" are different claims.

This writes a pair that exercises what the synthetic fixtures do not:

*   **uint16 digital numbers**, not float reflectance, so the value-scaling
    assumption in ``semantic.py`` has to make a real decision;
*   **LZW compression and internal tiling**, the normal way imagery ships;
*   **a genuine UTM CRS and transform**, so area in m2 and centroid lat/lon
    are computed rather than skipped;
*   **a nodata value**, which must be excluded from statistics rather than
    counted as data;
*   **Sentinel-2 band names** (B01..B12, B8A), so the index resolver has to
    map real names onto logical bands instead of being handed red/green/blue.

The pixels are real Sentinel-2 multispectral from the Onera Satellite Change
Detection dataset, so NDVI, NDWI and NDBI take physically meaningful values.
Producer B has only ever been exercised on synthetic arrays; this is the
first time its thresholds meet real spectra.

The georeferencing is *plausible, not authoritative*: OSCD's Hugging Face
export carries pixels without the original geotransform, so a UTM grid at
Sentinel-2's 10 m resolution is attached here. That is honest for testing the
geospatial code path -- the arithmetic is exercised exactly as it would be --
but the coordinates do not locate the real scene, and nothing should treat
them as ground truth.

Usage::

    python -m scripts.make_oscd_fixture --parquet test.parquet --out-dir fixtures/oscd
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Sentinel-2 band order as OSCD stacks them. The names matter: indices.py
# resolves red from "b04", nir from "b08" and swir from "b11", so writing
# them makes the resolver do the work it would do on a real product.
SENTINEL2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]

NODATA = 0


def write_geotiff(path: str, array: np.ndarray, transform, crs) -> str:
    """Write a realistically-shaped GeoTIFF: uint16, tiled, compressed."""
    import rasterio

    count, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "nodata": NODATA,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint16))
        dst.descriptions = tuple(SENTINEL2_BANDS[:count])
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True,
                        help="OSCD_MSI parquet (blanchon/OSCD_MSI on Hugging Face)")
    parser.add_argument("--out-dir", default="fixtures/oscd")
    parser.add_argument("--index", type=int, default=0, help="which scene pair")
    parser.add_argument("--gsd", type=float, default=10.0,
                        help="Sentinel-2 ground sample distance, metres")
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    from affine import Affine
    from rasterio.crs import CRS

    table = pq.ParquetFile(args.parquet).read_row_group(0)
    row = table.slice(args.index, 1).to_pylist()[0]
    t1 = np.array(row["image1"])
    t2 = np.array(row["image2"])
    if t1.shape != t2.shape:
        raise ValueError(f"pair shapes differ: {t1.shape} vs {t2.shape}")

    # A plausible UTM grid. See the module docstring: this exercises the
    # geospatial arithmetic honestly but does not locate the real scene.
    crs = CRS.from_epsg(32631)  # UTM 31N, where several OSCD scenes fall
    transform = Affine(args.gsd, 0.0, 400000.0, 0.0, -args.gsd, 5400000.0)

    paths = []
    for name, array in (("t1", t1), ("t2", t2)):
        paths.append(write_geotiff(
            os.path.join(args.out_dir, f"OSCD_{args.index:02d}_{name}.tif"),
            array, transform, crs,
        ))

    print(f"scene index {args.index}: {t1.shape[0]} bands, "
          f"{t1.shape[1]}x{t1.shape[2]} px at {args.gsd} m")
    for path in paths:
        print(f"  {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
    print(f"  bands: {', '.join(SENTINEL2_BANDS[:t1.shape[0]])}")
    print(f"  uint16, LZW, tiled 128x128, nodata={NODATA}, {crs}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
