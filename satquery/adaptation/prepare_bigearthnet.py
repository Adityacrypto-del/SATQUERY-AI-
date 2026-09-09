"""Prepare a manageable BigEarthNet-S2 subset for remote-sensing adaptation.

BigEarthNet is large (59 GiB). This script downloads a small subset
using the official HuggingFace mirror when available, or records
the official download links for manual acquisition.

Usage:
    python -m satquery.adaptation.prepare_bigearthnet --n 500 --out data/bigearthnet
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path


BIGEARTHNET_OFFICIAL = "https://bigearth.net/"
BIGEARTHNET_HF = "BigEarthNet/BigEarthNet-S2"  # community mirror


def try_hf_stream(out_dir: Path, n: int, token: str) -> list[dict]:
    """Try to stream BigEarthNet-S2 patches from a HF mirror."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        print(f"[bigearthnet] Trying HF mirror {BIGEARTHNET_HF} (streaming) …")
        ds = load_dataset(BIGEARTHNET_HF, split="train", streaming=True, token=token)

        for i, row in enumerate(itertools.islice(ds, n)):
            img = row.get("image") or row.get("patch")
            img_path = None
            if img is not None and hasattr(img, "save"):
                p = out_dir / f"patch_{i:05d}.jpg"
                img.save(str(p), format="JPEG")
                img_path = str(p)

            records.append({
                "id": i,
                "image_path": img_path,
                "labels": row.get("labels", row.get("label", [])),
                "split": "train",
                "source": "BigEarthNet-S2",
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n}] patches downloaded")

        print(f"[bigearthnet] Downloaded {len(records)} patches.")
    except Exception as exc:
        print(f"[bigearthnet] HF stream failed: {exc}")
        print(f"[bigearthnet] Official download: {BIGEARTHNET_OFFICIAL}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BigEarthNet-S2 subset")
    parser.add_argument("--n", type=int, default=500, help="Number of patches to download")
    parser.add_argument("--out", default="data/bigearthnet")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token
    records = try_hf_stream(out_dir, args.n, token)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2))

    info = {
        "dataset": "BigEarthNet-S2",
        "num_records": len(records),
        "manifest": str(manifest),
        "official_url": BIGEARTHNET_OFFICIAL,
        "note": (
            "Full dataset is ~59 GiB. This is a manageable subset for adaptation. "
            "See https://bigearth.net/ for the complete dataset."
        ),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
