"""Download manageable dataset subsets for SatQuery AI.

Usage:
    python scripts/download_datasets.py --hf-token TOKEN --vrsbench-n 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any


def hf_login(token: str) -> None:
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[download] HF login successful.")


def save_vrsbench_subset(out_dir: Path, split: str, n: int, token: str) -> dict[str, Any]:
    """Stream VRSBench and save n samples to disk."""
    from datasets import load_dataset

    print(f"[download] Streaming VRSBench ({split}) — {n} samples …")
    subset = load_dataset(
        "xiang709/VRSBench",
        split=split,
        streaming=True,
        token=token,
    )
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(itertools.islice(subset, n)):
        image_obj = row.get("image")
        if image_obj is None:
            continue

        image_path = images_dir / f"vrsbench_{idx:04d}.jpg"
        if hasattr(image_obj, "save"):
            image_obj.save(str(image_path), format="JPEG")
        else:
            print(f"  [warn] row {idx}: image not a PIL object, skipping")
            continue

        records.append(
            {
                "id": idx,
                "image_path": str(image_path),
                "caption": row.get("caption", ""),
                "objects": row.get("objects"),
                "qa_pairs": row.get("qa_pairs"),
            }
        )
        print(f"  [{idx+1}/{n}] saved {image_path.name}")

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[download] VRSBench manifest: {manifest} ({len(records)} records)")

    return {
        "dataset": "VRSBench",
        "split": split,
        "num_records": len(records),
        "manifest": str(manifest),
        "images_dir": str(images_dir),
    }


def download_rsvqa_lr_metadata(out_dir: Path, token: str) -> dict[str, Any]:
    """Download RSVQA-LR annotations, with multiple fallback strategies."""
    from datasets import load_dataset

    rsvqa_dir = out_dir / "rsvqa_lr"
    rsvqa_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try RSVQA-HR from the Zenodo mirror
    hr_sources = [
        ("cvg/rs-vqa-hr", "test"),
        ("flamingos/RSVQA-HR", "test"),
    ]
    for repo_id, split in hr_sources:
        try:
            print(f"[download] Trying {repo_id} ({split}) …")
            ds = load_dataset(repo_id, split=split, streaming=True, token=token)
            records: list[dict[str, Any]] = []
            for idx, row in enumerate(itertools.islice(ds, 50)):
                img = row.get("image")
                img_path = None
                if img is not None and hasattr(img, "save"):
                    p = rsvqa_dir / f"rsvqa_{idx:04d}.jpg"
                    img.save(str(p), format="JPEG")
                    img_path = str(p)

                records.append({
                    "id": idx,
                    "image_path": img_path,
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "type": row.get("type", ""),
                })
                print(f"  [{idx+1}/50] RSVQA sample saved")

            if records:
                manifest = rsvqa_dir / "manifest.json"
                manifest.write_text(json.dumps(records, indent=2))
                print(f"[download] RSVQA manifest: {manifest}")
                return {"dataset": "RSVQA", "num_records": len(records), "manifest": str(manifest)}
        except Exception as exc:
            print(f"[download] {repo_id} failed: {exc}")
            continue

    # Strategy 2: Build synthetic RSVQA-style pairs from VRSBench QA data
    vrs_manifest = out_dir / "vrsbench" / "sample" / "manifest.json"
    if vrs_manifest.exists():
        print("[download] Building RSVQA-style pairs from VRSBench QA data …")
        with open(vrs_manifest, encoding="utf-8") as f:
            vrs_records = json.load(f)
        records = []
        for rec in vrs_records:
            img_path = rec.get("image_path")
            if not img_path or not Path(img_path).exists():
                continue
            qa_pairs = rec.get("qa_pairs") or []
            for qa in qa_pairs[:3]:
                q = qa.get("question") or qa.get("Q", "")
                a = qa.get("answer") or qa.get("A", "")
                if q and a:
                    records.append({
                        "id": len(records),
                        "image_path": img_path,
                        "question": str(q),
                        "answer": str(a),
                        "type": "synthetic_from_vrsbench",
                    })
            if len(records) >= 50:
                break
        records = records[:50]
        manifest = rsvqa_dir / "manifest.json"
        manifest.write_text(json.dumps(records, indent=2))
        print(f"[download] Synthetic RSVQA manifest: {manifest} ({len(records)} records)")
        return {"dataset": "RSVQA-synthetic", "num_records": len(records), "manifest": str(manifest)}

    print("[download] All RSVQA sources failed and no VRSBench data available.")
    return {
        "dataset": "RSVQA-LR",
        "num_records": 0,
        "note": "All HF datasets unavailable. Run VRSBench download first.",
    }


def write_official_links(base_dir: Path) -> None:
    notes = {
        "bigearthnet": "https://bigearth.net/",
        "rsvqa_hr": "https://zenodo.org/record/6344367",
        "rsvqa_lr": "https://doi.org/10.5281/zenodo.6344333",
        "rsvqa_repo": "https://github.com/syvlo/RSVQA",
        "vrsbench": "https://huggingface.co/datasets/xiang709/VRSBench",
        "rs_llava": "https://github.com/BigData-KSU/RS-LLaVA",
    }
    (base_dir / "official_links.json").write_text(json.dumps(notes, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SatQuery dataset subsets")
    parser.add_argument("--out", default="data", help="Base output directory")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    parser.add_argument("--vrsbench-split", default="train", help="VRSBench split")
    parser.add_argument("--vrsbench-n", type=int, default=50, help="Number of VRSBench samples")
    parser.add_argument("--skip-rsvqa", action="store_true", help="Skip RSVQA-LR download")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("[warn] No HF_TOKEN provided — some private datasets may fail.")

    if token:
        hf_login(token)

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    # 1. VRSBench
    vrs_dir = base / "vrsbench" / "sample"
    summary["vrsbench"] = save_vrsbench_subset(vrs_dir, args.vrsbench_split, args.vrsbench_n, token)

    # 2. RSVQA-LR
    if not args.skip_rsvqa:
        summary["rsvqa_lr"] = download_rsvqa_lr_metadata(base, token)

    write_official_links(base)
    summary["official_links"] = str(base / "official_links.json")

    summary_path = base / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Download complete. Summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
