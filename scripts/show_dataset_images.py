"""Visualize dataset samples — saves a grid image to outputs/.

Usage:
    python scripts/show_dataset_images.py --manifest data/vrsbench/sample/manifest.json --n 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
from PIL import Image


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_grid(records: list[dict], n: int, out_path: str, title: str = "Dataset Samples") -> str:
    records = [r for r in records if r.get("image_path") and Path(r["image_path"]).exists()][:n]

    if not records:
        print("[show] No valid images found in manifest.")
        return ""

    cols = min(4, len(records))
    rows = math.ceil(len(records) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, rec in enumerate(records):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#222")
            ax.text(0.5, 0.5, "load error", ha="center", va="center", color="white")

        # Caption or QA info
        caption = rec.get("caption", "")
        qa = rec.get("qa_pairs") or []
        label_lines = []
        if caption:
            label_lines.append(textwrap.fill(f"📝 {caption[:80]}", 40))
        if qa and len(qa) > 0:
            q = qa[0].get("question", qa[0].get("Q", ""))
            a = qa[0].get("answer", qa[0].get("A", ""))
            if q:
                label_lines.append(textwrap.fill(f"❓ {q[:60]}", 40))
            if a:
                label_lines.append(f"✅ {str(a)[:40]}")

        label = "\n".join(label_lines) if label_lines else f"ID: {rec.get('id', i)}"
        ax.set_title(label, fontsize=7, pad=4, wrap=True)
        ax.axis("off")

    # Hide unused subplots
    for i in range(len(records), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[show] Saved dataset preview → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show dataset sample images")
    parser.add_argument(
        "--manifest",
        default="data/vrsbench/sample/manifest.json",
        help="Path to manifest.json",
    )
    parser.add_argument("--n", type=int, default=12, help="Number of images to show")
    parser.add_argument(
        "--out", default="outputs/dataset_preview.png", help="Output image path"
    )
    parser.add_argument("--title", default="VRSBench — Remote Sensing Dataset Samples")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    print(f"[show] Loaded {len(records)} records from {args.manifest}")
    make_grid(records, args.n, args.out, args.title)


if __name__ == "__main__":
    main()
