"""Prepare RS-LLaVA instruction dataset for LoRA adaptation.

Downloads the RS-instruction data from HuggingFace and converts it to a
format suitable for BLIP-2 LoRA fine-tuning.

Usage:
    python -m satquery.adaptation.prepare_instructions --out data/rs_instructions --n 500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def download_rs_instructions(out_dir: Path, n: int, token: str) -> list[dict]:
    """Try multiple HF sources for RS instruction/VQA data."""
    from datasets import load_dataset

    sources = [
        # RS-LLaVA instruction data (primary)
        ("BigData-KSU/RS-instructions", "train"),
        # RSVQA-LR as fallback
        ("EarthVQA/rsvqa-lr", "train"),
    ]

    records = []
    for repo_id, split in sources:
        try:
            print(f"[prepare] Trying {repo_id} ({split}) …")
            ds = load_dataset(repo_id, split=split, streaming=True, token=token)
            import itertools

            for row in itertools.islice(ds, n):
                img = row.get("image")
                img_path = None
                img_dir = out_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)

                if img is not None and hasattr(img, "save"):
                    p = img_dir / f"rs_inst_{len(records):05d}.jpg"
                    img.save(str(p), format="JPEG")
                    img_path = str(p)

                question = (
                    row.get("question")
                    or row.get("conversations", [{}])[0].get("value", "Describe this image.")
                )
                answer = (
                    row.get("answer")
                    or row.get("conversations", [{}])[-1].get("value", "")
                )

                records.append({
                    "image_path": img_path,
                    "question": str(question)[:512],
                    "answer": str(answer)[:512],
                })

            if records:
                print(f"[prepare] Loaded {len(records)} samples from {repo_id}.")
                break

        except Exception as exc:
            print(f"[prepare] {repo_id} failed: {exc}")
            continue

    return records


def build_synthetic_from_vrsbench(vrs_manifest: str, n: int) -> list[dict]:
    """Build instruction records from VRSBench QA pairs (always available)."""
    with open(vrs_manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    for rec in manifest:
        img_path = rec.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue
        caption = rec.get("caption", "")
        qa_pairs = rec.get("qa_pairs") or []

        # Captioning instruction
        if caption:
            records.append({
                "image_path": img_path,
                "question": "Describe this satellite image.",
                "answer": caption,
            })

        # VQA instructions
        for qa in qa_pairs[:3]:
            q = qa.get("question") or qa.get("Q", "")
            a = qa.get("answer") or qa.get("A", "")
            if q and a:
                records.append({
                    "image_path": img_path,
                    "question": str(q),
                    "answer": str(a),
                })

        if len(records) >= n:
            break

    return records[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RS instruction data for LoRA")
    parser.add_argument("--out", default="data/rs_instructions")
    parser.add_argument("--vrs-manifest", default="data/vrsbench/sample/manifest.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    token = args.hf_token
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try HF first, fall back to VRSBench
    records = download_rs_instructions(out_dir, args.n, token)

    if not records:
        print("[prepare] Using VRSBench QA pairs as training data …")
        records = build_synthetic_from_vrsbench(args.vrs_manifest, args.n)

    manifest_path = out_dir / "instructions.json"
    manifest_path.write_text(json.dumps(records, indent=2))
    print(f"[prepare] Saved {len(records)} instruction records → {manifest_path}")


if __name__ == "__main__":
    main()
