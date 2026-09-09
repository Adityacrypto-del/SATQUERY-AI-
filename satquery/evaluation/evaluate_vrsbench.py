"""Evaluate captioning on VRSBench using BLEU, ROUGE-L, and METEOR.

Usage:
    python -m satquery.evaluation.evaluate_vrsbench \\
        --manifest data/vrsbench/sample/manifest.json \\
        --out outputs/eval_vrsbench.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from satquery.models.captioning import generate_caption
from satquery.models.rs_vlm import RSVLM


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-4 and ROUGE-L."""
    metrics: dict = {}

    # BLEU via nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        smoothie = SmoothingFunction().method1
        bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie)
        metrics["bleu4"] = round(bleu, 4)
    except Exception as exc:
        metrics["bleu4"] = None
        metrics["bleu4_error"] = str(exc)

    # ROUGE-L via rouge_score
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for ref, pred in zip(references, predictions)]
        metrics["rougeL"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as exc:
        metrics["rougeL"] = None
        metrics["rougeL_error"] = str(exc)

    return metrics


def evaluate(manifest_path: str, out_path: str, n: int, lora_path: str = "") -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)

    records = [r for r in records if
               r.get("image_path") and Path(r["image_path"]).exists() and r.get("caption")][:n]

    if not records:
        print("[eval-vrsbench] No valid records with captions found.")
        return {}

    print(f"[eval-vrsbench] Evaluating captioning on {len(records)} samples …")

    if lora_path and Path(lora_path).exists():
        model = RSVLM.load_with_lora(lora_path)
    else:
        model = RSVLM(model_name="BLIP-2-RS", adapted=False)

    predictions = []
    references = []
    results = []

    for i, rec in enumerate(records):
        gt_caption = rec["caption"]
        with Image.open(rec["image_path"]) as image:
            pred = generate_caption(model, image.convert("RGB"), rec["image_path"])
        pred_caption = pred.get("caption", "")

        predictions.append(pred_caption)
        references.append(gt_caption)

        results.append({
            "id": rec.get("id", i),
            "image": rec["image_path"],
            "gt_caption": gt_caption,
            "pred_caption": pred_caption,
        })

        print(f"  [{i+1}/{len(records)}] pred: {pred_caption[:80]}…")

    metrics = compute_metrics(predictions, references)
    print(f"\n[eval-vrsbench] BLEU-4 : {metrics.get('bleu4')}")
    print(f"[eval-vrsbench] ROUGE-L: {metrics.get('rougeL')}")

    summary = {
        "dataset": "VRSBench",
        "num_samples": len(records),
        "metrics": metrics,
        "model": model.model_name,
        "adapted": model.adapted,
        "results": results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"[eval-vrsbench] Results → {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captioning on VRSBench")
    parser.add_argument("--manifest", default="data/vrsbench/sample/manifest.json")
    parser.add_argument("--out", default="outputs/eval_vrsbench.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--lora-path", default="")
    args = parser.parse_args()

    evaluate(args.manifest, args.out, args.n, args.lora_path)


if __name__ == "__main__":
    main()
