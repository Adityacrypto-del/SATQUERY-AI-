"""Evaluate VQA performance on RSVQA (or VRSBench QA) test set.

Computes exact-match accuracy and per-type breakdown.

Usage:
    python -m satquery.evaluation.evaluate_rsvqa \\
        --manifest data/rsvqa_lr/manifest.json \\
        --out outputs/eval_rsvqa.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from satquery.models.rs_vlm import RSVLM
from satquery.models.vqa import answer_vqa


def normalize(text: str) -> str:
    return str(text).strip().lower()


def evaluate(manifest_path: str, out_path: str, n: int, token: str, lora_path: str = "") -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)

    records = [r for r in records if r.get("image_path") and Path(r["image_path"]).exists()][:n]
    if not records:
        print("[eval-rsvqa] No valid records found.")
        return {}

    print(f"[eval-rsvqa] Evaluating on {len(records)} samples …")

    if lora_path and Path(lora_path).exists():
        model = RSVLM.load_with_lora(lora_path)
    else:
        model = RSVLM(model_name="BLIP-2-RS", adapted=False)

    results = []
    correct = 0
    type_stats: dict[str, dict] = {}

    for i, rec in enumerate(records):
        question = rec.get("question", "What do you see?")
        gt_answer = normalize(rec.get("answer", ""))
        q_type = rec.get("type", "unknown")

        pred = answer_vqa(model, rec["image_path"], question)
        pred_answer = normalize(pred.get("answer", ""))

        is_correct = pred_answer == gt_answer or gt_answer in pred_answer

        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "correct": 0}
        type_stats[q_type]["total"] += 1
        if is_correct:
            type_stats[q_type]["correct"] += 1
            correct += 1

        results.append({
            "id": rec.get("id", i),
            "image": rec["image_path"],
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "type": q_type,
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(records)}] accuracy so far: {correct/(i+1):.3f}")

    accuracy = correct / len(records) if records else 0.0
    per_type = {t: s["correct"] / max(s["total"], 1) for t, s in type_stats.items()}

    summary = {
        "dataset": "RSVQA",
        "num_samples": len(records),
        "overall_accuracy": round(accuracy, 4),
        "per_type_accuracy": per_type,
        "model": model.model_name,
        "adapted": model.adapted,
        "results": results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-rsvqa] Accuracy: {accuracy:.4f}")
    print(f"[eval-rsvqa] Results → {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA on RSVQA")
    parser.add_argument("--manifest", default="data/rsvqa_lr/manifest.json")
    parser.add_argument("--out", default="outputs/eval_rsvqa.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    evaluate(args.manifest, args.out, args.n, args.hf_token, args.lora_path)


if __name__ == "__main__":
    main()
