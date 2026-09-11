"""Track A: TerraQ-VL (third-party, VRSBench-adapted) zero-shot on its own held-out test.json.

Reference baseline only -- never Branch 1's result. Evaluates ONLY the published
``stage-2/data/test.json`` (197 images / 1,367 records), verified by SHA-256 before anything
runs. Generation is TerraQ-VL's own (``generate_heldout_records.load_records`` +
``inference.run_inference``, greedy, 256 new tokens); only the LLM precision differs (4-bit).

The cross-check scores TerraQ-VL's published predictions for the same checkpoint with the same
metric code, then compares ours vs theirs with a paired bootstrap over images.

    python reference/terraq_vl/run_vrsbench_eval.py --weights-dir <terraq-vl-weights> \
        --image-dir <dir with the 197 test images> [--limit 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]

TEST_JSON_SHA256 = "e2a83edba1b7de06f91a3b5259cbc5b9b97b9f1cfe2913454d09b1fc7f374697"  # stage-2/manifest.json
REF_PRED_SHA256 = "1251addc80145cb8b29ec4571b51f9850382e5f9beb1def74006d6cd14fdd287"
LICENSE_NOTE = (
    "TerraQ-VL weights: research / non-commercial, attribution required -- Qwen Research License "
    "(Qwen2.5-3B-Instruct base; forbids using outputs to improve non-Qwen LLMs) + VRSBench "
    "CC-BY-NC-4.0 (training data; imagery DOTA-v2/DIOR academic terms). TerraQ-VL code: MIT."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True, help="Local snapshot of grKnight/terraq-vl.")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--stage", choices=["stage2", "stage1"], default="stage2",
                    help="stage1 = connector-only fallback if stage2 does not fit.")
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "reference_eval"))
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test on the first N records.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from load_4bit import load_terraq, run_inference
    from metrics import paired_bootstrap, score
    from scripts.generate_heldout_records import load_records

    w = Path(args.weights_dir)
    test_json = w / "stage-2" / "data" / "test.json"
    ref_pred = w / "stage-2" / "predictions" / "predictions_full_heldout_stage2_checkpoint-2180.jsonl"
    if sha256(test_json) != TEST_JSON_SHA256:
        raise SystemExit(f"{test_json} is not the published held-out test split (sha256 mismatch); refusing.")
    if args.stage == "stage2":
        config, ckpt = w / "stage-2/config/finetune_vrsbench_stage2.yaml", w / "stage-2/checkpoints/checkpoint-2180"
    else:
        config, ckpt = w / "stage-1/config/pretrain_vrsbench.yaml", w / "stage-1/checkpoints/checkpoint-3270"

    records = load_records(str(test_json))
    if args.limit:
        records = records[: args.limit]
    missing = [r["image"] for r in records if not (Path(args.image_dir) / r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images missing from {args.image_dir}, e.g. {missing[:3]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"terraq_vl_{args.stage}_vrsbench" + (f"_smoke{args.limit}" if args.limit else "")
    pred_path = out_dir / f"{tag}_predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["index"]] = row

    model, load_info = load_terraq(str(config), str(ckpt))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with pred_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            if rec["index"] in done:
                continue
            resp = run_inference(model, str(Path(args.image_dir) / rec["image"]), prompt=rec["prompt"],
                                 max_new_tokens=args.max_new_tokens, temperature=0.0, device="cuda")
            row = {**rec, "response": resp}
            done[rec["index"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i == 1:
                print(f"[vram] peak after first sample: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            if i % 50 == 0:
                print(f"[{i}/{len(records)}] {time.time() - t0:.0f}s", flush=True)
    peak = torch.cuda.max_memory_allocated()

    ours = [done[r["index"]] for r in records]
    theirs_all = {json.loads(l)["index"]: json.loads(l) for l in ref_pred.open(encoding="utf-8")}
    theirs = [theirs_all[r["index"]] for r in records]
    for a, b in zip(ours, theirs):
        assert a["prompt"] == b["prompt"] and a["reference"] == b["reference"], a["index"]

    result = {
        "model": "TerraQ-VL (third-party, VRSBench-adapted) -- reference baseline, NOT Branch 1's own adaptation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": args.stage,
        "checkpoint": f"grKnight/terraq-vl {ckpt.relative_to(w).as_posix()}",
        "load": load_info,
        "split": {
            "file": "grKnight/terraq-vl stage-2/data/test.json",
            "sha256": TEST_JSON_SHA256,
            "records": len(records), "images": len({r["image"] for r in records}),
            "disjointness": "re-ran TerraQ-VL build_vrsbench_trainset.py (seed 42, test-fraction 0.02, "
                            "val-fraction 0.5) on VRSBench_train.json: rebuilt test/val equal published "
                            "files record-for-record; train/val/test share 0 images",
            "full_split": not args.limit,
        },
        "license": LICENSE_NOTE,
        "vram": {"gpu": torch.cuda.get_device_name(0),
                 "peak_allocated_gib_after_load": load_info["vram_after_load_bytes"] / 2**30,
                 "peak_allocated_gib_during_generation": peak / 2**30},
        "generation": {"prompt": "test.json human turn, <image> removed (TerraQ-VL generate_heldout_records)",
                       "decoding": "greedy", "max_new_tokens": args.max_new_tokens},
        "metrics": score(ours),
        "cross_check": {
            "reference_predictions": f"grKnight/terraq-vl {ref_pred.relative_to(w).as_posix()} (bf16, sha256 {REF_PRED_SHA256[:16]}...)",
            "reference_predictions_sha256_ok": sha256(ref_pred) == REF_PRED_SHA256,
            "reference_metrics_same_harness": score(theirs),
            "exact_response_agreement": sum(a["response"].strip() == b["response"].strip()
                                            for a, b in zip(ours, theirs)) / len(ours),
            "paired_bootstrap_ours_minus_reference": paired_bootstrap(ours, theirs) if not args.limit else None,
        },
        "predictions_file": pred_path.name,
    }
    out_json = out_dir / f"{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("metrics", "vram")}, indent=2))
    print(json.dumps(result["cross_check"], indent=2))
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
