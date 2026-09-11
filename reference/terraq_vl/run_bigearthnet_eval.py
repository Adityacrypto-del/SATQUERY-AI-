"""Track B: the same TerraQ-VL checkpoint, zero-shot, on the BigEarthNet.txt benchmark split.

Reference baseline only -- TerraQ-VL is third-party and VRSBench-adapted; BigEarthNet is out of
domain for it, so a score below Track A's is expected, not a bug.

What is evaluated: ``split == "bench"`` rows of BigEarthNet.txt (arXiv 2603.29630; HF
BIFOLD-BigEarthNetv2-0/BigEarthNet.txt), types ``binary`` (6,927) and ``mcq`` (5,550) over 1,082
image pairs -- the paper's manually verified benchmark split. Captioning / referring-box rows are
not scored here.

Input is RGB ONLY. TerraQ-VL's CLIP encoder takes 3 channels, so each Sentinel-2 patch is rendered
from B04/B03/B02 through Branch 1's own ``to_model_rgb`` (per-image 2-98 percentile stretch). No
NIR/SWIR/SAR information reaches the model -- the multi-sensor content of the benchmark is unused.

Prompting (the paper does not publish its prompt): ``[vqa] `` task tag (TerraQ-VL's training
format) + the question + a format instruction. Answers are extracted deterministically; a response
with no extractable answer is scored wrong and clears the IF (instruction-following) flag, as in
the paper's tables.

    python reference/terraq_vl/run_bigearthnet_eval.py --weights-dir <terraq-vl-weights> \
        --bench-parquet <bench.parquet> --patch-dir <dir of <patch_id>.npz> [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]

SUFFIX = {
    "binary": " Answer with yes or no.",
    "mcq": " Answer with the letter (a, b, c or d) of the correct option only.",
}
PAPER = "arXiv 2603.29630 v2, Tables 3-4 (bench split)"
LICENSE_NOTE = (
    "TerraQ-VL weights: research / non-commercial (Qwen Research License + VRSBench CC-BY-NC-4.0). "
    "BigEarthNet.txt annotations: CDLA-Permissive-1.0. BigEarthNet v2.0 imagery read from the "
    "unofficial LMDB mirror hackelle/BigEarthNetV2-LMDB (CDLA-Permissive-1.0; official files at "
    "zenodo.org/records/10891137 take precedence)."
)


def extract_binary(resp: str):
    found = set(re.findall(r"\b(yes|no)\b", resp.lower()))
    return found.pop() if len(found) == 1 else None  # both or neither -> ambiguous


_OPT = re.compile(r"(?:^|\s)([a-d])\)\s*(.*?)(?=,\s*[a-d]\)|$)", re.S)


def options_of(question: str) -> dict:
    return {k: v.strip().rstrip(",?.").lower() for k, v in _OPT.findall(question)}


def extract_mcq(resp: str, question: str):
    r = resp.strip().lower()
    if len(set(re.findall(r"(?:^|[\s(])([a-d])(?=[\s).,:]|$)", r))) > 1:
        return None  # several option letters named -> ambiguous
    m = (re.match(r"^\(?([a-d])\)?(?:[\s).:,]|$)", r)
         or re.search(r"\b(?:option|answer(?: is)?|correct(?: option)?(?: is)?)[:\s]*\(?([a-d])\)?\b", r))
    if m:
        return m.group(1)
    hits = [k for k, text in options_of(question).items() if text and text in r]
    return hits[0] if len(hits) == 1 else None


def render_rgb(npz_path: Path, png_path: Path) -> dict:
    import numpy as np
    from PIL import Image

    from satquery.preprocessing.multispectral import to_model_rgb

    bands = np.load(npz_path)
    arr = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1).astype("float32")
    rgb, meta = to_model_rgb(arr, {"band_descriptions": ["B04", "B03", "B02"]})
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb * 255).astype("uint8")).save(png_path)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--bench-parquet", required=True)
    ap.add_argument("--patch-dir", required=True)
    ap.add_argument("--render-dir", default=None)
    ap.add_argument("--stage", choices=["stage2", "stage1"], default="stage2")
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "reference_eval"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from load_4bit import load_terraq, run_inference

    df = pd.read_parquet(args.bench_parquet)
    if set(df["split"]) != {"bench"}:
        raise SystemExit(f"refusing: input contains non-bench splits {sorted(set(df['split']))}")
    df = df[df["type"].isin(["binary", "mcq"])].sort_values("ID").reset_index(drop=True)
    if args.limit:
        df = df.groupby("type", group_keys=False).head(args.limit // 2).reset_index(drop=True)

    patch_dir = Path(args.patch_dir)
    render_dir = Path(args.render_dir or patch_dir.parent / "bench_rgb")
    missing = [p for p in df["patch_id"].unique() if not (patch_dir / f"{p}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} bench patches missing from {patch_dir}, e.g. {missing[:3]}")
    render_meta = None
    for p in df["patch_id"].unique():
        png = render_dir / f"{p}.png"
        if not png.exists() or render_meta is None:
            render_meta = render_rgb(patch_dir / f"{p}.npz", png)

    w = Path(args.weights_dir)
    if args.stage == "stage2":
        config, ckpt = w / "stage-2/config/finetune_vrsbench_stage2.yaml", w / "stage-2/checkpoints/checkpoint-2180"
    else:
        config, ckpt = w / "stage-1/config/pretrain_vrsbench.yaml", w / "stage-1/checkpoints/checkpoint-3270"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"terraq_vl_{args.stage}_bigearthnet" + (f"_smoke{args.limit}" if args.limit else "")
    pred_path = out_dir / f"{tag}_predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["ID"]] = row

    model, load_info = load_terraq(str(config), str(ckpt))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with pred_path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(df.itertuples(index=False), 1):
            if int(r.ID) in done:
                continue
            prompt = f"[vqa] {r.input}{SUFFIX[r.type]}"
            resp = run_inference(model, str(render_dir / f"{r.patch_id}.png"), prompt=prompt,
                                 max_new_tokens=args.max_new_tokens, temperature=0.0, device="cuda")
            pred = extract_binary(resp) if r.type == "binary" else extract_mcq(resp, r.input)
            row = {"ID": int(r.ID), "patch_id": r.patch_id, "type": r.type, "category": r.category,
                   "prompt": prompt, "reference": r.output, "response": resp, "extracted": pred,
                   "correct": pred is not None and pred == str(r.output).strip().lower()}
            done[row["ID"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"[{i}/{len(df)}] {time.time() - t0:.0f}s", flush=True)
    peak = torch.cuda.max_memory_allocated()

    rows = [done[int(i)] for i in df["ID"]]
    metrics = {}
    for ty in ("binary", "mcq"):
        sub = [r for r in rows if r["type"] == ty]
        if not sub:
            continue
        cats = defaultdict(list)
        for r in sub:
            cats[r["category"]].append(r["correct"])
        metrics[ty] = {
            "n": len(sub),
            "overall_accuracy": sum(r["correct"] for r in sub) / len(sub),
            "per_category_accuracy": {c: sum(v) / len(v) for c, v in sorted(cats.items())},
            "extraction_failures": sum(r["extracted"] is None for r in sub),
            "IF_all_extracted": all(r["extracted"] is not None for r in sub),
        }
    paper = json.loads((HERE / "ben_paper_tables.json").read_text(encoding="utf-8"))

    result = {
        "model": "TerraQ-VL (third-party, VRSBench-adapted) -- reference baseline, NOT Branch 1's own adaptation",
        "domain_note": "out-of-domain zero-shot: adapted on VRSBench aerial RGB, evaluated on Sentinel-2 "
                       "120x120 patches rendered to RGB; lower than Track A is expected",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": args.stage,
        "checkpoint": f"grKnight/terraq-vl {ckpt.relative_to(w).as_posix()}",
        "load": load_info,
        "split": {"dataset": "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", "split": "bench",
                  "types": ["binary", "mcq"], "records": len(rows),
                  "image_pairs": int(df["patch_id"].nunique()), "full_split": not args.limit,
                  "disjointness": "bench/train/validation/test patch sets pairwise disjoint (checked on the parquet)"},
        "input": {"bands": "S2 B04/B03/B02 only (RGB); S1 and other S2 bands unused",
                  "render": render_meta},
        "prompting": {"template": "[vqa] {question}{suffix}", "suffix": SUFFIX,
                      "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
                      "note": "paper's own prompt unpublished; this format is ours and disclosed"},
        "license": LICENSE_NOTE,
        "vram": {"gpu": torch.cuda.get_device_name(0),
                 "peak_allocated_gib_after_load": load_info["vram_after_load_bytes"] / 2**30,
                 "peak_allocated_gib_during_generation": peak / 2**30},
        "metrics": metrics,
        "paper_baselines": {"source": PAPER, "binary_table_3": paper["table_3"]["rows"],
                            "mcq_table_4": paper["table_4"]["rows"]},
        "predictions_file": pred_path.name,
    }
    out_json = out_dir / f"{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "vram": result["vram"]}, indent=2))
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
