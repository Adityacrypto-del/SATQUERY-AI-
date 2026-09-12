"""Write outputs/reference_eval/eval_log.md from whatever TerraQ-VL reference results exist.

Everything reported is TerraQ-VL (third-party, VRSBench-adapted): a reference baseline, not Branch
1's own result. Runs that have not finished are shown as NOT FINISHED, never as numbers.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs" / "reference_eval"
LABEL = "TerraQ-VL (third-party, VRSBench-adapted)"


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jsonl(name):
    p = OUT / name
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}"


def track_a(lines):
    a = load("terraq_vl_stage2_vrsbench.json")
    n_pred = len(jsonl("terraq_vl_stage2_vrsbench_predictions.jsonl"))
    lines += ["## Track A: VRSBench held-out test (in-domain for TerraQ-VL)", ""]
    if not a or not a["split"]["full_split"]:
        lines += [f"**NOT FINISHED:** {n_pred}/1367 predictions written. The run resumes from where it stopped.", ""]
        return
    m, r = a["metrics"], a["cross_check"]["reference_metrics_same_harness"]
    boot = a["cross_check"]["paired_bootstrap_ours_minus_reference"] or {}
    lines += [
        f"- Model: {LABEL}, `{a['checkpoint']}`; 4-bit NF4 LLM (the reference predictions are TerraQ-VL's own bf16 run)",
        f"- Split: `{a['split']['file']}` sha256 `{a['split']['sha256'][:16]}...`, {a['split']['records']} records / {a['split']['images']} images. {a['split']['disjointness']}",
        f"- VRAM: {a['vram']['peak_allocated_gib_during_generation']:.2f} GiB peak on {a['vram']['gpu']}",
        f"- Timestamp: {a['timestamp_utc']}",
        "",
        "| task | metric | ours (4-bit) | TerraQ-VL published preds, same harness | 95% CI of diff (ours - ref) | CI contains 0 |",
        "|---|---|---|---|---|---|",
    ]
    for task, keys in {"caption": ["bleu4", "rougeL"], "vqa": ["harness_match", "strict_em"],
                       "refer": ["acc_iou50", "acc_iou70"]}.items():
        for k in keys:
            b = boot.get(f"{task}.{k}", {})
            ci = f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]" if b else "n/a"
            lines.append(f"| {task} (n={m[task]['n']}) | {k} | {m[task][k]:.4f} | {r[task][k]:.4f} | {ci} | {b.get('contains_zero', 'n/a')} |")
    lines += [
        "",
        f"Exact response agreement with the published predictions: {pct(a['cross_check']['exact_response_agreement'])}%. "
        f"Refer outputs unparseable: ours {m['refer']['unparseable']}, reference {r['refer']['unparseable']}.",
        "",
        "Cross-check reading: every CI containing 0 means the reproduction matches TerraQ-VL's own run "
        "within resampling noise, so the harness is trustworthy. A CI excluding 0 is a systematic "
        "difference. The likely causes are 4-bit NF4 vs bf16, and transformers 5.x applying Qwen's "
        "repetition_penalty only to new tokens under inputs_embeds. It is reported as a difference, "
        "not corrected away.",
        "",
    ]


def track_b(lines):
    b = load("terraq_vl_stage2_bigearthnet.json")
    preds = jsonl("terraq_vl_stage2_bigearthnet_predictions.jsonl")
    lines += ["## Track B: BigEarthNet.txt bench split (out-of-domain for TerraQ-VL)", ""]
    if not b or not b["split"]["full_split"]:
        lines += [f"**NOT FINISHED:** {len(preds)}/12477 predictions written. The run resumes from where it stopped.", ""]
        return
    m = b["metrics"]
    lines += [
        f"- Model: {LABEL}, `{b['checkpoint']}`, zero-shot. {b['domain_note']}",
        f"- Split: {b['split']['dataset']} `bench`, {b['split']['records']} binary+MCQ records / {b['split']['image_pairs']} image pairs. {b['split']['disjointness']}",
        f"- Input: {b['input']['bands']}. Prompt: `{b['prompting']['template']}` (our format, disclosed; the paper's prompt is unpublished)",
        f"- VRAM: {b['vram']['peak_allocated_gib_during_generation']:.2f} GiB peak. Timestamp: {b['timestamp_utc']}",
        "",
    ]
    paper = b["paper_baselines"]
    for ty, table, cols in (("binary", paper["binary_table_3"], ["presence", "area", "count", "adjacency"]),
                            ("mcq", paper["mcq_table_4"], ["presence", "area", "count", "adjacency", "relative pos",
                                                           "country", "season", "climate zone"])):
        mm = m[ty]
        pc = mm["per_category_accuracy"]
        head = table[0]
        lines += [f"### {ty} (n={mm['n']}, extraction failures {mm['extraction_failures']}, IF {'yes' if mm['IF_all_extracted'] else 'no'})", "",
                  "| model | " + " | ".join(head[1:]) + " |", "|---" * len(head) + "|",
                  f"| **{LABEL}** | " + " | ".join(pct(pc.get(c)) for c in cols) + f" | **{pct(mm['overall_accuracy'])}** | {'yes' if mm['IF_all_extracted'] else 'no'} |"]
        lines += ["| " + " | ".join(row) + " |" for row in table[1:]]
        dist = Counter(p["extracted"] for p in preds if p["type"] == ty)
        gold = Counter(str(p["reference"]).strip().lower() for p in preds if p["type"] == ty)
        tot = sum(dist.values())
        lines += ["", f"Answer distribution (predicted vs gold): " + ", ".join(
            f"`{k}` {100 * dist[k] / tot:.1f}% vs {100 * gold[k] / tot:.1f}%" for k in sorted(set(dist) | set(gold), key=str)),
            "A heavily skewed predicted distribution means accuracy partly reflects answer bias, not image understanding.", ""]
    lines += [f"Paper rows: {paper['source']}. OA = overall accuracy; IF = an answer could always be extracted.", ""]


def main():
    lines = [
        "# Branch 1 reference evaluation log",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `reference/terraq_vl/summarize_eval.py`.",
        "",
        f"> Every number below is **{LABEL}**, a third-party reference baseline. None of it is Branch 1's own",
        "> adaptation. Branch 1's BigEarthNet-adapted model: **NOT_YET_MEASURED** (cloud job, config",
        "> `reference/terraq_vl/configs/finetune_ben_stage2_t4.yaml`).",
        "",
        "License: TerraQ-VL weights are research / non-commercial (Qwen Research License + VRSBench CC-BY-NC-4.0); "
        "BigEarthNet.txt is CDLA-Permissive-1.0; the BigEarthNet v2.0 patches came from the unofficial "
        "hackelle/BigEarthNetV2-LMDB mirror.",
        "",
    ]
    track_a(lines)
    track_b(lines)
    lines += [
        "## Decisions on record (2026-09-12)",
        "- Architecture (final): the TerraQ-VL recipe on BigEarthNet data, not Qwen2-VL-7B / train_lora.py.",
        "- Track D target: a T4 with fp16 (via train_t4.py), batch 1 x 64, max_length 570 (measured maximum). "
        "T4 VRAM is NOT_YET_MEASURED; the `--pilot-steps 50` run measures it.",
        "- The overlap guard passed on real data: 78,974 train records / 5,000 patches vs 234,930 eval patches, 0 overlaps.",
        "- Open for Peek: Track C imports indices.py from the bi-temporal branch. Options: merge Branch 2, add a shared package, or keep the env-var path.",
        "- Git: commits 23606cf, 6ef8cdf, 06d362b on origin/peek/branch1-baseline. No PR to ayush yet. These results are uncommitted pending review.",
        "",
    ]
    path = OUT / "eval_log.md"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
