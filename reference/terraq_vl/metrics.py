"""Scoring for the TerraQ-VL reference runs, built on Branch 1's existing harness metrics.

- caption: BLEU-4 (corpus) + ROUGE-L -- ``satquery.evaluation.evaluate_vrsbench.compute_metrics``.
- vqa: two accuracies, both reported and labelled:
    ``harness_match`` = Branch 1's ``evaluate_rsvqa`` rule (lower/strip, then equal OR gold
    contained in prediction -- lenient); ``strict_em`` = equality after VQA normalisation
    (lowercase, punctuation and articles removed).
- refer: VRSBench boxes ``{<x1><y1><x2><y2>}`` on a 0-100 grid; Acc@IoU>=0.5 and >=0.7, the
  VRSBench paper's grounding metrics. Unparseable predictions count as misses and are counted.

Paired bootstrap over images (records of one image move together) gives the 95% CI of
(ours - reference) for the reproduction cross-check.
"""
from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from satquery.evaluation.evaluate_rsvqa import normalize as harness_normalize
from satquery.evaluation.evaluate_vrsbench import compute_metrics

_ARTICLES = {"a", "an", "the"}
_BOX = re.compile(r"<\s*(\d+(?:\.\d+)?)\s*>")


def vqa_normalize(s: str) -> str:
    s = s.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def harness_match(pred: str, gold: str) -> bool:
    p, g = harness_normalize(pred), harness_normalize(gold)
    return p == g or g in p


def strict_em(pred: str, gold: str) -> bool:
    return vqa_normalize(pred) == vqa_normalize(gold)


def parse_box(text: str) -> Optional[List[float]]:
    nums = [float(x) for x in _BOX.findall(text or "")]
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def iou(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def task_of(prompt: str) -> str:
    m = re.match(r"\s*\[(\w+)\]", prompt or "")
    return m.group(1).lower() if m else "unknown"


def score(rows: List[dict]) -> Dict[str, dict]:
    """rows: dicts with prompt, reference, response. Returns metrics per task."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[task_of(r["prompt"])].append(r)

    out: Dict[str, dict] = {}
    if by_task.get("caption"):
        cap = by_task["caption"]
        m = compute_metrics([r["response"] for r in cap], [r["reference"] for r in cap])
        out["caption"] = {"n": len(cap), "bleu4": m.get("bleu4"), "rougeL": m.get("rougeL")}
    if by_task.get("vqa"):
        v = by_task["vqa"]
        out["vqa"] = {
            "n": len(v),
            "harness_match": sum(harness_match(r["response"], r["reference"]) for r in v) / len(v),
            "strict_em": sum(strict_em(r["response"], r["reference"]) for r in v) / len(v),
        }
    if by_task.get("refer"):
        rf = by_task["refer"]
        ious, unparsed = [], 0
        for r in rf:
            p, g = parse_box(r["response"]), parse_box(r["reference"])
            if p is None or g is None:
                unparsed += 1
                ious.append(0.0)
            else:
                ious.append(iou(p, g))
        out["refer"] = {
            "n": len(rf),
            "acc_iou50": sum(i >= 0.5 for i in ious) / len(rf),
            "acc_iou70": sum(i >= 0.7 for i in ious) / len(rf),
            "mean_iou": sum(ious) / len(rf),
            "unparseable": unparsed,
        }
    return out


PRIMARY = {
    "caption": ["bleu4", "rougeL"],
    "vqa": ["harness_match", "strict_em"],
    "refer": ["acc_iou50", "acc_iou70"],
}


def paired_bootstrap(ours: List[dict], ref: List[dict], n_boot: int = 1000, seed: int = 0) -> Dict[str, dict]:
    """95% CI of (ours - ref) per primary metric, resampling images with replacement."""
    by_img = defaultdict(list)
    for a, b in zip(ours, ref):
        by_img[a["image"]].append((a, b))
    images = sorted(by_img)
    rng = random.Random(seed)
    diffs = defaultdict(list)
    for _ in range(n_boot):
        pick = [rng.choice(images) for _ in images]
        oa = [a for img in pick for a, _ in by_img[img]]
        rb = [b for img in pick for _, b in by_img[img]]
        so, sr = score(oa), score(rb)
        for task, keys in PRIMARY.items():
            for k in keys:
                if task in so and task in sr and so[task][k] is not None and sr[task][k] is not None:
                    diffs[f"{task}.{k}"].append(so[task][k] - sr[task][k])
    out = {}
    for name, d in diffs.items():
        d.sort()
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        out[name] = {"ci95_low": lo, "ci95_high": hi, "contains_zero": lo <= 0.0 <= hi}
    return out
