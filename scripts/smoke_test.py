"""Stub controller: (t1, t2, query) -> pipeline -> ChangeResult + trace.

Qualitative sanity checks, not scored. The gate this serves is: does the
pipeline run end to end without crashing, and does it refuse to invent
values the input cannot support (no area in m2 on ungeoreferenced input,
no class label where no rule applies)?
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from tools.change_analysis.io import load_rsimage
from tools.change_analysis.pipeline import (
    BiTemporalPipeline, PipelineConfig, export_report,
)


def run_case(case_id, t1_path, t2_path, query, modality="optical",
             checkpoint=None, out_dir="outputs/smoke", shift_px=0,
             device=None):
    t1 = load_rsimage(t1_path, modality=modality)
    t2 = load_rsimage(t2_path, modality=modality)

    if shift_px:
        # Roll t2 to induce a known misregistration, so pair.py's
        # co-registration check has something real to fire on.
        import numpy as np
        from tools.change_analysis.io import RSImage
        t2 = RSImage(
            array=np.roll(t2.array, shift_px, axis=2),
            crs=t2.crs, transform=t2.transform, modality=t2.modality,
            band_names=t2.band_names, gsd_m=t2.gsd_m,
            _pixel_size_m=t2._pixel_size_m, _gsd_source=t2._gsd_source,
        )

    config = PipelineConfig(
        checkpoint=checkpoint,
        overlay_dir=os.path.join(out_dir, "overlays"),
        device=device,
    )
    result = BiTemporalPipeline(config).run(t1, t2, query, scene_id=case_id)
    path = export_report(result, os.path.join(out_dir, f"{case_id}.json"))

    extras = getattr(result, "evidence_extras", {})
    return {
        "case": case_id,
        "query": query,
        "t1": os.path.basename(t1_path),
        "t2": os.path.basename(t2_path),
        "shift_px": shift_px,
        "route": extras.get("route"),
        "question_type": extras.get("question_type"),
        "answer": result.answer,
        "changed": result.changed,
        "change_type": result.change_type,
        "confidence": result.confidence,
        "confidence_basis": extras.get("confidence_basis"),
        "changed_area_pixels": result.changed_area_pixels,
        "changed_area_m2": result.changed_area_m2,
        "n_regions": len(result.regions),
        "georeferenced": result.georeferencing.get("georeferenced"),
        "validation_ok": (extras.get("validation") or {}).get("ok"),
        "coregistration": next(
            (c["detail"] for c in (extras.get("validation") or {}).get("checks", [])
             if c["name"] == "co_registration"), None),
        "n_trace_stages": len(result.trace),
        "summary": result.summary,
        "report": path,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default="outputs/smoke")
    p.add_argument("--second-only", action="store_true")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--device", default=None,
                   help="force cpu to avoid contending with a training run")
    p.add_argument("--results-name", default="smoke_results.json")
    args = p.parse_args(argv)

    fx = "outputs/smoke/fixtures"
    safe = json.load(open("outputs/spare_scenes_safe.json"))["safe_scenes"]
    spare = "datasets/SECOND_raw/test_extract/test"
    queries = {q["id"]: q["text"]
               for q in json.load(open("outputs/smoke/queries.json"))["queries"]}

    cases = []
    if not args.second_only:
        cases += [
            dict(case_id="C1_synth_optical_multispectral",
                 t1_path=f"{fx}/SYNTHETIC_optical_t1.tif",
                 t2_path=f"{fx}/SYNTHETIC_optical_t2.tif",
                 query=queries["q4_open_ended"], modality="optical"),
            dict(case_id="C2_synth_sar_ratio",
                 t1_path=f"{fx}/SYNTHETIC_sar_t1.tif",
                 t2_path=f"{fx}/SYNTHETIC_sar_t2.tif",
                 query=queries["q4_open_ended"], modality="sar"),
            dict(case_id="C3_synth_optical_shifted",
                 t1_path=f"{fx}/SYNTHETIC_optical_t1.tif",
                 t2_path=f"{fx}/SYNTHETIC_optical_t2.tif",
                 query=queries["q6_scene_ratio"], modality="optical", shift_px=4),
        ]
    if not args.synthetic_only:
        cases += [
            dict(case_id="C4_second_held_out_change_or_not",
                 t1_path=f"{spare}/im1/{safe[0]}", t2_path=f"{spare}/im2/{safe[0]}",
                 query=queries["q1_change_or_not"], checkpoint=args.checkpoint),
            dict(case_id="C5_second_held_out_change_to_what",
                 t1_path=f"{spare}/im1/{safe[1]}", t2_path=f"{spare}/im2/{safe[1]}",
                 query=queries["q2_change_to_what"], checkpoint=args.checkpoint),
            dict(case_id="C6_second_held_out_absent_class",
                 t1_path=f"{spare}/im1/{safe[2]}", t2_path=f"{spare}/im2/{safe[2]}",
                 query=queries["q5_absent_class"], checkpoint=args.checkpoint),
        ]

    results = []
    for case in cases:
        case.setdefault("checkpoint", args.checkpoint)
        case["out_dir"] = args.out_dir
        case.setdefault("device", args.device)
        try:
            results.append(run_case(**case))
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - smoke test must report, not raise
            results.append({"case": case["case_id"], "CRASHED": f"{type(exc).__name__}: {exc}"})
            status = "CRASH"
        print(f"  {case['case_id']:<40} {status}", flush=True)

    out = os.path.join(args.out_dir, args.results_name)
    with open(out, "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2, default=str)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
