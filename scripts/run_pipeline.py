"""Full end-to-end SatQuery AI pipeline runner.

Steps:
  1. Download datasets (VRSBench + RSVQA-LR)
  2. Show dataset images (save grid to outputs/)
  3. Prepare RS instruction data for LoRA
  4. Run LoRA adaptation
  5. Run VQA + captioning inference on all samples
  6. Evaluate VQA (RSVQA) and captioning (VRSBench)
  7. Save all results

Usage:
    python scripts/run_pipeline.py --hf-token TOKEN [--n-samples 10] [--skip-lora]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def step(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — full pipeline runner")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    parser.add_argument("--n-samples", type=int, default=10, help="Images to download and run inference on")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if data already exists")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA adaptation")
    parser.add_argument("--lora-steps", type=int, default=50, help="Max LoRA training steps")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    token = args.hf_token
    n = args.n_samples
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # STEP 1 — Download datasets
    # ─────────────────────────────────────────────────────────
    vrs_manifest = Path("data/vrsbench/sample/manifest.json")
    rsvqa_manifest = Path("data/rsvqa_lr/manifest.json")

    if not args.skip_download or not vrs_manifest.exists():
        step("STEP 1: Download datasets")
        from scripts.download_datasets import hf_login, save_vrsbench_subset, download_rsvqa_lr_metadata, write_official_links
        if token:
            hf_login(token)
        vrs_dir = Path("data/vrsbench/sample")
        save_vrsbench_subset(vrs_dir, "train", max(n, 20), token)
        download_rsvqa_lr_metadata(Path("data"), token)
        write_official_links(Path("data"))
        print("✅ Datasets downloaded.")
    else:
        step("STEP 1: Skipping download (data exists)")

    # ─────────────────────────────────────────────────────────
    # STEP 2 — Show dataset images
    # ─────────────────────────────────────────────────────────
    step("STEP 2: Show dataset images")
    preview_path = str(out_dir / "dataset_preview.png")
    from scripts.show_dataset_images import load_manifest, make_grid
    try:
        records = load_manifest(str(vrs_manifest))
        make_grid(records, n, preview_path, title="VRSBench — Remote Sensing Dataset Samples")
        print(f"✅ Dataset preview saved → {preview_path}")
    except Exception as exc:
        print(f"[warn] Could not generate preview: {exc}")

    # ─────────────────────────────────────────────────────────
    # STEP 3 — Prepare RS instructions
    # ─────────────────────────────────────────────────────────
    step("STEP 3: Prepare RS instruction data")
    instructions_path = "data/rs_instructions/instructions.json"
    try:
        from satquery.adaptation.prepare_instructions import (
            download_rs_instructions,
            build_synthetic_from_vrsbench,
        )
        Path("data/rs_instructions").mkdir(parents=True, exist_ok=True)
        records_instr = download_rs_instructions(Path("data/rs_instructions"), min(n * 5, 200), token)
        if not records_instr:
            records_instr = build_synthetic_from_vrsbench(str(vrs_manifest), min(n * 5, 200))
        Path(instructions_path).write_text(json.dumps(records_instr, indent=2))
        print(f"✅ {len(records_instr)} instruction records prepared.")
    except Exception as exc:
        print(f"[warn] Instruction prep failed: {exc}")

    # ─────────────────────────────────────────────────────────
    # STEP 4 — LoRA Adaptation
    # ─────────────────────────────────────────────────────────
    lora_path = "checkpoints/rs_vlm_lora"
    if not args.skip_lora and Path(instructions_path).exists():
        step("STEP 4: LoRA adaptation of BLIP-2")
        try:
            from satquery.adaptation.train_lora import train
            train(
                instructions_path=instructions_path,
                out_dir=lora_path,
                epochs=1,
                batch_size=1,
                lr=3e-4,
                max_steps=args.lora_steps,
                token=token,
            )
            print(f"✅ LoRA adapter saved → {lora_path}")
        except Exception as exc:
            print(f"[warn] LoRA training failed: {exc}")
            lora_path = ""
    else:
        step("STEP 4: Skipping LoRA adaptation")
        if Path(lora_path).exists():
            print(f"  Using existing adapter: {lora_path}")
        else:
            lora_path = ""

    # ─────────────────────────────────────────────────────────
    # STEP 5 — Run end-to-end inference on all samples
    # ─────────────────────────────────────────────────────────
    step("STEP 5: End-to-end inference (VQA + Captioning)")
    from satquery.single_image_api import analyze_single_image

    vrs_records = load_manifest(str(vrs_manifest))[:n]
    all_results = []

    demo_questions = [
        "What type of land cover is visible in this image?",
        "How many distinct land cover classes can you identify?",
        "Are there any water bodies in this image?",
        "What is the dominant feature in this scene?",
        "Describe the spatial arrangement of features in this image.",
    ]

    for i, rec in enumerate(vrs_records):
        img_path = rec.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue

        # VQA
        question = demo_questions[i % len(demo_questions)]
        print(f"\n  [{i+1}/{len(vrs_records)}] VQA: {question[:50]}…")
        vqa_result = analyze_single_image(img_path, question)
        print(f"    → {vqa_result.get('answer', '[error]')[:100]}")

        # Captioning
        print(f"  [{i+1}/{len(vrs_records)}] Captioning …")
        cap_result = analyze_single_image(img_path, "Describe this image.")
        print(f"    → {cap_result.get('caption', '[error]')[:100]}")

        all_results.append({
            "image": img_path,
            "vqa": vqa_result,
            "captioning": cap_result,
            "ground_truth_caption": rec.get("caption", ""),
        })

    results_path = out_dir / "pipeline_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n✅ Inference complete. Results → {results_path}")

    # ─────────────────────────────────────────────────────────
    # STEP 6 — Evaluate
    # ─────────────────────────────────────────────────────────
    step("STEP 6: Evaluation")

    # VQA eval
    if rsvqa_manifest.exists():
        try:
            from satquery.evaluation.evaluate_rsvqa import evaluate as eval_rsvqa
            eval_rsvqa(str(rsvqa_manifest), str(out_dir / "eval_rsvqa.json"), n, token, lora_path)
        except Exception as exc:
            print(f"[warn] RSVQA eval failed: {exc}")
    else:
        print("[warn] RSVQA manifest not found — skipping VQA eval")

    # Captioning eval on VRSBench
    try:
        from satquery.evaluation.evaluate_vrsbench import evaluate as eval_vrsbench
        eval_vrsbench(str(vrs_manifest), str(out_dir / "eval_vrsbench.json"), n, lora_path)
    except Exception as exc:
        print(f"[warn] VRSBench eval failed: {exc}")

    # ─────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────
    step("PIPELINE COMPLETE ✅")
    print(f"""
  Outputs:
    📊 Dataset preview  : {preview_path}
    📝 Inference results: {results_path}
    📈 VQA eval         : {out_dir}/eval_rsvqa.json
    📈 Caption eval     : {out_dir}/eval_vrsbench.json
    🤖 LoRA adapter     : {lora_path or '(not trained)'}
""")


if __name__ == "__main__":
    main()
