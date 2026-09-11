"""Run TerraQ-VL's own train.py with fp16 mixed precision on a T4. Prepared, NOT run.

Why a launcher: TerraQ-VL's train.py builds ``Accelerator(mixed_precision="bf16" if bf16 else "no")``.
It has no fp16 path, and a T4 (Turing) has no bf16. So ``bf16: false`` alone would train in full
fp32. This launcher changes only that argument, to ``"fp16"``, which gives autocast fp16 plus
Accelerate's GradScaler. Their training loop, data pipeline and checkpointing run unchanged.

Guards, since fp16 is less forgiving than bf16:
- the config must say ``training.fp16: true`` and ``language_model.torch_dtype: float16``;
- every trainable parameter (connector + LoRA) must be fp32, or GradScaler cannot unscale it.
  The launcher refuses to start otherwise;
- ``--pilot-steps N`` stops after N optimizer steps and writes ``pilot_t4.json``: peak VRAM,
  samples/s, and whether any loss was non-finite. That file is the T4 VRAM measurement. Until it
  exists, every T4 memory figure is an estimate.

    TERRAQ_REPO=/path/to/terraq-vl@48f8d9b python reference/terraq_vl/train_t4.py \
        --config reference/terraq_vl/configs/finetune_ben_stage2_t4.yaml --pilot-steps 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot-steps", type=int, default=0, help="Stop after N optimizer steps (0 = full run).")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tcfg, lcfg = cfg["training"], cfg["language_model"]
    if not tcfg.get("fp16") or tcfg.get("bf16") or lcfg.get("torch_dtype") != "float16":
        raise SystemExit("config must set training.fp16: true, training.bf16: false, language_model.torch_dtype: float16")

    repo = os.environ.get("TERRAQ_REPO")
    if not repo or not os.path.isfile(os.path.join(repo, "train.py")):
        raise SystemExit("set TERRAQ_REPO to a clone of crimsonKn1ght/terraq-vl at 48f8d9b")
    sys.path.insert(0, repo)
    import train as terraq_train  # their entry point

    base_accelerator = terraq_train.Accelerator
    base_trainer = terraq_train.VLMTrainer
    accum = int(tcfg["gradient_accumulation_steps"])
    per_device = int(tcfg["per_device_batch_size"])

    def fp16_accelerator(*a, **kw):
        kw["mixed_precision"] = "fp16"
        acc = base_accelerator(*a, **kw)
        state = {"calls": 0, "t0": None, "nonfinite": 0}
        backward = acc.backward

        def counted_backward(loss, **bkw):
            if state["t0"] is None:
                torch.cuda.reset_peak_memory_stats()
                state["t0"] = time.time()
            if not math.isfinite(float(loss.detach())):
                state["nonfinite"] += 1
            backward(loss, **bkw)
            state["calls"] += 1
            if args.pilot_steps and state["calls"] >= args.pilot_steps * accum:
                secs = time.time() - state["t0"]
                out = {
                    "gpu": torch.cuda.get_device_name(0),
                    "optimizer_steps": args.pilot_steps,
                    "per_device_batch_size": per_device,
                    "gradient_accumulation_steps": accum,
                    "max_length": cfg["data"]["max_length"],
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "samples_per_s": state["calls"] * per_device / secs,
                    "nonfinite_losses": state["nonfinite"],
                    "grad_scaler_scale": float(acc.scaler.get_scale()) if acc.scaler else None,
                }
                path = os.path.join(tcfg["output_dir"], "pilot_t4.json")
                os.makedirs(tcfg["output_dir"], exist_ok=True)
                json.dump(out, open(path, "w"), indent=2)
                print(f"[pilot] {json.dumps(out)} -> {path}", flush=True)
                os._exit(0)

        acc.backward = counted_backward
        return acc

    class FP32TrainablesTrainer(base_trainer):
        def __init__(self, model, *a, **kw):
            bad = [(n, p.dtype) for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
            if bad:
                raise SystemExit(f"fp16 training needs fp32 trainable params; found {bad[:5]} ({len(bad)} total)")
            super().__init__(model, *a, **kw)

    terraq_train.Accelerator = fp16_accelerator
    terraq_train.VLMTrainer = FP32TrainablesTrainer
    sys.argv = ["train.py", "--config", args.config] + (["--resume", args.resume] if args.resume else [])
    terraq_train.main()


if __name__ == "__main__":
    main()
