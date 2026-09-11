# TerraQ-VL — vendored third-party reference model

**This is not Branch 1's model.** Label it "TerraQ-VL (third-party, VRSBench-adapted)" everywhere
its numbers appear. Branch 1's own adaptation (BigEarthNet) is a separate result, and it is
`NOT_YET_MEASURED`.

| | |
|---|---|
| Code | https://github.com/crimsonKn1ght/terraq-vl @ `48f8d9b88559aeacec324d3aa212e91101dba887` (the commit named in the Stage-2 model card), MIT |
| Weights | https://huggingface.co/grKnight/terraq-vl (DOI 10.57967/hf/9584): `stage-2/checkpoints/checkpoint-2180` (connector + LoRA); fallback `stage-1/checkpoints/checkpoint-3270` (connector only) |
| Architecture | CLIP ViT-L/14 (frozen, penultimate layer, 256 patches) → 2-layer MLP connector 1024→2048→2048 → Qwen2.5-3B-Instruct (frozen) + LoRA r=16/α=32 on q,k,v,o,gate,up,down |
| Weights license | **Research / non-commercial, attribution required.** Qwen Research License (Qwen2.5-3B-Instruct base; it also forbids using outputs to improve any non-Qwen LLM) + VRSBench CC-BY-NC-4.0 (training data; imagery from DOTA-v2/DIOR under their academic terms). |

## Files copied verbatim (not modified)

| file | sha256 |
|---|---|
| `vlm_model/__init__.py` (empty) | `e3b0c442…b855` |
| `vlm_model/connector.py` | `fa66dc0d…e23b` |
| `vlm_model/language_model.py` | `5c008e95…13a2` |
| `vlm_model/utils.py` | `cdd34d77…a076` |
| `vlm_model/vision_encoder.py` | `6ae1fcb7…5320` |
| `vlm_model/vlm.py` | `b174d851…a272` |
| `inference.py` | `60c0563f…7c37277` |
| `training/checkpoint.py` | `4734172a…3ceb273` |
| `data/image_processing.py` | `370992d4…9e66` |
| `scripts/generate_heldout_records.py` | as of the pinned commit |
| `configs/finetune_vrsbench_stage2.yaml`, `configs/pretrain_vrsbench.yaml` | as of the pinned commit |

`training/__init__.py`, `data/__init__.py` and `scripts/__init__.py` are empty package markers
added here. Only the modules above are vendored, not the rest of `training/` or `data/`.

## Branch 1 code in this directory

- `load_4bit.py`: builds the model through their `VLMForCausalLM`, but loads the LLM in 4-bit NF4
  (bitsandbytes) with `device_map`, so it fits a 6 GB GPU. That is the only deviation from their
  `inference.load_vlm`. The connector, LoRA, vision tower, prompt template and decoding are all theirs.
- `metrics.py`: scores predictions with Branch 1's harness metrics (see the module docstring).
- `run_vrsbench_eval.py`: Track A, on their published held-out `test.json` only.
- `run_bigearthnet_eval.py`: Track B, the BigEarthNet.txt `bench` split, RGB render only.
- `export_ben_llava.py`, `configs/finetune_ben_stage2_t4.yaml`, `train_t4.py`: Track D, prepared
  but **not run**. `train_t4.py` runs their unchanged `train.py` with fp16 mixed precision (their
  code only supports bf16 or fp32) and has a `--pilot-steps` VRAM measurement.
- `data_access/`: HTTP range readers used to pull just the needed files. The VRSBench test images
  come from `Images_train.zip`, and the BigEarthNet bench patches from the `hackelle/BigEarthNetV2-LMDB`
  mirror (unofficial; the official files at zenodo.org/records/10891137 take precedence).
