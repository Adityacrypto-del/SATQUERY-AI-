# SatQuery AI — Single-Image Remote-Sensing Assistant

SatQuery AI is a focused single-satellite-image module for a larger geospatial
assistant. Given exactly one PNG, JPEG, TIFF, or GeoTIFF and a natural-language
query, it either answers a visual question or creates a scene description.

## What this branch delivers

- Deterministic query routing: descriptions go to captioning; other prompts go
  to VQA.
- Strict single-image validation with PNG/JPEG/TIFF/GeoTIFF decoding.
- GeoTIFF-aware preprocessing that preserves source metadata and explicitly
  records the RGB band representation sent to the vision-language model.
- BLIP-2 inference wrapper, with optional PEFT/LoRA remote-sensing adapter
  loading.
- Small-data preparation, LoRA adaptation, RSVQA evaluation, VRSBench caption
  evaluation, and an end-to-end runner.

The base model is deliberately identified as **not adapted** until an actual
LoRA checkpoint is supplied. It must not be presented as a remote-sensing model
without that artifact.

## Quick start

```bash
cd SATQUERY-AI-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fast structural checks; they do not download model weights.
pytest -q

# Download small evaluation/development subsets (HF_TOKEN may be required).
python scripts/download_datasets.py --hf-token "$HF_TOKEN" --vrsbench-n 20

# Optional: prepare remote-sensing instructions and train a small LoRA run.
python -m satquery.adaptation.prepare_instructions --n 100
python -m satquery.adaptation.train_lora --max-steps 50

# Run either task on one image.
python app/single_image_ui.py --image path/to/image.tif --query "Describe this image."
python app/single_image_ui.py --image path/to/image.jpg --query "How many buildings are visible?"
```

For a compact end-to-end experiment, use:

```bash
python scripts/run_pipeline.py --hf-token "$HF_TOKEN" --n-samples 10 --lora-steps 50
```

## Required before a final demo

1. A reachable Hugging Face token (when the selected data/model requires one).
2. A small, documented remote-sensing instruction subset and the resulting
   LoRA checkpoint under `checkpoints/rs_vlm_lora/`.
3. Saved evaluation output from both RSVQA (VQA) and VRSBench (captioning).
4. A few representative demo images, including at least one GeoTIFF if that
   format is in scope.

## Fine-tuning status

The single-image pipeline is ready for inference, preprocessing, and
evaluation, but **remote-sensing fine-tuning is still required on this
branch** before presenting model quality as domain-adapted. Run the LoRA
preparation/training commands above on the selected RS-instruction or
BigEarthNet subset, save the adapter under `checkpoints/rs_vlm_lora/`, and
evaluate the adapted checkpoint against the base model.

See [steps_to_follow.md](steps_to_follow.md) for the original implementation
requirements and dataset references.
