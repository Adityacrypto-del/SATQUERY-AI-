# SatQuery AI — Single-Image Pipeline Implementation Guide

## Purpose

This document is for the GitHub coding agent implementing the **single-image component** of SatQuery AI.

The single-image component must support:

1. **Visual Question Answering (VQA)** — mandatory.
2. **One additional single-image task** — use **image captioning / scene description**.
3. **Remote-sensing adaptation** of at least one vision/vision-language component.
4. A clean interface so the main SatQuery agent can later route user queries to this component.

Do NOT implement bi-temporal change detection, optical-SAR fusion, earthquake analysis, live satellite retrieval, or other unrelated features in this module.

---

# 1. Official / Primary Resources

## 1.1 BigEarthNet

Official website:

https://bigearth.net/

BigEarthNet v2.0 contains 549,488 paired Sentinel-1/Sentinel-2 image patches. The official page currently lists approximately 59 GiB for BigEarthNet-S2 and approximately 51 GiB for BigEarthNet-S1.

For this 2-day implementation:

- Prefer **BigEarthNet-S2** for optical/multispectral adaptation.
- Do NOT download the entire dataset unless storage/time permits.
- Download/use a manageable subset for development.
- Preserve the official metadata and labels.

Official page:
https://bigearth.net/

BigEarthNet v2.0 S2 download is linked from:
https://bigearth.net/

The official page also provides:
- dataset description
- metadata.parquet
- reference maps
- pretrained S1/S2 models
- BigEarthNet processing code

IMPORTANT:
BigEarthNet is primarily a remote-sensing image/land-cover benchmark, not a conventional VQA dataset. Use it for remote-sensing domain adaptation/representation learning, not as the only VQA dataset.

---

# 2. RSVQA — Mandatory VQA Dataset

## 2.1 Official RSVQA project

Original repository:

https://github.com/syvlo/RSVQA

The user's supplied fork:

https://github.com/kaaydin/vqa-remote-sensing

The supplied fork contains a VQA implementation and instructions for the high-resolution dataset.

Use the supplied fork mainly as a reference/baseline if useful. Do not blindly copy its old architecture into the final SatQuery architecture.

---

## 2.2 RSVQA-HR

Dataset DOI:

https://doi.org/10.5281/zenodo.6344366

Direct Zenodo record commonly used for the dataset:

https://zenodo.org/record/6344367

Wageningen dataset page:

https://research.wur.nl/en/datasets/remote-sensing-vqa-high-resolution-rsvqa-hr/

RSVQA-HR contains high-resolution aerial RGB images and natural-language question/answer pairs.

For the first implementation, use RSVQA-HR if sufficient GPU/storage is available.

---

## 2.3 RSVQA-LR

Dataset DOI:

https://doi.org/10.5281/zenodo.6344333

The dataset is the lower-resolution Sentinel-2 version.

RSVQA-LR is much smaller and is recommended for fast local/Colab prototyping.

A convenient PyTorch dataset implementation is available through:

https://github.com/isaaccorley/torchrs

The torchrs documentation describes RSVQA-LR as 772 256x256 Sentinel-2 RGB images with associated questions and answers.

---

# 3. VRSBench — Captioning + VQA + Grounding

Official project:

https://vrsbench.github.io/

Official GitHub:

https://github.com/lx709/VRSBench

Official Hugging Face dataset:

https://huggingface.co/datasets/xiang709/VRSBench

VRSBench contains:

- 29,614 remote-sensing images
- 29,614 human-verified detailed captions
- 52,472 object references
- 123,221 question-answer pairs

It supports:

- image captioning
- visual grounding
- visual question answering

For this project, use **VRSBench primarily for captioning evaluation/development**. It can also provide an additional VQA evaluation source.

The Hugging Face dataset can be loaded with:

```python
from datasets import load_dataset

dataset = load_dataset("xiang709/VRSBench")
```

The full Hugging Face dataset is large. Do not download the entire dataset unless required. Use the relevant evaluation files/splits or streaming where possible.

---

# 4. Recommended Model

## Primary recommendation: RS-LLaVA

Repository:

https://github.com/BigData-KSU/RS-LLaVA

RS-LLaVA is specifically designed for remote-sensing vision-language tasks and supports joint captioning and question answering.

The repository provides an RS-instruction dataset and released model/code.

RS-instruction dataset:
- combines remote-sensing captioning and VQA data
- includes RSVQA-LR and RSIVQA-DOTA among its sources
- has a train/test split
- is suitable for remote-sensing VLM adaptation

Use this model rather than building a VQA model from scratch.

---

# 5. Alternative Model

## GeoChat

GeoChat is another remote-sensing VLM option.

Use it only if the environment/checkpoint is easier to run than RS-LLaVA.

Do NOT spend hours comparing many VLMs.

Priority:

1. RS-LLaVA if it runs reliably.
2. GeoChat if RS-LLaVA setup becomes problematic.
3. A conventional RSVQA model only as a fallback baseline.

The objective is a working remote-sensing VLM pipeline, not a model-comparison paper.

---

# 6. Why the old RSVQA repository should not be the final architecture

The repository:

https://github.com/kaaydin/vqa-remote-sensing

is useful because it provides a working RSVQA implementation and training/evaluation structure.

However, its architecture is a conventional VQA system using image/text feature extractors and fusion.

SatQuery AI requires a more general vision-language component that can support:

- VQA
- captioning
- natural-language interaction
- later agentic routing

Therefore:

- use the repository as a reference/baseline;
- do not make the entire SatQuery system dependent on its old architecture;
- prefer a modern remote-sensing VLM such as RS-LLaVA.

---

# 7. Final Single-Image Architecture

Implement:

```text
                    USER
                     |
                     v
              SINGLE IMAGE
                     |
                     v
             Input Validator
                     |
                     v
              Query Classifier
                     |
              +------+------+
              |             |
              v             v
             VQA        CAPTIONING
              |             |
              v             v
          RS-LLaVA       RS-LLaVA
       / adapted VLM   / adapted VLM
              |             |
              +------+------+
                     |
                     v
             Evidence / Result
                     |
                     v
              Execution Trace
                     |
                     v
              Final Response
```

---

# 8. Query Routing

Create a simple deterministic router first.

Examples:

### Query

"What type of land cover is present?"

Route:

```text
task = VQA
```

### Query

"How many buildings are visible?"

Route:

```text
task = VQA
```

### Query

"Describe this image."

Route:

```text
task = CAPTIONING
```

### Query

"Give me a detailed scene description."

Route:

```text
task = CAPTIONING
```

Do NOT use a large LLM just to classify these two tasks initially.

A small rule-based classifier is sufficient.

Later, the main SatQuery agent can replace the router with its own agentic controller.

---

# 9. Input Validation

Create:

```text
validate_single_image()
```

It must check:

- file exists
- supported extension
- image can be decoded
- image has valid dimensions
- TIFF/GeoTIFF support
- PNG/JPEG support for benchmark datasets
- number of images == 1
- image modality if metadata is available

Supported extensions:

```text
.tif
.tiff
.png
.jpg
.jpeg
```

For GeoTIFF:

- preserve geospatial metadata where possible
- use rasterio/GDAL for reading
- convert the image into the model's expected tensor/image format
- do not silently discard all bands

For ordinary RGB benchmark images, use PIL where appropriate.

---

# 10. Multispectral Handling

BigEarthNet-S2 is multispectral.

Do NOT assume every model can consume all Sentinel-2 bands.

Implement an explicit preprocessing layer:

```text
GeoTIFF / Sentinel-2
        |
        v
Band inspection
        |
        v
Model-compatible band selection
        |
        v
Normalization
        |
        v
RGB/model tensor
```

The selected bands and preprocessing configuration must be recorded in the execution trace.

Example:

```json
{
  "bands_used": ["B04", "B03", "B02"],
  "resize": "model_default",
  "normalization": "model_default"
}
```

If the chosen VLM only supports RGB input, clearly document that it is using an RGB-compatible representation of the multispectral image.

Do NOT pretend that an RGB-only VLM is directly processing all multispectral channels.

---

# 11. Remote-Sensing Adaptation

This is mandatory.

The final system must not simply call a generic VLM and claim it is remote-sensing adapted.

Preferred implementation:

```text
Base VLM
   |
   +---- frozen base parameters
   |
   +---- LoRA / PEFT adapters
                |
                v
       Remote-sensing adaptation
                |
                v
        Adapted RS-VLM
```

Use PEFT/LoRA where practical.

Recommended training sources:

### Option A — BigEarthNet

Use BigEarthNet-S2 labels/metadata to expose the model to remote-sensing land-cover concepts.

### Option B — Open remote-sensing instruction data

Use the RS-LLaVA RS-instruction dataset:

https://github.com/BigData-KSU/RS-LLaVA

This is strongly recommended for fast VLM adaptation because it is already instruction-formatted.

### Option C — Both

Best final story:

```text
BigEarthNet
     +
RS-instruction data
     |
     v
Remote-sensing adaptation
     |
     v
RS-LLaVA / adapted VLM
```

Because of the 2-day deadline, prioritize getting the RS-instruction/LoRA pipeline working rather than attempting full-scale training on all of BigEarthNet.

---

# 12. BigEarthNet Practical Strategy

Do NOT attempt to train on all 549,488 patches during this 2-day sprint.

Instead:

1. Download/access a manageable subset.
2. Inspect metadata.
3. Select patches with useful land-cover labels.
4. Create a reproducible subset.
5. Train/adapt using that subset.
6. Record the exact number of images used.
7. Save the adapter/checkpoint.
8. Compare before/after adaptation on a small remote-sensing validation set.

Example:

```text
BigEarthNet-S2
      |
      v
subset selection
      |
      v
1,000–10,000 patches
      |
      v
preprocessing
      |
      v
LoRA adaptation
```

The exact subset size should be chosen according to available GPU/storage.

Do NOT hard-code a huge dataset requirement.

---

# 13. VQA Pipeline

Implement:

```python
answer_vqa(image_path, question)
```

Input:

```text
image_path
question
```

Output:

```json
{
  "task": "vqa",
  "answer": "...",
  "model": "RS-LLaVA",
  "confidence": 0.0,
  "evidence": {
    "image": "..."
  }
}
```

The confidence may initially be a calibrated/heuristic confidence if the model does not provide a reliable probability.

Do NOT fabricate probability values.

If true calibrated confidence is unavailable, expose:

```text
confidence_method = "heuristic"
```

or omit confidence until implemented properly.

---

# 14. Captioning Pipeline

Implement:

```python
generate_caption(image_path)
```

Input:

```text
image_path
```

Output:

```json
{
  "task": "captioning",
  "caption": "...",
  "model": "RS-LLaVA"
}
```

The caption should describe:

- land cover
- major visible objects
- spatial context when identifiable
- remote-sensing scene characteristics

Avoid hallucinating exact locations or objects that are not visually supported.

---

# 15. Execution Trace

Every request must generate an observable execution trace.

Example:

```text
SATQUERY EXECUTION TRACE

1. Input validation
   ✓ Single image detected
   ✓ Format: GeoTIFF
   ✓ Image readable

2. Query classification
   ✓ Task: VQA

3. Model selection
   ✓ Model: RS-LLaVA
   ✓ Remote-sensing adapted: YES

4. Preprocessing
   ✓ Image converted to model-compatible representation

5. Inference
   ✓ VQA completed

6. Output
   ✓ Answer generated
```

Do NOT expose chain-of-thought or internal reasoning.

Only show the observable tool/model execution information.

---

# 16. Evaluation

## VQA

Use RSVQA test splits.

Evaluate:

- answer accuracy
- exact match where applicable
- task/category performance where provided by the benchmark
- qualitative examples

Do not invent a new metric when the benchmark already specifies one.

## Captioning

Use VRSBench caption evaluation resources.

Possible metrics include:

- BLEU
- METEOR
- ROUGE-L
- CIDEr
- other metrics provided by the benchmark

Use the benchmark's own evaluation scripts when possible.

VRSBench also provides evaluation JSON files for captioning, referring expressions, and VQA.

---

# 17. Development Evaluation

Before attempting full benchmark evaluation, create a tiny smoke-test set.

Example:

```text
tests/
  single_image/
    image_001.jpg
    image_002.jpg
    questions.json
```

Run:

```text
image + question -> VQA answer
image -> caption
```

Every code change must preserve this smoke test.

---

# 18. Suggested Project Structure

```text
satquery/
│
├── app/
│   └── single_image_ui.py
│
├── agent/
│   └── single_image_router.py
│
├── models/
│   ├── rs_vlm.py
│   ├── vqa.py
│   └── captioning.py
│
├── preprocessing/
│   ├── image_loader.py
│   ├── geotiff.py
│   └── multispectral.py
│
├── adaptation/
│   ├── prepare_bigearthnet.py
│   ├── prepare_instructions.py
│   └── train_lora.py
│
├── evaluation/
│   ├── evaluate_rsvqa.py
│   └── evaluate_vrsbench.py
│
├── tests/
│   └── single_image/
│
├── data/
│   ├── bigearthnet/
│   ├── rsvqa/
│   └── vrsbench/
│
├── checkpoints/
│   └── rs_vlm_lora/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

---

# 19. Required Dependencies

Use the minimum required stack.

Likely dependencies:

```text
torch
torchvision
transformers
datasets
peft
accelerate
Pillow
rasterio
numpy
opencv-python
safetensors
```

Only add additional packages when required by the selected model.

Do NOT create a giant requirements file containing unrelated packages.

---

# 20. Important: Do Not Download Everything

The biggest practical mistake would be downloading every dataset in full.

Recommended order:

### First

RSVQA-LR or a small RSVQA subset.

Purpose:

```text
prove VQA works
```

### Second

VRSBench evaluation/sample data.

Purpose:

```text
prove captioning works
```

### Third

RS-LLaVA model/checkpoint.

Purpose:

```text
run remote-sensing VLM inference
```

### Fourth

Small BigEarthNet-S2 subset.

Purpose:

```text
remote-sensing adaptation
```

Only download full datasets if the hardware/storage and deadline allow it.

---

# 21. Two-Day Priority

## Day 1

### Priority 1
Set up RS-LLaVA or another selected RS-VLM.

### Priority 2
Run:

```text
image + question -> answer
```

### Priority 3
Run:

```text
image -> caption
```

### Priority 4
Implement image validation.

### Priority 5
Implement the router.

At the end of Day 1:

```text
single image
     |
     +--> VQA
     |
     +--> caption
```

must work.

---

## Day 2

### Priority 1
Implement LoRA/PEFT adaptation.

### Priority 2
Use a manageable BigEarthNet/RS-instruction subset.

### Priority 3
Evaluate VQA.

### Priority 4
Evaluate captioning.

### Priority 5
Add execution trace.

### Priority 6
Expose a clean Python/API interface for the main SatQuery agent.

---

# 22. Interface to the Main SatQuery Agent

The single-image module must expose one clean entry point:

```python
result = analyze_single_image(
    image_path=image_path,
    query=query
)
```

Return:

```json
{
  "task": "vqa",
  "model": "RS-LLaVA",
  "adapted": true,
  "answer": "...",
  "confidence": null,
  "evidence": {},
  "execution_trace": []
}
```

For captioning:

```json
{
  "task": "captioning",
  "model": "RS-LLaVA",
  "adapted": true,
  "caption": "...",
  "evidence": {},
  "execution_trace": []
}
```

The main SatQuery agent can then call this module.

---

# 23. What NOT to Implement in This Module

Do NOT spend time on:

- bi-temporal change detection
- CDVQA
- optical-SAR fusion
- SAR-specific models
- Sentinel live-data retrieval
- NDVI explorer
- earthquake risk
- GIS map server
- user authentication
- database
- mobile app
- training a VQA CNN from scratch
- training a VLM from scratch
- downloading the complete BigEarthNet unless required

Those belong to other parts of the overall SatQuery system.

---

# 24. Definition of Done

The single-image module is considered complete when ALL of the following work:

### Input

- [ ] PNG/JPEG benchmark image works.
- [ ] TIFF/GeoTIFF input works.
- [ ] Invalid files are rejected.
- [ ] Single-image input is detected.

### VQA

- [ ] Natural-language question accepted.
- [ ] Remote-sensing VLM generates answer.
- [ ] RSVQA evaluation can run.

### Captioning

- [ ] Natural-language scene description generated.
- [ ] VRSBench caption evaluation can run.

### Adaptation

- [ ] Remote-sensing adaptation method exists.
- [ ] LoRA/PEFT adapter or equivalent checkpoint is saved.
- [ ] Training data source is documented.
- [ ] Base vs adapted model can be compared.

### Agent interface

- [ ] Query router distinguishes VQA and captioning.
- [ ] Correct specialist is selected.
- [ ] Model name is returned.
- [ ] Execution trace is returned.

### Integration

- [ ] `analyze_single_image()` can be called by the main SatQuery agent.
- [ ] Output is structured JSON/dict.
- [ ] No hidden dependency on the UI.

---

# 25. Recommended Implementation Decision

Use this as the default stack unless a dependency is genuinely incompatible:

```text
DATASETS
---------
BigEarthNet-S2       -> remote-sensing adaptation
RSVQA-LR/HR          -> VQA evaluation
VRSBench             -> captioning evaluation


MODEL
-----
RS-LLaVA             -> primary remote-sensing VLM


ADAPTATION
----------
LoRA / PEFT
+
remote-sensing instruction data
+
manageable BigEarthNet subset where practical


FRAMEWORK
---------
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT


IMAGE HANDLING
--------------
Rasterio -> GeoTIFF
Pillow   -> PNG/JPEG
NumPy    -> preprocessing


INTERFACE
---------
analyze_single_image(image_path, query)


TASKS
-----
VQA
Captioning
```

---

# 26. Agent Instructions

When implementing this document:

1. Inspect the selected model repository before writing integration code.
2. Use the official dataset sources above.
3. Do not invent download URLs.
4. Do not download entire multi-GB datasets automatically if a subset/streaming option exists.
5. Store datasets outside the Git repository or use `.gitignore`.
6. Never commit dataset files or model weights to Git.
7. Keep API keys/secrets out of source code.
8. Create a reproducible download/preparation script.
9. Create a small smoke test before full training.
10. Prefer a working pretrained remote-sensing VLM plus LoRA adaptation over training from scratch.
11. Do not expose chain-of-thought.
12. Record only observable execution information in the execution trace.
13. Do not claim confidence if the model does not provide a valid confidence estimate.
14. Keep the module independent so the main SatQuery agent can call it.
15. If a model cannot run on the available hardware, fall back to the next model in the priority list instead of spending excessive time debugging.

---

# 27. Quick Reference Links

BigEarthNet:
https://bigearth.net/

RSVQA original:
https://github.com/syvlo/RSVQA

RSVQA implementation supplied by user:
https://github.com/kaaydin/vqa-remote-sensing

RSVQA-HR:
https://doi.org/10.5281/zenodo.6344366

RSVQA-LR:
https://doi.org/10.5281/zenodo.6344333

VRSBench project:
https://vrsbench.github.io/

VRSBench GitHub:
https://github.com/lx709/VRSBench

VRSBench Hugging Face:
https://huggingface.co/datasets/xiang709/VRSBench

RS-LLaVA:
https://github.com/BigData-KSU/RS-LLaVA

TorchRS:
https://github.com/isaaccorley/torchrs

---

# Final Target

The final single-image subsystem should be:

```text
                 Satellite Image
                       |
                       v
               Input Validation
                       |
                       v
                  User Query
                       |
                       v
                 Query Router
                  /         \
                 /           \
                v             v
              VQA         Captioning
                \             /
                 \           /
                  v         v
                Remote-Sensing
                     VLM
                      |
               LoRA-adapted
                      |
                      v
               Evidence/Answer
                      |
                      v
                Execution Trace
                      |
                      v
              Main SatQuery Agent
```

Build this first. Once it is stable, the other team members can connect their change-detection and optical-SAR specialists to the same agentic framework.

