# Temporal Datasets

This directory contains dataset adapters and documentation for
bi-temporal remote-sensing analysis.

## Development Dataset

### ChangeChat-105k + LEVIR-CC

Used for development and adaptation of the DeltaVLM-based
bi-temporal analysis pipeline.

ChangeChat-105k provides temporal instruction data for tasks
including:

- Change captioning
- Binary change classification
- Change quantification
- Change localization
- Open-ended question answering
- Multi-turn temporal dialogue

LEVIR-CC provides the corresponding bi-temporal image pairs.

## Evaluation Dataset

### CDVQA

Used for evaluating multitemporal change-based visual question
answering as specified by the SatQuery AI problem statement.

The prescribed evaluation split must remain separate from
training/development data to avoid data leakage.

## Optional Spatial Evidence Dataset

### LEVIR-MCI

May be used later for explicit change-mask and spatial
localization experiments.

It is not required for the initial implementation.

