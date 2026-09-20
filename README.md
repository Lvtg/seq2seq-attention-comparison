# Multi30k Seq2Seq Experiments

This repository contains the code and saved results for a small study of English–German neural machine translation on Multi30k. The main comparison is between an LSTM encoder–decoder without attention, an attention-based LSTM, a gated-attention variant, and a compact Transformer.

The experiments focus on three practical questions:

- Does attention reduce the fixed-vector bottleneck of an encoder–decoder model?
- How do the models behave as source sentences become longer?
- How much do beam search and length penalties change the comparison?

The project description and the final experimental conclusions are in `COURSE_PROJECT_GOAL.md` and `analysis/report_notes/final_findings.md`. A compact English report is available in `output/pdf/project_report.pdf`, with its editable source in `output/pdf/project_report.md`.

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default configurations are deliberately small enough to run on a single consumer GPU. If the CUDA 12.8 PyTorch wheel does not match the local driver, install a compatible PyTorch build instead.

## Data Preparation

Download and preprocess Multi30k English-German:

```powershell
python scripts/prepare_multi30k.py --config configs/base.json
```

The script downloads raw files from the official Multi30k dataset repository, tokenizes them with the project tokenizer, builds vocabularies from the training split, and writes processed JSONL files.

## Quick checks

Inspect a config without requiring PyTorch:

```powershell
python scripts/inspect_config.py --config configs/lstm_attention.json
```

After dependencies and data are ready, run a short training check:

```powershell
python scripts/train.py --config configs/lstm_attention.json --max-epochs 1 --limit-train-batches 5 --limit-valid-batches 2
```

## Experiments

Train the planned models:

```powershell
python scripts/train.py --config configs/lstm_no_attention.json
python scripts/train.py --config configs/lstm_attention.json
python scripts/train.py --config configs/gated_attention.json
python scripts/train.py --config configs/transformer_small.json
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/lstm_attention.json --checkpoint checkpoints/lstm_attention/best.pt --split test
```

Export attention examples from an attention model:

```powershell
python scripts/export_attention.py --config configs/lstm_attention.json --checkpoint checkpoints/lstm_attention/best.pt --split test --num-examples 12 --min-source-length 10
```

Processed data, checkpoints, and raw training logs are local-only and are ignored by git. The checked-in `artifacts/` and `analysis/` directories contain the compact result files used for the report.

## Repository layout

- `src/seq2seq_project/`: data loading, vocabulary, model definitions, training, and decoding.
- `scripts/`: commands for preprocessing, training, evaluation, decoding, and analysis.
- `configs/`: the four experiment configurations.
- `artifacts/`: selected translations and machine-readable result summaries.
- `analysis/`: metric tables, figures, length buckets, decoding comparisons, and error cases.
- `output/pdf/`: the report source, figures, and generated PDF.
