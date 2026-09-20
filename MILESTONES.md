# Milestone Log

This file records important saved states for the Multi30k Seq2Seq course project.

## 2026-08-20

### project-goal-and-scaffold

- Commit: `040c1e6`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Initialized the current course directory as the project git repository.
  - Bound `origin` to `https://github.com/Lvtg/seq2seq_project.git`.
  - Added project goal record, README, configs, source package, model skeletons, training/evaluation scripts, and dependency files.
  - Kept the repository focused on the translation experiments and their supporting code.

### data-pipeline

- Commit: see git log entry `data-pipeline`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Ran `python scripts/prepare_multi30k.py --config configs/base.json`.
  - Downloaded and preprocessed Multi30k English-German.
  - Raw and processed data are intentionally ignored by git.
  - Dataset summary is saved in `artifacts/tables/multi30k_data_summary.json`.

### lstm-baseline

- Commit: see git log entry `lstm-baseline`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Trained `configs/lstm_no_attention.json` on CUDA.
  - Early stopped at epoch 14; best validation loss at epoch 9.
  - Test greedy BLEU: 20.46884762591954.
  - Test greedy chrF: 43.88261832459875.
  - Checkpoints and raw run logs are intentionally ignored by git.
  - Lightweight summary files are saved under `artifacts/tables/`.

### lstm-attention

- Commit: see git log entry `lstm-attention`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Trained `configs/lstm_attention.json` on CUDA.
  - Early stopped at epoch 18; best validation loss at epoch 13.
  - Test greedy BLEU: 27.309644894819208.
  - Test greedy chrF: 49.54956633988433.
  - BLEU improved by 6.840797268899668 over the no-attention LSTM baseline.
  - Added `scripts/export_attention.py` and exported attention examples for later heatmap analysis.
  - Checkpoints and raw run logs are intentionally ignored by git.

### gated-attention

- Commit: see git log entry `gated-attention`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Trained `configs/gated_attention.json` on CUDA.
  - Added a length-aware scalar gate over decoder output and attention context.
  - Early stopped at epoch 19; best validation loss at epoch 14.
  - Test greedy BLEU: 25.965621591958236.
  - Test greedy chrF: 49.601161908773385.
  - The best validation loss nearly matches plain attention, but BLEU is lower under greedy decoding.
  - Exported gate and attention examples for later heatmap and gate-curve analysis.
  - Checkpoints and raw run logs are intentionally ignored by git.

### small-transformer

- Commit: see git log entry `small-transformer`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Trained `configs/transformer_small.json` on CUDA.
  - Used a compact 3-layer encoder-decoder Transformer with 9,508,042 trainable parameters.
  - Early stopped at epoch 16; best validation loss at epoch 11.
  - Best validation loss: 1.8265231856202657.
  - Test greedy BLEU: 25.939559187170754.
  - Test greedy chrF: 52.015681134763156.
  - The model has the strongest validation loss and chrF so far, but does not beat LSTM attention on greedy BLEU.
  - Checkpoints and raw run logs are intentionally ignored by git.

### analysis-scaffold-and-model-summary

- Commit: see git log entry `analysis-scaffold-and-model-summary`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added `analysis/` as the main directory for post-experiment analysis materials.
  - Added a generated model comparison table for all four trained models.
  - Added report-ready figures for greedy metrics, validation loss curves, and training time.
  - Kept original milestone result files under `artifacts/` unchanged.

### decoding-evaluation

- Commit: see git log entry `decoding-evaluation`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added beam search decoding for LSTM and Transformer models.
  - Added validation-grid decoding evaluation for LSTM attention, gated attention, and small Transformer.
  - Selected each model's final test decoding strategy from validation BLEU, with chrF as a near-tie breaker.
  - Saved decoding tables and selected translations under `analysis/decoding/`.

### length-and-generation-analysis

- Commit: see git log entry `length-and-generation-analysis`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added source-length bucket metrics for greedy and selected decoding outputs.
  - Added generation-length statistics, including hypothesis/reference length ratios.
  - Generated report-ready BLEU, chrF, and length-ratio figures under `analysis/figures/`.
  - Saved length analysis tables and summary notes under `analysis/length_buckets/`.

### attention-gate-analysis

- Commit: see git log entry `attention-gate-analysis`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added attention heatmaps for plain attention and gated attention examples.
  - Added gated-attention gate curves, gate-step values, and aggregate gate statistics.
  - Added gate distribution and gate-vs-source-length figures.
  - Saved interpretation notes under `analysis/attention_gate/`.

### qualitative-error-analysis

- Commit: see git log entry `qualitative-error-analysis`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added representative test cases covering model disagreement, long sentences, rare words, and strong outputs.
  - Added fixed error taxonomy for qualitative report discussion.
  - Saved case samples in CSV, JSON, and Markdown formats under `analysis/error_analysis/`.
  - Added an error-tag count figure under `analysis/figures/`.

### final-analysis-artifacts

- Commit: see git log entry `final-analysis-artifacts`
- Remote: `origin/main`
- Status: pushed
- Notes:
  - Added final report-oriented findings under `analysis/report_notes/`.
  - Added a figure and table index for selecting report-ready materials.
  - Consolidated quantitative, decoding, length, attention/gate, and qualitative observations into reusable notes.
