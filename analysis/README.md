# Analysis Directory

This directory stores post-experiment analysis materials for the Multi30k Seq2Seq course project.

The original milestone artifacts remain under `artifacts/`. This directory collects derived tables, figures, notes, and selected examples for the final report.

## Structure

- `metrics/`: model-level comparison tables and training curves.
- `decoding/`: greedy, beam search, and length-penalty evaluation results.
- `length_buckets/`: source-length bucket metrics and generation-length analysis.
- `attention_gate/`: attention heatmaps, gate curves, and gate statistics.
- `error_analysis/`: selected qualitative examples and error taxonomy.
- `figures/`: report-ready figures generated from analysis tables.
- `report_notes/`: final observations, table index, and draft report claims.

## Data Policy

- Track lightweight tables, figures, selected translations, and notes.
- Keep checkpoints, full raw data, caches, and large temporary outputs out of git.
- Use validation split results for decoding selection; use test split only for final confirmation.
