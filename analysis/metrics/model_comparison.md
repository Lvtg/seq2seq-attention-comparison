# Model Comparison

| Model | Parameters | Best epoch | Best valid loss | Training seconds | Greedy BLEU | Greedy chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM no attention | 8534474 | 9 | 2.171956 | 355.66 | 20.4688 | 43.8826 |
| LSTM attention | 9124298 | 13 | 1.956222 | 645.80 | 27.3096 | 49.5496 |
| Gated attention | 9567434 | 14 | 1.956874 | 954.53 | 25.9656 | 49.6012 |
| Small Transformer | 9508042 | 11 | 1.826523 | 170.16 | 25.9396 | 52.0157 |

Notes:

- All scores use the shared tokenized Multi30k preprocessing.
- Checkpoints and raw run logs remain local-only and are not tracked by git.
- This table is generated from the milestone result JSON files under `artifacts/tables/`.
