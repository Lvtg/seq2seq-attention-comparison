# Length and Generation Analysis

## Corpus-Level Length Statistics

| Result set | Model | Decoding | BLEU | chrF | Hyp/ref length ratio | Shorter than ref |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| greedy | LSTM no attention | greedy | 20.4688 | 43.8826 | 0.9714 | 0.4390 |
| greedy | LSTM attention | greedy | 27.3096 | 49.5496 | 0.9557 | 0.4870 |
| greedy | Gated attention | greedy | 25.9656 | 49.6012 | 0.9695 | 0.4560 |
| greedy | Small Transformer | greedy | 25.9396 | 52.0157 | 0.9742 | 0.4540 |
| selected | LSTM attention | beam5_lp1p0 | 28.5962 | 50.9426 | 0.9423 | 0.4790 |
| selected | Gated attention | beam5_lp1p0 | 28.0440 | 51.3688 | 0.9480 | 0.4920 |
| selected | Small Transformer | beam3_lp1p0 | 26.8451 | 52.9121 | 0.9339 | 0.5120 |

## Main Observations

- Among selected decoding results, LSTM attention has the highest BLEU (28.5962).
- Among selected decoding results, Small Transformer has the highest chrF (52.9121).
- The small Transformer keeps the strongest chrF while not leading BLEU, supporting the claim that BLEU and chrF emphasize different generation behavior.
- Transformer greedy hyp/ref length ratio is 0.9742, which should be discussed together with beam-search length penalty results.
- The `31+` bucket contains only two test examples, so use it as qualitative evidence only.
- Selected beam outputs remain slightly shorter than references overall; length penalty improves BLEU without fully matching reference length.

## Files

- Bucket metrics: `length_bucket_results.csv` and `length_bucket_results.json`
- Corpus generation metrics: `generation_length_stats.csv` and `generation_length_stats.json`
- Figures are saved under `analysis/figures/`.
