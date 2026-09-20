# Decoding Evaluation Summary

## Setup

- Models: LSTM attention, gated attention, small Transformer
- Validation grid: greedy, beam size 3/5, length penalty 0.0/0.6/1.0
- Selection policy: choose the highest validation BLEU; if BLEU differs by at most 0.1, choose higher chrF
- Test split: evaluated only with the selected validation configuration for each model

The length penalty uses the GNMT-style normalization `score / (((5 + length) / 6) ** alpha)`.

## Validation Selection

| Model | Selected decoding | Valid BLEU | Valid chrF |
| --- | --- | ---: | ---: |
| LSTM attention | beam5_lp1p0 | 28.834461689876054 | 50.832128511437546 |
| Gated attention | beam5_lp1p0 | 29.59477499236177 | 51.2655493973148 |
| Small Transformer | beam3_lp1p0 | 27.305077861266195 | 52.22322145265269 |

All three models improve over greedy decoding on validation BLEU. The larger length penalty is consistently helpful in this grid, suggesting that greedy decoding and unpenalized beam search may prefer outputs that are too short for this tokenized evaluation setup.

## Selected Test Results

| Model | Selected decoding | Test BLEU | Test chrF |
| --- | --- | ---: | ---: |
| LSTM attention | beam5_lp1p0 | 28.596180949452656 | 50.9425661418625 |
| Gated attention | beam5_lp1p0 | 28.04403796592976 | 51.368765020259545 |
| Small Transformer | beam3_lp1p0 | 26.845074885114556 | 52.9121277803892 |

The ranking remains nuanced. LSTM attention has the strongest selected-test BLEU, while the small Transformer has the strongest selected-test chrF. Gated attention improves substantially over its greedy BLEU, but still does not clearly beat the plain attention baseline on test BLEU.

## Report Takeaway

Decoding strategy changes the absolute scores and narrows the gap between models, but it does not completely overturn the model-level story. Beam search helps all models, length penalty matters, and BLEU/chrF emphasize different aspects of translation quality.
