# LSTM Attention Baseline

## Setup

- Dataset: Multi30k English-German
- Model: LSTM encoder-decoder with Luong-style attention
- Config: `configs/lstm_attention.json`
- Device: CUDA, NVIDIA GeForce RTX 5070 Laptop GPU
- Parameters: 9,124,298
- Decoding: greedy
- Attention export: `artifacts/attention/lstm_attention_test_attention_examples.json`

## Training Result

Training stopped early at epoch 18. The best validation loss was reached at epoch 13.

| Metric | Value |
| --- | ---: |
| Best epoch | 13 |
| Best validation loss | 1.9562223262533678 |
| Final epoch | 18 |
| Final training loss | 0.8531262354486269 |
| Final validation loss | 2.016518550408625 |
| Total training time | 645.80 seconds |

The attention model improves validation loss over the no-attention baseline while using only a modest number of additional parameters. It still starts overfitting after the best epoch, but its best validation point is substantially better.

## Test Result

| Metric | LSTM no attention | LSTM attention | Delta |
| --- | ---: | ---: | ---: |
| BLEU | 20.46884762591954 | 27.309644894819208 | +6.840797268899668 |
| chrF | 43.88261832459875 | 49.54956633988433 | +5.66694801528558 |
| Best validation loss | 2.1719555077088617 | 1.9562223262533678 | -0.2157331814554939 |

The score is computed on the project-tokenized test split. The same tokenized evaluation is used for all models, so these numbers are intended for within-project comparison.

## Qualitative Notes

Attention clearly improves source-detail retention. Several first-page examples become more faithful than the no-attention baseline:

- "a man in an orange hat..." is translated as "ein mann mit orangefarbener kappe starrt etwas an", preserving the main action.
- "a guy works on a building" is translated exactly as "ein typ arbeitet an einem gebäude".
- The volleyball example preserves the sport, unlike the no-attention baseline that confused it with football.

Remaining errors are still useful for the report:

- Rare object words such as "igloo" can still be dropped or replaced by incomplete generic phrases.
- Long, compositional descriptions improve but still lose fine details.
- Some generated sentences are locally grammatical while semantically incomplete.

This supports the course-level claim that attention reduces the fixed-vector bottleneck but does not eliminate all data, vocabulary, and decoding limitations.

## Attention Export

The script `scripts/export_attention.py` exports source tokens, generated hypothesis tokens, and the attention matrix for selected examples. The source token list includes `<eos>` so matrix columns align with tokens.
