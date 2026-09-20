# LSTM No-Attention Baseline

## Setup

- Dataset: Multi30k English-German
- Model: LSTM encoder-decoder without attention
- Config: `configs/lstm_no_attention.json`
- Device: CUDA, NVIDIA GeForce RTX 5070 Laptop GPU
- Parameters: 8,534,474
- Decoding: greedy

## Training Result

Training stopped early at epoch 14. The best validation loss was reached at epoch 9.

| Metric | Value |
| --- | ---: |
| Best epoch | 9 |
| Best validation loss | 2.1719555077088617 |
| Final epoch | 14 |
| Final training loss | 0.9891154744914079 |
| Final validation loss | 2.226636076530524 |
| Total training time | 355.66 seconds |

The training loss continued to fall after epoch 9, while validation loss stopped improving. This is a useful baseline pattern: the model can memorize the training distribution, but without attention it has limited generalization capacity.

## Test Result

| Metric | Value |
| --- | ---: |
| BLEU | 20.46884762591954 |
| chrF | 43.88261832459875 |

The score is computed on the project-tokenized test split. The same tokenized evaluation will be used for later models, so the numbers are intended for within-project comparison.

## Qualitative Notes

The model learns common sentence frames such as "ein mann ..." and simple object/action patterns. It performs acceptably on some short sentences, for example "a guy works on a building" becomes "ein mann arbeitet an einem gebäude".

Failure cases already show the expected encoder-decoder bottleneck:

- Long sentences often lose details or replace rare phrases with generic visual-description phrases.
- Rare objects are mistranslated, for example "igloo" becomes a generic "haus".
- Some outputs are grammatical but semantically wrong, for example roof repair becomes a scene near water.
- Long source sentences can produce repetitive or structurally confused translations.

This gives the next attention-based model a clear target: improve source-detail retention and long-sentence faithfulness.
