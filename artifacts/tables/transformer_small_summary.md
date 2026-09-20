# Small Transformer Baseline

## Setup

- Dataset: Multi30k English-German
- Model: compact Transformer encoder-decoder
- Config: `configs/transformer_small.json`
- Device: CUDA, NVIDIA GeForce RTX 5070 Laptop GPU
- Parameters: 9,508,042
- Decoding: greedy

## Architecture

The model uses a 3-layer Transformer encoder and 3-layer Transformer decoder with 256-dimensional embeddings, 4 attention heads, a 512-dimensional feed-forward block, sinusoidal positional encodings, and dropout 0.1. It has a similar parameter scale to the LSTM attention variants, so it is a useful architecture-level comparison rather than a much larger model.

## Training Result

Training stopped early at epoch 16. The best validation loss was reached at epoch 11.

| Metric | Value |
| --- | ---: |
| Best epoch | 11 |
| Best validation loss | 1.8265231856202657 |
| Final epoch | 16 |
| Final training loss | 1.0849272192503971 |
| Final validation loss | 1.859501098666571 |
| Total training time | 170.16 seconds |

The Transformer trains much faster than the LSTM attention variants in this project and reaches the best validation loss among the models trained so far.

## Test Result

| Metric | LSTM no attention | LSTM attention | Gated attention | Small Transformer |
| --- | ---: | ---: | ---: | ---: |
| BLEU | 20.46884762591954 | 27.309644894819208 | 25.965621591958236 | 25.939559187170754 |
| chrF | 43.88261832459875 | 49.54956633988433 | 49.601161908773385 | 52.015681134763156 |
| Best validation loss | 2.1719555077088617 | 1.9562223262533678 | 1.9568738340445324 | 1.8265231856202657 |
| Parameters | 8,534,474 | 9,124,298 | 9,567,434 | 9,508,042 |

Compared with the no-attention LSTM baseline:

- BLEU: +5.470711561251215
- chrF: +8.133062810164406
- Best validation loss: -0.345432322088596

Compared with the plain LSTM attention baseline:

- BLEU: -1.3700857076484532
- chrF: +2.466114794878827
- Best validation loss: -0.12969914063310206

The result is nuanced: the Transformer has the best validation loss and chrF, but it does not beat the LSTM attention model on greedy BLEU. This makes it a strong candidate for the next decoding experiment, because beam search and length penalty may change the BLEU ranking.

## Qualitative Notes

The Transformer often produces fluent short translations and preserves some content words that earlier models missed. For example, the first test examples include better translations for the terrier sentence, the roof-repair sentence, and the igloo sentence.

Remaining errors are still visible in longer compositional descriptions:

- some fine-grained modifiers are dropped,
- repeated roles can be collapsed,
- long action-heavy sentences may become syntactically odd under greedy decoding.

These observations support a careful report claim: the Transformer reduces training time and improves token-level modeling, but decoding quality still depends on the generation strategy.

## Artifacts

- Machine-readable result summary: `artifacts/tables/transformer_small_results.json`
- Greedy test translations: `artifacts/translations/transformer_small_test_greedy.txt`
