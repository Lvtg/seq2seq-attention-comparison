# Length-Aware Gated Attention

## Setup

- Dataset: Multi30k English-German
- Model: LSTM encoder-decoder with Luong-style attention plus a scalar fusion gate
- Config: `configs/gated_attention.json`
- Device: CUDA, NVIDIA GeForce RTX 5070 Laptop GPU
- Parameters: 9,567,434
- Decoding: greedy
- Gate export: `artifacts/gates/gated_attention_test_gate_examples.json`

## Method

This variant keeps the regular attention model and adds a lightweight gate before the output projection. At each decoding step, the model combines three signals:

- the decoder output,
- the attention context,
- the normalized source length.

The source length is normalized by a fixed `length_feature_norm = 80.0`, rather than by the maximum length inside the current batch. This keeps the gate feature comparable across batches and makes the gate easier to interpret.

The gate can be used as a small original extension for the report: it tests whether the model benefits from dynamically adjusting how much attention context it uses, especially for longer source sentences.

## Training Result

Training stopped early at epoch 19. The best validation loss was reached at epoch 14.

| Metric | Value |
| --- | ---: |
| Best epoch | 14 |
| Best validation loss | 1.9568738340445324 |
| Final epoch | 19 |
| Final training loss | 0.7772488395629709 |
| Final validation loss | 2.0352844478894125 |
| Total training time | 954.53 seconds |

The gated model reaches almost the same best validation loss as the plain attention model, but it trains more slowly and overfits after the best epoch.

## Test Result

| Metric | LSTM no attention | LSTM attention | Gated attention |
| --- | ---: | ---: | ---: |
| BLEU | 20.46884762591954 | 27.309644894819208 | 25.965621591958236 |
| chrF | 43.88261832459875 | 49.54956633988433 | 49.601161908773385 |
| Best validation loss | 2.1719555077088617 | 1.9562223262533678 | 1.9568738340445324 |
| Parameters | 8,534,474 | 9,124,298 | 9,567,434 |

Compared with the no-attention baseline, the gated model still gives a large gain:

- BLEU: +5.496773966038695
- chrF: +5.718543584174633
- Best validation loss: -0.2150816736643293

Compared with the plain attention model, the result is mixed:

- BLEU: -1.3440233028609718
- chrF: +0.0515955688890557
- Best validation loss: +0.0006515077911646294
- Parameters: +443,136

This suggests that the proposed gate is useful as an analysis extension, but it is not a clear improvement over standard attention under the current training budget and greedy decoding setup.

## Gate Analysis

The exported gate examples focus on test sentences with at least 10 source tokens.

| Statistic | Value |
| --- | ---: |
| Examples | 12 |
| Gate values | 181 |
| Mean gate value | 0.6999745915607852 |
| Min gate value | 0.5994956493377686 |
| Max gate value | 0.7760640978813171 |
| Min per-example gate mean | 0.6831034421920776 |
| Max per-example gate mean | 0.7208638404096875 |
| Source length range | 10-29 |

The gate is consistently above 0.5 in the exported examples, meaning the fused decoder representation usually leans toward the attention context. For the final report, this can support a careful claim: attention context remains important for longer descriptions, but the simple length-aware gate does not automatically translate into better BLEU.

## Artifacts

- Machine-readable result summary: `artifacts/tables/gated_attention_results.json`
- Greedy test translations: `artifacts/translations/gated_attention_test_greedy.txt`
- Gate and attention examples: `artifacts/gates/gated_attention_test_gate_examples.json`
