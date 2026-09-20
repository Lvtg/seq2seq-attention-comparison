# Course Project Goal Record

Last updated: 2026-09-20

## Project scope

Use the Multi30k English-German machine translation task as the experimental base for the course project.

The project will study the sequence-to-sequence bottleneck problem and the value of attention by comparing classic RNN encoder-decoder models, attention-based RNN encoder-decoder models, and a small Transformer baseline.

## Working Title

From Encoder-Decoder to Transformer: An Experimental Study of Attention, Decoding, and Long-Sequence Behavior on Multi30k English-German Translation

Chinese title:

从 Encoder-Decoder 到 Transformer: 基于 Multi30k 英德翻译的 Attention、解码策略与长序列表现实验研究

## Motivation

This course covers neural networks, RNNs, LSTMs, encoder-decoder models, attention mechanisms, Transformers, decoding strategies, and practical training/evaluation issues. Multi30k English-German translation is a suitable project setting because it directly requires sequence-to-sequence modeling and is small enough to run on the local RTX 5070-class machine within a 10-day experimental window.

The central question is:

How much does attention help with the encoder-decoder bottleneck, and how does a small Transformer compare with RNN-based seq2seq models under the same dataset and compute constraints?

## Main Research Questions

1. Does a vanilla LSTM encoder-decoder without attention degrade more strongly on longer source sentences?
2. Does attention improve translation quality mainly by improving long-sequence handling, or does it also improve short sentences?
3. Is a small Transformer better than LSTM with attention on Multi30k, and if so, is the advantage in quality, training speed, decoding behavior, or length robustness?
4. How do greedy decoding, beam search, and length penalty affect translation quality and output length?
5. Can a simple self-designed gated fusion variant improve or at least clarify the role of global encoder state versus attention context?

## Proposed Self-Designed Variant

The project also includes a lightweight gated attention variant:

Length-Aware Gated Attention

At each decoder step, the model computes a standard attention context vector and also keeps a global encoder representation. A learned gate decides how much to rely on each source of information:

```text
context_fused = gate * attention_context + (1 - gate) * global_context
```

The gate can be predicted from decoder state, attention context, global context, and optionally normalized source length:

```text
gate = sigmoid(W [decoder_state; attention_context; global_context; length_feature] + b)
```

This is a small experimental extension rather than a claim of a new state-of-the-art method. Its purpose is to test a clear hypothesis:

For short sentences, the global encoder state may already be sufficient; for longer sentences, the model should rely more strongly on attention. If the gate learns such a pattern, it provides interpretable evidence about the bottleneck problem.

## Models To Implement

Required models:

1. LSTM Encoder-Decoder without attention
2. LSTM Encoder-Decoder with Bahdanau or Luong attention
3. LSTM Encoder-Decoder with Length-Aware Gated Attention
4. Small Transformer encoder-decoder baseline

Optional extensions, only if time remains:

1. Local attention window
2. Extra Transformer size ablation
3. Additional random seed for more reliable comparison

## Dataset

Primary dataset:

Multi30k English-German translation.

Planned preprocessing:

1. Lowercase or case-preserving setting will be fixed before training.
2. Tokenization will use a reproducible tokenizer.
3. Vocabulary will be built from the training split only.
4. Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`.
5. Exact split sizes and vocabulary sizes will be recorded after preprocessing.

## Evaluation

Main metrics:

1. Validation loss
2. Test BLEU
3. chrF, if easy to integrate
4. Average generated length
5. Training time per epoch
6. Inference time
7. Peak GPU memory, if measurable

Length-based analysis:

Source sentences will be divided into length buckets, for example:

```text
short:  <= 10 tokens
medium: 11-20 tokens
long:   > 20 tokens
```

The report will compare BLEU or sentence-level quality proxies across these buckets.

Decoding experiments:

1. Greedy decoding
2. Beam search with several beam sizes
3. Beam search with length penalty

Visualization:

1. Attention heatmaps for selected examples
2. Gate values versus source length for the gated attention model
3. Training curves
4. Length-bucket performance plots

## Compute Constraint

Experiments are designed for a single local RTX 5070-class GPU. On this machine, `nvidia-smi` reports about 8 GB visible VRAM, so default batch sizes and model dimensions should stay conservative.

Practical assumptions:

1. Use compact models rather than large pretrained models.
2. Prefer reproducible experiments over very large hyperparameter sweeps.
3. Use early stopping or fixed epoch budgets.
4. Save checkpoints and logs for later report writing.
5. Keep batch sizes adjustable for the observed 8 GB VRAM budget.

## 10-Day Experimental Schedule

Day 1:

Set up project structure, environment, dataset loading, preprocessing, vocabulary, batching, and a minimal training loop.

Day 2:

Implement and train the vanilla LSTM encoder-decoder baseline.

Day 3:

Implement and train LSTM encoder-decoder with attention.

Day 4:

Implement and train the Length-Aware Gated Attention variant.

Day 5:

Implement and train the small Transformer baseline.

Day 6:

Run decoding experiments: greedy, beam search, beam size comparison, and length penalty.

Day 7:

Run length-bucket evaluation and collect long-sequence examples.

Day 8:

Run ablations: hidden size, dropout, or gate feature variants. Keep only ablations that are feasible.

Day 9:

Rerun important experiments or an additional seed for key models. Consolidate logs, checkpoints, and generated translations.

Day 10:

Generate final tables, plots, attention heatmaps, gate analysis figures, and a clean experiment summary for report writing.

## Expected Deliverables

Code deliverables:

1. Dataset preprocessing scripts
2. Model implementations
3. Training script
4. Evaluation script
5. Decoding script
6. Plotting and visualization scripts
7. Reproducible experiment configuration files

Experiment deliverables:

1. Model comparison table
2. Decoding comparison table
3. Length-bucket performance table
4. Training curves
5. Attention heatmaps
6. Gate behavior analysis for the proposed variant
7. Selected translation examples and error analysis

Report deliverables:

1. Motivation and course connection
2. Method descriptions
3. Experimental setup
4. Results
5. Analysis of attention and long-sequence behavior
6. Analysis of the self-designed gated variant
7. Limitations and future work

## Scope Control

In scope:

1. Training compact seq2seq models from scratch on Multi30k English-German.
2. Comparing attention, no-attention, gated attention, and small Transformer models.
3. Studying decoding strategies and sentence-length effects.

Out of scope unless all required experiments finish early:

1. Training or fine-tuning large language models.
2. Full comparison with Mamba, SSM, or linear attention implementations.
3. Large-scale hyperparameter search.
4. Multi-dataset benchmarking.

## Success Criteria

The project is successful if it produces:

1. At least three trained model families: vanilla LSTM seq2seq, LSTM with attention, and small Transformer.
2. The proposed gated attention variant trained and evaluated, even if it does not outperform the standard attention baseline.
3. BLEU or comparable translation metrics on the Multi30k test split.
4. A length-based analysis showing how model behavior changes with source sentence length.
5. At least one decoding-strategy comparison.
6. Visual or numerical analysis of attention behavior.
7. Enough organized logs, tables, and figures to write the final report.

## Collaboration Note

The implementation is deliberately compact and reproducible. The main research decisions, experiment settings, and conclusions are recorded in the configuration files, analysis tables, and report notes.
