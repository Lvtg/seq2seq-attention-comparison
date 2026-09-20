# Figure And Table Index

## Tables

| File | Suggested label | Suggested use |
| --- | --- | --- |
| `analysis/metrics/model_comparison.md` | Table 1 | 四个模型的参数量、best valid loss、训练时间、greedy BLEU 和 chrF。 |
| `analysis/decoding/decoding_valid_grid.csv` | Table 2 | Validation split 上 greedy/beam/length penalty 解码网格。 |
| `analysis/decoding/decoding_test_selected.csv` | Table 3 | 基于 validation 选择后，在 test split 上确认的最终解码结果。 |
| `analysis/length_buckets/length_bucket_results.csv` | Table 4 | 按 source length 分桶的 BLEU、chrF 和长度比例。 |
| `analysis/length_buckets/generation_length_stats.csv` | Table 5 | 不同模型输出长度、短于/长于 reference 的比例。 |
| `analysis/attention_gate/gate_stats.csv` | Table 6 | Gated attention 每个导出样例的 gate 统计。 |
| `analysis/error_analysis/case_samples.md` | Table 7 | 定性错误案例，可挑选部分放入正文，其余放附录。 |

## Figures

| File | Suggested label | Suggested caption |
| --- | --- | --- |
| `analysis/figures/error_tag_counts.png` | Figure 1 | 定性错误分析样例的错误标签计数。 |
| `analysis/figures/gate_mean_distribution.png` | Figure 2 | Gated attention 示例的平均 gate 分布。 |
| `analysis/figures/gate_mean_vs_source_length.png` | Figure 3 | Gated attention 平均 gate 与 source length 的关系。 |
| `analysis/figures/generation_length_ratio.png` | Figure 4 | 不同模型和解码策略的 hypothesis/reference 长度比例。 |
| `analysis/figures/length_bucket_bleu_greedy.png` | Figure 5 | Greedy decoding 下按 source length 分桶的 BLEU。 |
| `analysis/figures/length_bucket_bleu_selected.png` | Figure 6 | Selected decoding 下按 source length 分桶的 BLEU。 |
| `analysis/figures/length_bucket_chrf_greedy.png` | Figure 7 | Greedy decoding 下按 source length 分桶的 chrF。 |
| `analysis/figures/length_bucket_chrf_selected.png` | Figure 8 | Selected decoding 下按 source length 分桶的 chrF。 |
| `analysis/figures/model_metric_bars.png` | Figure 9 | 四个模型的 greedy BLEU、greedy chrF 与 best valid loss 对比。 |
| `analysis/figures/training_time_comparison.png` | Figure 10 | 四个模型的训练耗时对比。 |
| `analysis/figures/valid_loss_curves.png` | Figure 11 | 四个模型 validation loss 随 epoch 变化曲线。 |
| `analysis/attention_gate/gated_attention_gate_curve_01.png` | Figure 12 | Gated attention 的 gate-over-time 曲线，用于观察每个生成步骤的 gate 变化。 |
| `analysis/attention_gate/gated_attention_gate_curve_02.png` | Figure 13 | Gated attention 的 gate-over-time 曲线，用于观察每个生成步骤的 gate 变化。 |
| `analysis/attention_gate/gated_attention_gate_curve_03.png` | Figure 14 | Gated attention 的 gate-over-time 曲线，用于观察每个生成步骤的 gate 变化。 |
| `analysis/attention_gate/gated_attention_gate_curve_04.png` | Figure 15 | Gated attention 的 gate-over-time 曲线，用于观察每个生成步骤的 gate 变化。 |
| `analysis/attention_gate/gated_attention_heatmap_01.png` | Figure 16 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/gated_attention_heatmap_02.png` | Figure 17 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/gated_attention_heatmap_03.png` | Figure 18 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/gated_attention_heatmap_04.png` | Figure 19 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/lstm_attention_heatmap_01.png` | Figure 20 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/lstm_attention_heatmap_02.png` | Figure 21 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/lstm_attention_heatmap_03.png` | Figure 22 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |
| `analysis/attention_gate/lstm_attention_heatmap_04.png` | Figure 23 | Attention 权重热力图，用于展示目标 token 对源 token 的对齐。 |

## Recommended Main-Text Figures

- `analysis/figures/model_metric_bars.png`
- `analysis/figures/valid_loss_curves.png`
- `analysis/figures/length_bucket_bleu_selected.png`
- `analysis/figures/generation_length_ratio.png`
- `analysis/figures/gate_mean_vs_source_length.png`
- `analysis/figures/error_tag_counts.png`

Attention heatmap 和 gate curve 建议各选 1-2 张放正文，其余作为附录材料。
