# Attention and Gate Analysis

## Gate Summary

| Statistic | Value |
| --- | ---: |
| examples | 12 |
| gate_values | 181 |
| gate_mean | 0.699975 |
| gate_min | 0.599496 |
| gate_max | 0.776064 |
| per_example_gate_mean_min | 0.683103 |
| per_example_gate_mean_max | 0.720864 |
| pearson_gate_mean_source_length | -0.090373 |

## Attention Sharpness

| Model | Mean attention entropy |
| --- | ---: |
| LSTM attention | 0.412084 |
| Gated attention | 0.307059 |

## Main Observations

- The exported gated examples have gate means around 0.70, so the fused representation usually leans toward the attention context.
- The gate range is modest in the exported examples, suggesting the learned gate is closer to a stable bias than a highly dynamic controller.
- The source-length correlation should be interpreted cautiously because the export contains only 12 selected long-ish examples.
- Heatmaps and gate curves under `analysis/attention_gate/` are report-ready visual evidence.
