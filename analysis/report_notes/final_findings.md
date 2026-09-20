# Final Findings

## Research Question

本项目围绕 Multi30k 英德图像描述翻译任务，比较四个轻量级序列到序列模型：无 attention 的 LSTM、带 attention 的 LSTM、长度感知 gated attention 变体、小型 Transformer。核心问题不是追求最大模型效果，而是通过可控实验回答：attention 能否缓解 fixed-vector bottleneck，gated attention 是否带来可解释的新变化，解码策略和句长是否会改变模型排序。

## Quantitative Summary

| Model | Params | Best valid loss | Greedy BLEU | Greedy chrF | Selected decode | Selected test BLEU | Selected test chrF |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| LSTM no attention | 8.53M | 2.1720 | 20.47 | 43.88 | - | - | - |
| LSTM attention | 9.12M | 1.9562 | 27.31 | 49.55 | beam5_lp1p0 | 28.60 | 50.94 |
| Gated attention | 9.57M | 1.9569 | 25.97 | 49.60 | beam5_lp1p0 | 28.04 | 51.37 |
| Small Transformer | 9.51M | 1.8265 | 25.94 | 52.02 | beam3_lp1p0 | 26.85 | 52.91 |

## Main Claims

1. **Attention 明显缓解 fixed-vector bottleneck。** 相比无 attention LSTM，plain attention LSTM 的 test greedy BLEU 提升 6.84，chrF 提升 5.67，best valid loss 降低 0.2157。这是本项目最稳定、最容易支撑的主要结论。

2. **Gated attention 是一个有解释价值的小变体，但当前设置下没有稳定超过 plain attention。** 在 greedy test 上，gated attention 相比 plain attention BLEU 低 1.34，chrF 高 0.05；在 selected beam decoding 上，gated attention BLEU 低 0.55，chrF 高 0.43。它更适合作为“结构改动与可解释性分析”的贡献，而不是宣称全面性能提升。

3. **Small Transformer 的 valid loss 和 chrF 最强，但 BLEU 排名不最高。** Transformer best valid loss 为 1.8265，selected test chrF 比其他 selected 模型最高值高 1.54；但 selected test BLEU 为 26.85，低于 LSTM attention 的 28.60。报告里可以把它解释为：不同指标关注的语言质量侧面不同，模型排序会随指标改变。

4. **训练效率上 Transformer 优势明显。** Small Transformer 训练约 170.16 秒，而 gated attention 约 954.53 秒；在这台机器和当前配置下，gated attention 训练时间约为 Transformer 的 5.61 倍。

5. **解码策略会显著影响模型排序和绝对分数。** 所有进入解码网格的模型都在 validation 上从 beam search 和 length penalty 中获益，最终 test 只使用 validation 选出的配置，避免直接用 test split 调参。

## Decoding Results

| Model | Valid greedy BLEU | Valid selected BLEU | Valid selected chrF | Selected test BLEU | Selected test chrF | Selected config |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LSTM attention | 26.78 | 28.83 | 50.83 | 28.60 | 50.94 | beam5_lp1p0 |
| Gated attention | 27.47 | 29.59 | 51.27 | 28.04 | 51.37 | beam5_lp1p0 |
| Small Transformer | 26.05 | 27.31 | 52.22 | 26.85 | 52.91 | beam3_lp1p0 |

可写进报告的观察：beam search 在 validation 上整体优于 greedy；`length_penalty=1.0` 被三个模型的最佳 validation 配置选中，说明在本任务中适当补偿序列长度有助于提高 BLEU/chrF。但在 test 上，Transformer 仍然表现为 chrF 更强、BLEU 不最高，这说明解码优化没有完全消除模型之间的指标差异。

## Length And Generation Behavior

| Model | Bucket | Examples | BLEU | chrF | Hyp/ref length ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM attention | 1-10 | 283 | 30.19 | 52.33 | 0.949 |
| LSTM attention | 11-20 | 661 | 29.91 | 51.76 | 0.936 |
| LSTM attention | 21-30 | 54 | 15.81 | 42.07 | 0.975 |
| LSTM attention | 31+ | 2 | 29.44 | 57.85 | 0.984 |
| Gated attention | 1-10 | 283 | 32.50 | 53.82 | 0.942 |
| Gated attention | 11-20 | 661 | 28.73 | 51.81 | 0.941 |
| Gated attention | 21-30 | 54 | 14.85 | 43.52 | 1.014 |
| Gated attention | 31+ | 2 | 24.31 | 52.34 | 0.903 |
| Small Transformer | 1-10 | 283 | 32.90 | 57.60 | 0.983 |
| Small Transformer | 11-20 | 661 | 27.24 | 53.32 | 0.923 |
| Small Transformer | 21-30 | 54 | 12.36 | 40.64 | 0.914 |
| Small Transformer | 31+ | 2 | 19.87 | 51.64 | 0.855 |

核心观察：

- LSTM attention: selected BLEU 从短句桶 30.19 降到 21-30 桶 15.81，chrF 从 52.33 降到 42.07。
- Gated attention: selected BLEU 从短句桶 32.50 降到 21-30 桶 14.85，chrF 从 53.82 降到 43.52。
- Small Transformer: selected BLEU 从短句桶 32.90 降到 21-30 桶 12.36，chrF 从 57.60 降到 40.64。

`31+` 桶只有 2 个样本，因此只适合做个案观察，不适合给出强结论。整体上，模型在 21-30 token 的较长句上 BLEU/chrF 都明显下降，错误分析中也能看到长句更容易出现内容压缩、动作关系错误和修饰成分丢失。

生成长度方面，selected decoding 的 hyp/ref length ratio 在 0.934 到 0.948 之间，均略短于 reference。可在报告中结合 beam search 讨论：提高搜索质量并不等于生成更长句子，length penalty 主要改善了候选排序，而不是简单拉长输出。

## Attention And Gate

Gate 统计来自已导出的 12 个 gated attention 示例，共 181 个 gate 值。gate mean 为 0.700，范围为 0.599-0.776，与 source length 的 Pearson 相关系数为 -0.090。

这支持一个谨慎结论：当前 gated attention 确实学到了偏向 attention context 的控制信号，但 gate 在样例间变化不大，也没有随 source length 呈现强相关，更像是稳定偏置而不是明显动态的长度控制器。与此同时，gated attention 的平均 attention entropy 为 0.307，plain attention 为 0.412；这说明 gated 模型的导出样例 attention 更尖锐，但这种尖锐性没有直接转化为稳定 BLEU 优势。

## Qualitative Evidence

半自动挑选了 24 个 test case，覆盖模型分歧、长句、稀有词和强输出。错误标签分布如下：

| Error tag | Cases |
| --- | ---: |
| action error | 10 |
| correct/paraphrase | 10 |
| fluent but incomplete | 9 |
| repetition | 9 |
| rare word | 8 |
| long sentence compression | 7 |
| modifier loss | 5 |

建议在报告中选 4-6 个代表性案例展开：一个 attention 相比 no-attention 明显改进的案例，一个 Transformer 表达更流畅但 BLEU 不占优的案例，一个 gated attention gate/attention 可视化案例，一个长句压缩案例，一个稀有词或具体物体翻译错误案例。

## Limitations

- 本项目只有一次固定配置训练，没有做多随机种子统计，因此不能把小幅差距解释为显著性能差异。
- Gated attention 的 gate 分析只基于导出的 12 个示例，适合解释模型行为，不适合做总体统计检验。
- Multi30k 句子较短、领域较集中，结论主要适用于小规模图像描述翻译。
- BLEU 和 chrF 只能衡量自动指标，最终报告需要结合错误案例避免只看单一数值。

## Report Skeleton

1. Introduction: 说明 seq2seq、fixed-vector bottleneck、attention 的动机，以及本项目提出的 gated attention 分析问题。
2. Methods: 介绍四个模型、训练配置、评价指标、beam search 和 length penalty。
3. Quantitative Results: 放模型总表、解码表、训练曲线和指标柱状图。
4. Analysis: 放长度分桶、生成长度、attention heatmap、gate 曲线和错误案例。
5. Discussion: 强调 attention 的收益、gated 变体的解释价值、Transformer 的效率和指标差异。
6. Conclusion: 给出本项目的课程项目定位：复现基础模型，对比解码策略，并提出一个可解释的小变体及其负结果分析。
