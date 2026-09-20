from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def f2(value: float) -> str:
    return f"{value:.2f}"


def f3(value: float) -> str:
    return f"{value:.3f}"


def row_by_id(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for row in rows:
        if row["model_id"] == model_id:
            return row
    raise KeyError(model_id)


def selected_by_id(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["model_id"] == model_id:
            return row
    return None


def selected_valid_by_id(rows: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        if row["model_id"] == selected["model_id"] and row["decode_label"] == selected["decode_label"]:
            return row
    raise KeyError((selected["model_id"], selected["decode_label"]))


def greedy_valid_by_id(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for row in rows:
        if row["model_id"] == model_id and row["decode_label"] == "greedy":
            return row
    raise KeyError(model_id)


def parse_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def parse_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def metric_table(
    model_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "| Model | Params | Best valid loss | Greedy BLEU | Greedy chrF | Selected decode | Selected test BLEU | Selected test chrF |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in model_rows:
        selected = selected_by_id(selected_rows, row["model_id"])
        if selected is None:
            decode = "-"
            selected_bleu = "-"
            selected_chrf = "-"
        else:
            decode = selected["decode_label"]
            selected_bleu = f2(float(selected["bleu"]))
            selected_chrf = f2(float(selected["chrf"]))
        lines.append(
            f"| {row['label']} | {row['parameters'] / 1_000_000:.2f}M | "
            f"{float(row['best_valid_loss']):.4f} | {f2(float(row['greedy_bleu']))} | "
            f"{f2(float(row['greedy_chrf']))} | {decode} | {selected_bleu} | {selected_chrf} |"
        )
    return "\n".join(lines)


def decoding_table(
    valid_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "| Model | Valid greedy BLEU | Valid selected BLEU | Valid selected chrF | Selected test BLEU | Selected test chrF | Selected config |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for selected in selected_rows:
        greedy = greedy_valid_by_id(valid_rows, selected["model_id"])
        valid = selected_valid_by_id(valid_rows, selected)
        lines.append(
            f"| {selected['label']} | {f2(float(greedy['bleu']))} | {f2(float(valid['bleu']))} | "
            f"{f2(float(valid['chrf']))} | {f2(float(selected['bleu']))} | "
            f"{f2(float(selected['chrf']))} | {selected['decode_label']} |"
        )
    return "\n".join(lines)


def length_table(length_rows: list[dict[str, Any]]) -> str:
    selected = [row for row in length_rows if row["result_set"] == "selected"]
    lines = [
        "| Model | Bucket | Examples | BLEU | chrF | Hyp/ref length ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    order = {"lstm_attention": 0, "gated_attention": 1, "transformer_small": 2}
    bucket_order = {"1-10": 0, "11-20": 1, "21-30": 2, "31+": 3}
    selected.sort(key=lambda row: (order[row["model_id"]], bucket_order[row["bucket"]]))
    for row in selected:
        lines.append(
            f"| {row['label']} | {row['bucket']} | {parse_int(row, 'examples')} | "
            f"{f2(parse_float(row, 'bleu'))} | {f2(parse_float(row, 'chrf'))} | "
            f"{f3(parse_float(row, 'hyp_ref_length_ratio'))} |"
        )
    return "\n".join(lines)


def count_error_tags(case_rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in case_rows:
        for tag in row["error_tags"].split("; "):
            if tag:
                counts[tag] += 1
    return counts


def error_tag_table(counts: Counter[str]) -> str:
    lines = ["| Error tag | Cases |", "| --- | ---: |"]
    for tag, count in counts.most_common():
        lines.append(f"| {tag} | {count} |")
    return "\n".join(lines)


def build_final_findings(analysis_dir: Path) -> str:
    model_rows = read_json(analysis_dir / "metrics" / "model_comparison.json")
    valid_rows = read_csv(analysis_dir / "decoding" / "decoding_valid_grid.csv")
    selected_rows = read_json(analysis_dir / "decoding" / "decoding_test_selected.json")
    length_rows = read_json(analysis_dir / "length_buckets" / "length_bucket_results.json")
    generation_rows = read_json(analysis_dir / "length_buckets" / "generation_length_stats.json")
    gate_summary = read_json(analysis_dir / "attention_gate" / "gate_summary.json")
    attention_rows = read_json(analysis_dir / "attention_gate" / "attention_stats.json")
    case_rows = read_csv(analysis_dir / "error_analysis" / "case_samples.csv")

    no_attn = row_by_id(model_rows, "lstm_no_attention")
    attn = row_by_id(model_rows, "lstm_attention")
    gated = row_by_id(model_rows, "gated_attention")
    transformer = row_by_id(model_rows, "transformer_small")
    attn_selected = selected_by_id(selected_rows, "lstm_attention")
    gated_selected = selected_by_id(selected_rows, "gated_attention")
    transformer_selected = selected_by_id(selected_rows, "transformer_small")
    if attn_selected is None or gated_selected is None or transformer_selected is None:
        raise ValueError("Selected decoding rows are incomplete.")

    attention_delta_bleu = attn["greedy_bleu"] - no_attn["greedy_bleu"]
    attention_delta_chrf = attn["greedy_chrf"] - no_attn["greedy_chrf"]
    attention_delta_valid_loss = no_attn["best_valid_loss"] - attn["best_valid_loss"]
    gated_delta_bleu = gated["greedy_bleu"] - attn["greedy_bleu"]
    gated_delta_chrf = gated["greedy_chrf"] - attn["greedy_chrf"]
    gated_selected_delta_bleu = gated_selected["bleu"] - attn_selected["bleu"]
    gated_selected_delta_chrf = gated_selected["chrf"] - attn_selected["chrf"]
    transformer_chrf_advantage = transformer_selected["chrf"] - max(
        attn_selected["chrf"], gated_selected["chrf"]
    )
    transformer_speed_vs_gated = gated["total_training_seconds"] / transformer["total_training_seconds"]

    selected_length = [
        row for row in length_rows if row["result_set"] == "selected" and row["bucket"] != "31+"
    ]
    long_bucket_notes = []
    for model_id in ["lstm_attention", "gated_attention", "transformer_small"]:
        rows = {row["bucket"]: row for row in selected_length if row["model_id"] == model_id}
        short = rows["1-10"]
        long = rows["21-30"]
        long_bucket_notes.append(
            f"- {short['label']}: selected BLEU 从短句桶 {f2(float(short['bleu']))} "
            f"降到 21-30 桶 {f2(float(long['bleu']))}，chrF 从 "
            f"{f2(float(short['chrf']))} 降到 {f2(float(long['chrf']))}。"
        )

    generation_selected = [
        row for row in generation_rows if row["result_set"] == "selected"
    ]
    shortest_selected = min(generation_selected, key=lambda row: row["hyp_ref_length_ratio"])
    longest_selected = max(generation_selected, key=lambda row: row["hyp_ref_length_ratio"])

    attention_entropy = {}
    for model_id in ["lstm_attention", "gated_attention"]:
        rows = [row for row in attention_rows if row["model_id"] == model_id]
        attention_entropy[model_id] = sum(row["attention_entropy_mean"] for row in rows) / len(rows)

    error_counts = count_error_tags(case_rows)
    metric_lines = metric_table(model_rows, selected_rows)
    decode_lines = decoding_table(valid_rows, selected_rows)
    length_lines = length_table(length_rows)
    error_lines = error_tag_table(error_counts)

    return f"""# Final Findings

## Research Question

本项目围绕 Multi30k 英德图像描述翻译任务，比较四个轻量级序列到序列模型：无 attention 的 LSTM、带 attention 的 LSTM、长度感知 gated attention 变体、小型 Transformer。核心问题不是追求最大模型效果，而是通过可控实验回答：attention 能否缓解 fixed-vector bottleneck，gated attention 是否带来可解释的新变化，解码策略和句长是否会改变模型排序。

## Quantitative Summary

{metric_lines}

## Main Claims

1. **Attention 明显缓解 fixed-vector bottleneck。** 相比无 attention LSTM，plain attention LSTM 的 test greedy BLEU 提升 {f2(attention_delta_bleu)}，chrF 提升 {f2(attention_delta_chrf)}，best valid loss 降低 {attention_delta_valid_loss:.4f}。这是本项目最稳定、最容易支撑的主要结论。

2. **Gated attention 是一个有解释价值的小变体，但当前设置下没有稳定超过 plain attention。** 在 greedy test 上，gated attention 相比 plain attention BLEU 低 {f2(abs(gated_delta_bleu))}，chrF 高 {f2(gated_delta_chrf)}；在 selected beam decoding 上，gated attention BLEU 低 {f2(abs(gated_selected_delta_bleu))}，chrF 高 {f2(gated_selected_delta_chrf)}。它更适合作为“结构改动与可解释性分析”的贡献，而不是宣称全面性能提升。

3. **Small Transformer 的 valid loss 和 chrF 最强，但 BLEU 排名不最高。** Transformer best valid loss 为 {transformer['best_valid_loss']:.4f}，selected test chrF 比其他 selected 模型最高值高 {f2(transformer_chrf_advantage)}；但 selected test BLEU 为 {f2(transformer_selected['bleu'])}，低于 LSTM attention 的 {f2(attn_selected['bleu'])}。报告里可以把它解释为：不同指标关注的语言质量侧面不同，模型排序会随指标改变。

4. **训练效率上 Transformer 优势明显。** Small Transformer 训练约 {f2(transformer['total_training_seconds'])} 秒，而 gated attention 约 {f2(gated['total_training_seconds'])} 秒；在这台机器和当前配置下，gated attention 训练时间约为 Transformer 的 {f2(transformer_speed_vs_gated)} 倍。

5. **解码策略会显著影响模型排序和绝对分数。** 所有进入解码网格的模型都在 validation 上从 beam search 和 length penalty 中获益，最终 test 只使用 validation 选出的配置，避免直接用 test split 调参。

## Decoding Results

{decode_lines}

可写进报告的观察：beam search 在 validation 上整体优于 greedy；`length_penalty=1.0` 被三个模型的最佳 validation 配置选中，说明在本任务中适当补偿序列长度有助于提高 BLEU/chrF。但在 test 上，Transformer 仍然表现为 chrF 更强、BLEU 不最高，这说明解码优化没有完全消除模型之间的指标差异。

## Length And Generation Behavior

{length_lines}

核心观察：

{chr(10).join(long_bucket_notes)}

`31+` 桶只有 2 个样本，因此只适合做个案观察，不适合给出强结论。整体上，模型在 21-30 token 的较长句上 BLEU/chrF 都明显下降，错误分析中也能看到长句更容易出现内容压缩、动作关系错误和修饰成分丢失。

生成长度方面，selected decoding 的 hyp/ref length ratio 在 {f3(shortest_selected['hyp_ref_length_ratio'])} 到 {f3(longest_selected['hyp_ref_length_ratio'])} 之间，均略短于 reference。可在报告中结合 beam search 讨论：提高搜索质量并不等于生成更长句子，length penalty 主要改善了候选排序，而不是简单拉长输出。

## Attention And Gate

Gate 统计来自已导出的 {gate_summary['examples']} 个 gated attention 示例，共 {gate_summary['gate_values']} 个 gate 值。gate mean 为 {f3(gate_summary['gate_mean'])}，范围为 {f3(gate_summary['gate_min'])}-{f3(gate_summary['gate_max'])}，与 source length 的 Pearson 相关系数为 {f3(gate_summary['pearson_gate_mean_source_length'])}。

这支持一个谨慎结论：当前 gated attention 确实学到了偏向 attention context 的控制信号，但 gate 在样例间变化不大，也没有随 source length 呈现强相关，更像是稳定偏置而不是明显动态的长度控制器。与此同时，gated attention 的平均 attention entropy 为 {f3(attention_entropy['gated_attention'])}，plain attention 为 {f3(attention_entropy['lstm_attention'])}；这说明 gated 模型的导出样例 attention 更尖锐，但这种尖锐性没有直接转化为稳定 BLEU 优势。

## Qualitative Evidence

半自动挑选了 {len(case_rows)} 个 test case，覆盖模型分歧、长句、稀有词和强输出。错误标签分布如下：

{error_lines}

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
"""


def figure_caption(path: Path) -> str:
    name = path.name
    captions = {
        "model_metric_bars.png": "四个模型的 greedy BLEU、greedy chrF 与 best valid loss 对比。",
        "training_time_comparison.png": "四个模型的训练耗时对比。",
        "valid_loss_curves.png": "四个模型 validation loss 随 epoch 变化曲线。",
        "generation_length_ratio.png": "不同模型和解码策略的 hypothesis/reference 长度比例。",
        "length_bucket_bleu_greedy.png": "Greedy decoding 下按 source length 分桶的 BLEU。",
        "length_bucket_bleu_selected.png": "Selected decoding 下按 source length 分桶的 BLEU。",
        "length_bucket_chrf_greedy.png": "Greedy decoding 下按 source length 分桶的 chrF。",
        "length_bucket_chrf_selected.png": "Selected decoding 下按 source length 分桶的 chrF。",
        "gate_mean_distribution.png": "Gated attention 示例的平均 gate 分布。",
        "gate_mean_vs_source_length.png": "Gated attention 平均 gate 与 source length 的关系。",
        "error_tag_counts.png": "定性错误分析样例的错误标签计数。",
    }
    return captions.get(name, "报告可选补充图。")


def build_index(analysis_dir: Path) -> str:
    table_entries = [
        (
            "analysis/metrics/model_comparison.md",
            "Table 1",
            "四个模型的参数量、best valid loss、训练时间、greedy BLEU 和 chrF。",
        ),
        (
            "analysis/decoding/decoding_valid_grid.csv",
            "Table 2",
            "Validation split 上 greedy/beam/length penalty 解码网格。",
        ),
        (
            "analysis/decoding/decoding_test_selected.csv",
            "Table 3",
            "基于 validation 选择后，在 test split 上确认的最终解码结果。",
        ),
        (
            "analysis/length_buckets/length_bucket_results.csv",
            "Table 4",
            "按 source length 分桶的 BLEU、chrF 和长度比例。",
        ),
        (
            "analysis/length_buckets/generation_length_stats.csv",
            "Table 5",
            "不同模型输出长度、短于/长于 reference 的比例。",
        ),
        (
            "analysis/attention_gate/gate_stats.csv",
            "Table 6",
            "Gated attention 每个导出样例的 gate 统计。",
        ),
        (
            "analysis/error_analysis/case_samples.md",
            "Table 7",
            "定性错误案例，可挑选部分放入正文，其余放附录。",
        ),
    ]

    figure_files = sorted((analysis_dir / "figures").glob("*.png"))
    attention_files = sorted((analysis_dir / "attention_gate").glob("*.png"))

    lines = ["# Figure And Table Index", ""]
    lines.append("## Tables")
    lines.append("")
    lines.append("| File | Suggested label | Suggested use |")
    lines.append("| --- | --- | --- |")
    for file_path, label, use in table_entries:
        lines.append(f"| `{file_path}` | {label} | {use} |")

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("| File | Suggested label | Suggested caption |")
    lines.append("| --- | --- | --- |")
    for index, path in enumerate(figure_files, start=1):
        rel_path = path.as_posix()
        lines.append(f"| `{rel_path}` | Figure {index} | {figure_caption(path)} |")

    start = len(figure_files) + 1
    for offset, path in enumerate(attention_files, start=start):
        rel_path = path.as_posix()
        if "heatmap" in path.name:
            caption = "Attention 权重热力图，用于展示目标 token 对源 token 的对齐。"
        elif "gate_curve" in path.name:
            caption = "Gated attention 的 gate-over-time 曲线，用于观察每个生成步骤的 gate 变化。"
        else:
            caption = "Attention/gate 补充可视化。"
        lines.append(f"| `{rel_path}` | Figure {offset} | {caption} |")

    lines.append("")
    lines.append("## Recommended Main-Text Figures")
    lines.append("")
    lines.append("- `analysis/figures/model_metric_bars.png`")
    lines.append("- `analysis/figures/valid_loss_curves.png`")
    lines.append("- `analysis/figures/length_bucket_bleu_selected.png`")
    lines.append("- `analysis/figures/generation_length_ratio.png`")
    lines.append("- `analysis/figures/gate_mean_vs_source_length.png`")
    lines.append("- `analysis/figures/error_tag_counts.png`")
    lines.append("")
    lines.append("Attention heatmap 和 gate curve 建议各选 1-2 张放正文，其余作为附录材料。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final report notes from analysis outputs.")
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    notes_dir = analysis_dir / "report_notes"
    final_findings = build_final_findings(analysis_dir)
    figure_index = build_index(analysis_dir)
    write_markdown(notes_dir / "final_findings.md", final_findings)
    write_markdown(notes_dir / "figure_table_index.md", figure_index)

    print(
        json.dumps(
            {
                "final_findings": str(notes_dir / "final_findings.md"),
                "figure_table_index": str(notes_dir / "figure_table_index.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
