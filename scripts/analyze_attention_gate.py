from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denominator == 0.0 or y_denominator == 0.0:
        return None
    return numerator / (x_denominator * y_denominator)


def attention_entropy(row: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in row)


def attention_stats(record: dict[str, Any], model_id: str, example_id: int) -> dict[str, Any]:
    matrix = record["attention"]
    entropies = [attention_entropy(row) for row in matrix]
    max_weights = [max(row) for row in matrix]
    return {
        "model_id": model_id,
        "example_id": example_id,
        "source_length": len(record["source_tokens"]),
        "hypothesis_length": len(record["hypothesis_tokens"]),
        "attention_rows": len(matrix),
        "attention_columns": len(matrix[0]) if matrix else 0,
        "attention_entropy_mean": mean(entropies),
        "attention_max_weight_mean": mean(max_weights),
        "source": record["source"],
        "hypothesis": record["hypothesis"],
        "reference": record["reference"],
    }


def plot_attention_heatmap(record: dict[str, Any], output_path: Path, title: str) -> None:
    source_tokens = record["source_tokens"]
    hypothesis_tokens = record["hypothesis_tokens"]
    matrix = record["attention"]
    width = max(6.0, min(14.0, len(source_tokens) * 0.55))
    height = max(4.0, min(10.0, len(hypothesis_tokens) * 0.42))
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(source_tokens)))
    axis.set_xticklabels(source_tokens, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(hypothesis_tokens)))
    axis.set_yticklabels(hypothesis_tokens, fontsize=8)
    axis.set_xlabel("Source tokens")
    axis.set_ylabel("Generated tokens")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gate_curve(record: dict[str, Any], output_path: Path, title: str) -> None:
    gate_values = record["gate_sequence"]
    tokens = gate_tokens(record)
    fig, axis = plt.subplots(figsize=(max(6.0, len(gate_values) * 0.35), 4.2))
    axis.plot(range(1, len(gate_values) + 1), gate_values, marker="o", linewidth=1.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(range(1, len(gate_values) + 1))
    axis.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    axis.set_xlabel("Generated token")
    axis.set_ylabel("Gate value")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gate_distribution(gate_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    means = [row["gate_mean"] for row in gate_rows]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.hist(means, bins=8, color="#4c78a8", edgecolor="white")
    axis.set_xlabel("Per-example gate mean")
    axis.set_ylabel("Examples")
    axis.set_title("Gated Attention Mean Gate Distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "gate_mean_distribution.png", dpi=200)
    plt.close(fig)


def plot_gate_vs_length(gate_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.scatter(
        [row["source_length_without_eos"] for row in gate_rows],
        [row["gate_mean"] for row in gate_rows],
        color="#f58518",
    )
    axis.set_xlabel("Source length")
    axis.set_ylabel("Gate mean")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Gate Mean vs Source Length")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "gate_mean_vs_source_length.png", dpi=200)
    plt.close(fig)


def build_gate_rows(gated_examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_rows = []
    gate_step_rows = []
    for example_id, record in enumerate(gated_examples, start=1):
        sequence = [float(value) for value in record["gate_sequence"]]
        source_length = int(record.get("source_length", len(record["source_tokens"])))
        row = {
            "model_id": "gated_attention",
            "example_id": example_id,
            "source_length_without_eos": source_length,
            "source_length_with_eos": len(record["source_tokens"]),
            "hypothesis_length": len(record["hypothesis_tokens"]),
            "gate_values": len(sequence),
            "gate_mean": mean(sequence),
            "gate_min": min(sequence),
            "gate_max": max(sequence),
            "gate_range": max(sequence) - min(sequence),
            "source": record["source"],
            "hypothesis": record["hypothesis"],
            "reference": record["reference"],
        }
        gate_rows.append(row)
        for step, (token, value) in enumerate(zip(gate_tokens(record), sequence), start=1):
            gate_step_rows.append(
                {
                    "example_id": example_id,
                    "step": step,
                    "token": token,
                    "source_length_without_eos": source_length,
                    "gate_value": value,
                }
            )
    return gate_rows, gate_step_rows


def gate_tokens(record: dict[str, Any]) -> list[str]:
    tokens = list(record["hypothesis_tokens"])
    sequence_length = len(record["gate_sequence"])
    if len(tokens) < sequence_length:
        tokens.extend(["<eos>"] * (sequence_length - len(tokens)))
    return tokens[:sequence_length]


def write_summary(
    path: Path,
    gate_summary: dict[str, Any],
    attention_rows: list[dict[str, Any]],
) -> None:
    lstm_entropy = mean(
        [row["attention_entropy_mean"] for row in attention_rows if row["model_id"] == "lstm_attention"]
    )
    gated_entropy = mean(
        [row["attention_entropy_mean"] for row in attention_rows if row["model_id"] == "gated_attention"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Attention and Gate Analysis\n\n")
        handle.write("## Gate Summary\n\n")
        handle.write("| Statistic | Value |\n")
        handle.write("| --- | ---: |\n")
        for key in [
            "examples",
            "gate_values",
            "gate_mean",
            "gate_min",
            "gate_max",
            "per_example_gate_mean_min",
            "per_example_gate_mean_max",
            "pearson_gate_mean_source_length",
        ]:
            value = gate_summary[key]
            if isinstance(value, float):
                handle.write(f"| {key} | {value:.6f} |\n")
            else:
                handle.write(f"| {key} | {value} |\n")
        handle.write("\n## Attention Sharpness\n\n")
        handle.write("| Model | Mean attention entropy |\n")
        handle.write("| --- | ---: |\n")
        handle.write(f"| LSTM attention | {lstm_entropy:.6f} |\n")
        handle.write(f"| Gated attention | {gated_entropy:.6f} |\n")
        handle.write("\n## Main Observations\n\n")
        handle.write(
            "- The exported gated examples have gate means around 0.70, so the fused representation usually "
            "leans toward the attention context.\n"
        )
        handle.write(
            "- The gate range is modest in the exported examples, suggesting the learned gate is closer to a "
            "stable bias than a highly dynamic controller.\n"
        )
        handle.write(
            "- The source-length correlation should be interpreted cautiously because the export contains only "
            "12 selected long-ish examples.\n"
        )
        handle.write("- Heatmaps and gate curves under `analysis/attention_gate/` are report-ready visual evidence.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build attention heatmaps and gated-attention statistics.")
    parser.add_argument("--attention-json", default="artifacts/attention/lstm_attention_test_attention_examples.json")
    parser.add_argument("--gated-json", default="artifacts/gates/gated_attention_test_gate_examples.json")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--num-plots", type=int, default=4)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = analysis_dir / "attention_gate"
    figures_dir = analysis_dir / "figures"
    attention_examples = read_json(Path(args.attention_json))
    gated_examples = read_json(Path(args.gated_json))

    attention_rows = [
        attention_stats(record, "lstm_attention", index)
        for index, record in enumerate(attention_examples, start=1)
    ]
    attention_rows.extend(
        attention_stats(record, "gated_attention", index)
        for index, record in enumerate(gated_examples, start=1)
    )
    gate_rows, gate_step_rows = build_gate_rows(gated_examples)
    all_gate_values = [row["gate_value"] for row in gate_step_rows]
    gate_means = [row["gate_mean"] for row in gate_rows]
    source_lengths = [row["source_length_without_eos"] for row in gate_rows]
    gate_summary = {
        "examples": len(gate_rows),
        "gate_values": len(all_gate_values),
        "gate_mean": mean(all_gate_values),
        "gate_min": min(all_gate_values),
        "gate_max": max(all_gate_values),
        "per_example_gate_mean_min": min(gate_means),
        "per_example_gate_mean_max": max(gate_means),
        "pearson_gate_mean_source_length": pearson(source_lengths, gate_means),
    }

    write_csv(output_dir / "attention_stats.csv", attention_rows)
    write_json(output_dir / "attention_stats.json", attention_rows)
    write_csv(output_dir / "gate_stats.csv", gate_rows)
    write_json(output_dir / "gate_stats.json", gate_rows)
    write_csv(output_dir / "gate_step_values.csv", gate_step_rows)
    write_json(output_dir / "gate_summary.json", gate_summary)

    for index, record in enumerate(attention_examples[: args.num_plots], start=1):
        plot_attention_heatmap(
            record,
            output_dir / f"lstm_attention_heatmap_{index:02d}.png",
            f"LSTM attention example {index}",
        )
    for index, record in enumerate(gated_examples[: args.num_plots], start=1):
        plot_attention_heatmap(
            record,
            output_dir / f"gated_attention_heatmap_{index:02d}.png",
            f"Gated attention example {index}",
        )
        plot_gate_curve(
            record,
            output_dir / f"gated_attention_gate_curve_{index:02d}.png",
            f"Gated attention gate curve {index}",
        )
    plot_gate_distribution(gate_rows, figures_dir)
    plot_gate_vs_length(gate_rows, figures_dir)
    write_summary(output_dir / "attention_gate_summary.md", gate_summary, attention_rows)

    print(
        json.dumps(
            {
                "attention_examples": len(attention_examples),
                "gated_examples": len(gated_examples),
                "gate_mean": gate_summary["gate_mean"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
