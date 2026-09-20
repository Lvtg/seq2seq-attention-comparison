from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_FILES = [
    ("lstm_no_attention", "LSTM no attention", "lstm_no_attention_results.json"),
    ("lstm_attention", "LSTM attention", "lstm_attention_results.json"),
    ("gated_attention", "Gated attention", "gated_attention_results.json"),
    ("transformer_small", "Small Transformer", "transformer_small_results.json"),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Model Comparison\n\n")
        handle.write(
            "| Model | Parameters | Best epoch | Best valid loss | Training seconds | Greedy BLEU | Greedy chrF |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['parameters']} | {row['best_epoch']} | "
                f"{row['best_valid_loss']:.6f} | {row['total_training_seconds']:.2f} | "
                f"{row['greedy_bleu']:.4f} | {row['greedy_chrf']:.4f} |\n"
            )
        handle.write("\n")
        handle.write("Notes:\n\n")
        handle.write("- All scores use the shared tokenized Multi30k preprocessing.\n")
        handle.write("- Checkpoints and raw run logs remain local-only and are not tracked by git.\n")
        handle.write("- This table is generated from the milestone result JSON files under `artifacts/tables/`.\n")


def collect_rows(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    full_results = {}
    for model_id, label, filename in MODEL_FILES:
        result = read_json(results_dir / filename)
        training = result["training"]
        test = result["test"]
        row = {
            "model_id": model_id,
            "label": label,
            "parameters": training["parameters"],
            "best_epoch": training["best_epoch"],
            "best_valid_loss": training["best_valid_loss"],
            "final_epoch": training["final_epoch"],
            "final_train_loss": training["final_train_loss"],
            "final_valid_loss": training["final_valid_loss"],
            "total_training_seconds": training["total_training_seconds"],
            "greedy_bleu": test["bleu"],
            "greedy_chrf": test["chrf"],
        }
        rows.append(row)
        full_results[model_id] = result
    return rows, full_results


def plot_metric_bars(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    x_positions = list(range(len(rows)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("greedy_bleu", "Greedy BLEU", "#4c78a8"),
        ("greedy_chrf", "Greedy chrF", "#f58518"),
        ("best_valid_loss", "Best valid loss", "#54a24b"),
    ]
    for axis, (key, title, color) in zip(axes, metrics):
        values = [row[key] for row in rows]
        axis.bar(x_positions, values, color=color)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_metric_bars.png", dpi=200)
    plt.close(fig)


def plot_training_time(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row["total_training_seconds"] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#9c755f")
    axis.set_ylabel("Seconds")
    axis.set_title("Training Time")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.0f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "training_time_comparison.png", dpi=200)
    plt.close(fig)


def plot_valid_loss_curves(full_results: dict[str, Any], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for model_id, label, _ in MODEL_FILES:
        metrics = full_results[model_id]["epoch_metrics"]
        epochs = [item["epoch"] for item in metrics]
        valid_losses = [item["valid_loss"] for item in metrics]
        axis.plot(epochs, valid_losses, marker="o", linewidth=1.6, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation loss")
    axis.set_title("Validation Loss Curves")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "valid_loss_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-experiment model summary tables and figures.")
    parser.add_argument("--results-dir", default="artifacts/tables")
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir)
    metrics_dir = analysis_dir / "metrics"
    figures_dir = analysis_dir / "figures"

    rows, full_results = collect_rows(results_dir)
    fields = [
        "model_id",
        "label",
        "parameters",
        "best_epoch",
        "best_valid_loss",
        "final_epoch",
        "final_train_loss",
        "final_valid_loss",
        "total_training_seconds",
        "greedy_bleu",
        "greedy_chrf",
    ]
    write_csv(metrics_dir / "model_comparison.csv", rows, fields)
    write_json(metrics_dir / "model_comparison.json", rows)
    write_markdown(metrics_dir / "model_comparison.md", rows)
    plot_metric_bars(rows, figures_dir)
    plot_training_time(rows, figures_dir)
    plot_valid_loss_curves(full_results, figures_dir)

    print(
        json.dumps(
            {
                "models": len(rows),
                "metrics_dir": str(metrics_dir),
                "figures_dir": str(figures_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
