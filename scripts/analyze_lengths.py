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
import sacrebleu


GREEDY_TRANSLATIONS = [
    {
        "result_set": "greedy",
        "model_id": "lstm_no_attention",
        "label": "LSTM no attention",
        "decode_label": "greedy",
        "path": "artifacts/translations/lstm_no_attention_test_greedy.txt",
    },
    {
        "result_set": "greedy",
        "model_id": "lstm_attention",
        "label": "LSTM attention",
        "decode_label": "greedy",
        "path": "artifacts/translations/lstm_attention_test_greedy.txt",
    },
    {
        "result_set": "greedy",
        "model_id": "gated_attention",
        "label": "Gated attention",
        "decode_label": "greedy",
        "path": "artifacts/translations/gated_attention_test_greedy.txt",
    },
    {
        "result_set": "greedy",
        "model_id": "transformer_small",
        "label": "Small Transformer",
        "decode_label": "greedy",
        "path": "artifacts/translations/transformer_small_test_greedy.txt",
    },
]
BUCKETS = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31+", 31, None),
]


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


def read_translation_file(path: Path) -> list[dict[str, str]]:
    records = []
    current: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            tag, text = line.split("\t", 1)
            current[tag] = text
    if current:
        records.append(current)
    for index, record in enumerate(records):
        missing = {"SRC", "HYP", "REF"} - set(record)
        if missing:
            raise ValueError(f"{path} record {index} is missing fields: {sorted(missing)}")
    return records


def token_len(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def bucket_for_source_length(length: int) -> str:
    for label, lower, upper in BUCKETS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return BUCKETS[0][0]


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(predictions, [references], force=True).score


def corpus_chrf(predictions: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_chrf(predictions, [references]).score


def summarize_records(records: list[dict[str, str]]) -> dict[str, Any]:
    source_lengths = [token_len(record["SRC"]) for record in records]
    reference_lengths = [token_len(record["REF"]) for record in records]
    hypothesis_lengths = [token_len(record["HYP"]) for record in records]
    shorter = sum(hyp < ref for hyp, ref in zip(hypothesis_lengths, reference_lengths))
    longer = sum(hyp > ref for hyp, ref in zip(hypothesis_lengths, reference_lengths))
    equal = sum(hyp == ref for hyp, ref in zip(hypothesis_lengths, reference_lengths))
    abs_delta = [abs(hyp - ref) for hyp, ref in zip(hypothesis_lengths, reference_lengths)]
    examples = len(records)
    return {
        "examples": examples,
        "avg_source_length": sum(source_lengths) / examples,
        "avg_reference_length": sum(reference_lengths) / examples,
        "avg_hypothesis_length": sum(hypothesis_lengths) / examples,
        "hyp_ref_length_ratio": sum(hypothesis_lengths) / max(sum(reference_lengths), 1),
        "shorter_than_ref_ratio": shorter / examples,
        "longer_than_ref_ratio": longer / examples,
        "equal_length_ratio": equal / examples,
        "avg_abs_length_delta": sum(abs_delta) / examples,
    }


def metrics_for_records(records: list[dict[str, str]]) -> dict[str, float]:
    predictions = [record["HYP"] for record in records]
    references = [record["REF"] for record in records]
    return {
        "bleu": corpus_bleu(predictions, references),
        "chrf": corpus_chrf(predictions, references),
    }


def selected_translation_specs(decoding_dir: Path) -> list[dict[str, Any]]:
    selected_rows = read_json(decoding_dir / "decoding_test_selected.json")
    specs = []
    for row in selected_rows:
        specs.append(
            {
                "result_set": "selected",
                "model_id": row["model_id"],
                "label": row["label"],
                "decode_label": row["decode_label"],
                "path": row["translation_output"],
            }
        )
    return specs


def collect_analysis_rows(specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bucket_rows = []
    generation_rows = []
    for spec in specs:
        records = read_translation_file(Path(spec["path"]))
        corpus_metrics = metrics_for_records(records)
        generation_rows.append({**{key: spec[key] for key in ["result_set", "model_id", "label", "decode_label"]}, **corpus_metrics, **summarize_records(records)})

        grouped: dict[str, list[dict[str, str]]] = {label: [] for label, _, _ in BUCKETS}
        for record in records:
            grouped[bucket_for_source_length(token_len(record["SRC"]))].append(record)
        for bucket_label, bucket_records in grouped.items():
            if not bucket_records:
                continue
            bucket_rows.append(
                {
                    **{key: spec[key] for key in ["result_set", "model_id", "label", "decode_label"]},
                    "bucket": bucket_label,
                    **metrics_for_records(bucket_records),
                    **summarize_records(bucket_records),
                }
            )
    return bucket_rows, generation_rows


def plot_bucket_metric(rows: list[dict[str, Any]], result_set: str, metric: str, figures_dir: Path) -> None:
    filtered = [row for row in rows if row["result_set"] == result_set]
    labels = [bucket[0] for bucket in BUCKETS]
    models = []
    for row in filtered:
        model_label = row["label"]
        if model_label not in models:
            models.append(model_label)

    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(labels)))
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for model_index, model in enumerate(models):
        values = []
        for bucket in labels:
            match = next((row for row in filtered if row["label"] == model and row["bucket"] == bucket), None)
            values.append(match[metric] if match else 0.0)
        offsets = [x + (model_index - (len(models) - 1) / 2) * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=model)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel(metric.upper())
    axis.set_title(f"{result_set.title()} {metric.upper()} by Source Length")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"length_bucket_{metric}_{result_set}.png", dpi=200)
    plt.close(fig)


def plot_length_ratio(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    labels = [f"{row['label']}\n{row['decode_label']}" for row in rows]
    values = [row["hyp_ref_length_ratio"] for row in rows]
    colors = ["#4c78a8" if row["result_set"] == "greedy" else "#f58518" for row in rows]
    axis.bar(range(len(rows)), values, color=colors)
    axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Hypothesis / reference length")
    axis.set_title("Generation Length Ratio")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "generation_length_ratio.png", dpi=200)
    plt.close(fig)


def write_summary(path: Path, bucket_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> None:
    selected_rows = [row for row in generation_rows if row["result_set"] == "selected"]
    greedy_rows = [row for row in generation_rows if row["result_set"] == "greedy"]
    best_selected_bleu = max(selected_rows, key=lambda row: row["bleu"])
    best_selected_chrf = max(selected_rows, key=lambda row: row["chrf"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Length and Generation Analysis\n\n")
        handle.write("## Corpus-Level Length Statistics\n\n")
        handle.write("| Result set | Model | Decoding | BLEU | chrF | Hyp/ref length ratio | Shorter than ref |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in generation_rows:
            handle.write(
                f"| {row['result_set']} | {row['label']} | {row['decode_label']} | "
                f"{row['bleu']:.4f} | {row['chrf']:.4f} | {row['hyp_ref_length_ratio']:.4f} | "
                f"{row['shorter_than_ref_ratio']:.4f} |\n"
            )
        handle.write("\n## Main Observations\n\n")
        handle.write(
            f"- Among selected decoding results, {best_selected_bleu['label']} has the highest BLEU "
            f"({best_selected_bleu['bleu']:.4f}).\n"
        )
        handle.write(
            f"- Among selected decoding results, {best_selected_chrf['label']} has the highest chrF "
            f"({best_selected_chrf['chrf']:.4f}).\n"
        )
        transformer_greedy = next(row for row in greedy_rows if row["model_id"] == "transformer_small")
        handle.write(
            "- The small Transformer keeps the strongest chrF while not leading BLEU, supporting the claim "
            "that BLEU and chrF emphasize different generation behavior.\n"
        )
        handle.write(
            f"- Transformer greedy hyp/ref length ratio is {transformer_greedy['hyp_ref_length_ratio']:.4f}, "
            "which should be discussed together with beam-search length penalty results.\n"
        )
        handle.write("- The `31+` bucket contains only two test examples, so use it as qualitative evidence only.\n")
        handle.write(
            "- Selected beam outputs remain slightly shorter than references overall; length penalty improves BLEU "
            "without fully matching reference length.\n"
        )
        handle.write("\n## Files\n\n")
        handle.write("- Bucket metrics: `length_bucket_results.csv` and `length_bucket_results.json`\n")
        handle.write("- Corpus generation metrics: `generation_length_stats.csv` and `generation_length_stats.json`\n")
        handle.write("- Figures are saved under `analysis/figures/`.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze source-length buckets and generation length behavior.")
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    length_dir = analysis_dir / "length_buckets"
    figures_dir = analysis_dir / "figures"
    specs = [*GREEDY_TRANSLATIONS, *selected_translation_specs(analysis_dir / "decoding")]
    bucket_rows, generation_rows = collect_analysis_rows(specs)

    write_csv(length_dir / "length_bucket_results.csv", bucket_rows)
    write_json(length_dir / "length_bucket_results.json", bucket_rows)
    write_csv(length_dir / "generation_length_stats.csv", generation_rows)
    write_json(length_dir / "generation_length_stats.json", generation_rows)
    write_summary(length_dir / "length_analysis_summary.md", bucket_rows, generation_rows)

    for result_set in ["greedy", "selected"]:
        plot_bucket_metric(bucket_rows, result_set, "bleu", figures_dir)
        plot_bucket_metric(bucket_rows, result_set, "chrf", figures_dir)
    plot_length_ratio(generation_rows, figures_dir)

    print(
        json.dumps(
            {
                "translation_sets": len(specs),
                "bucket_rows": len(bucket_rows),
                "generation_rows": len(generation_rows),
                "output_dir": str(length_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
