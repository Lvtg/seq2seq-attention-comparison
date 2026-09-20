from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "(", ")", "\"", "'"}
STOPWORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "of",
    "and",
    "or",
    "to",
    "with",
    "is",
    "are",
    "his",
    "her",
    "their",
    "this",
    "that",
}
ACTION_TOKENS = {
    "running",
    "walking",
    "standing",
    "sitting",
    "playing",
    "jumping",
    "holding",
    "catching",
    "catch",
    "kicking",
    "breaking",
    "fixing",
    "photographs",
    "photographing",
    "wearing",
    "riding",
    "climbing",
    "dancing",
    "swimming",
    "looking",
    "talking",
}
MODIFIER_TOKENS = {
    "red",
    "blue",
    "green",
    "white",
    "black",
    "orange",
    "yellow",
    "dark",
    "light",
    "young",
    "old",
    "small",
    "large",
    "winter",
    "striped",
    "colored",
    "beautiful",
}
MODEL_TRANSLATIONS = [
    {
        "column": "lstm_no_attention_greedy",
        "model_id": "lstm_no_attention",
        "label": "LSTM no attention greedy",
        "path": "artifacts/translations/lstm_no_attention_test_greedy.txt",
    },
    {
        "column": "lstm_attention_selected",
        "model_id": "lstm_attention",
        "label": "LSTM attention beam5_lp1p0",
        "path": "analysis/decoding/translations/test/lstm_attention_test_beam5_lp1p0.txt",
    },
    {
        "column": "gated_attention_selected",
        "model_id": "gated_attention",
        "label": "Gated attention beam5_lp1p0",
        "path": "analysis/decoding/translations/test/gated_attention_test_beam5_lp1p0.txt",
    },
    {
        "column": "transformer_selected",
        "model_id": "transformer_small",
        "label": "Small Transformer beam3_lp1p0",
        "path": "analysis/decoding/translations/test/transformer_small_test_beam3_lp1p0.txt",
    },
]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


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
    return records


def content_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text) if token.lower() not in STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    pred_counter = Counter(content_tokens(prediction))
    ref_counter = Counter(content_tokens(reference))
    if not pred_counter or not ref_counter:
        return 0.0
    overlap = sum((pred_counter & ref_counter).values())
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(ref_counter.values())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def source_bucket(length: int) -> str:
    if length <= 10:
        return "1-10"
    if length <= 20:
        return "11-20"
    if length <= 30:
        return "21-30"
    return "31+"


def repeated_pattern(text: str) -> bool:
    tokens = content_tokens(text)
    counts = Counter(tokens)
    if any(count >= 3 for token, count in counts.items() if token not in STOPWORDS):
        return True
    bigrams = list(zip(tokens, tokens[1:]))
    return any(count >= 2 for count in Counter(bigrams).values())


def load_source_frequencies(train_path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in read_jsonl(train_path):
        counter.update(token for token in record["src_tokens"] if token not in PUNCTUATION and token not in STOPWORDS)
    return counter


def rare_tokens(source: str, source_freq: Counter[str]) -> list[str]:
    tokens = [token.lower() for token in source.split() if token.lower() not in PUNCTUATION and token.lower() not in STOPWORDS]
    if not source_freq:
        return []
    return sorted({token for token in tokens if source_freq[token] <= 3})


def classify_case(
    source: str,
    reference: str,
    hypotheses: dict[str, str],
    f1_scores: dict[str, float],
    rare_source_tokens: list[str],
) -> list[str]:
    tags = []
    source_tokens = set(source.lower().split())
    source_length = len(source.split())
    ref_len = max(len(reference.split()), 1)
    hyp_lengths = [len(text.split()) for text in hypotheses.values()]
    best_f1 = max(f1_scores.values())
    mean_f1 = sum(f1_scores.values()) / len(f1_scores)

    if rare_source_tokens and mean_f1 < 0.75:
        tags.append("rare word")
    if source_tokens & ACTION_TOKENS and mean_f1 < 0.65:
        tags.append("action error")
    if source_tokens & MODIFIER_TOKENS and best_f1 < 0.75:
        tags.append("modifier loss")
    if any(repeated_pattern(text) for text in hypotheses.values()):
        tags.append("repetition")
    if source_length >= 21 and any(length / ref_len < 0.90 for length in hyp_lengths):
        tags.append("long sentence compression")
    if best_f1 < 0.60 or any(length / ref_len < 0.75 for length in hyp_lengths):
        tags.append("fluent but incomplete")
    if best_f1 >= 0.80:
        tags.append("correct/paraphrase")
    if not tags:
        tags.append("fluent but incomplete")
    return tags


def build_case_pool(source_freq: Counter[str]) -> list[dict[str, Any]]:
    translations = [read_translation_file(Path(spec["path"])) for spec in MODEL_TRANSLATIONS]
    lengths = {len(records) for records in translations}
    if len(lengths) != 1:
        raise ValueError(f"Translation files have mismatched lengths: {sorted(lengths)}")

    cases = []
    for index in range(lengths.pop()):
        source = translations[0][index]["SRC"]
        reference = translations[0][index]["REF"]
        hypotheses = {spec["column"]: records[index]["HYP"] for spec, records in zip(MODEL_TRANSLATIONS, translations)}
        f1_scores = {column: token_f1(hypothesis, reference) for column, hypothesis in hypotheses.items()}
        best_column = max(f1_scores, key=f1_scores.get)
        worst_column = min(f1_scores, key=f1_scores.get)
        rare_source_tokens = rare_tokens(source, source_freq)
        source_length = len(source.split())
        cases.append(
            {
                "test_index": index + 1,
                "source_length": source_length,
                "bucket": source_bucket(source_length),
                "source": source,
                "reference": reference,
                "hypotheses": hypotheses,
                "f1_scores": f1_scores,
                "f1_spread": f1_scores[best_column] - f1_scores[worst_column],
                "best_model_by_token_f1": best_column,
                "worst_model_by_token_f1": worst_column,
                "rare_source_tokens": rare_source_tokens,
            }
        )
    return cases


def select_cases(cases: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    def add(items: list[dict[str, Any]], limit: int, reason: str) -> None:
        for item in items:
            if len(selected) >= sample_count or limit <= 0:
                break
            if item["test_index"] in selected_indices:
                continue
            clone = dict(item)
            clone["selection_reason"] = reason
            selected.append(clone)
            selected_indices.add(item["test_index"])
            limit -= 1

    add(sorted(cases, key=lambda item: item["f1_spread"], reverse=True), sample_count // 2, "high model disagreement")
    add([item for item in sorted(cases, key=lambda item: item["source_length"], reverse=True) if item["source_length"] >= 21], 6, "long source sentence")
    add([item for item in sorted(cases, key=lambda item: len(item["rare_source_tokens"]), reverse=True) if item["rare_source_tokens"]], 4, "rare source tokens")
    add(sorted(cases, key=lambda item: max(item["f1_scores"].values()), reverse=True), 4, "strong/correct example")
    add(sorted(cases, key=lambda item: item["test_index"]), sample_count - len(selected), "coverage filler")
    return selected[:sample_count]


def flatten_case(case_id: int, case: dict[str, Any]) -> dict[str, Any]:
    hypotheses = case["hypotheses"]
    f1_scores = case["f1_scores"]
    tags = classify_case(
        case["source"],
        case["reference"],
        hypotheses,
        f1_scores,
        case["rare_source_tokens"],
    )
    row = {
        "case_id": case_id,
        "test_index": case["test_index"],
        "selection_reason": case["selection_reason"],
        "source_length": case["source_length"],
        "bucket": case["bucket"],
        "rare_source_tokens": " ".join(case["rare_source_tokens"]),
        "source": case["source"],
        "reference": case["reference"],
    }
    row.update(hypotheses)
    row.update({f"{column}_token_f1": score for column, score in f1_scores.items()})
    row["f1_spread"] = case["f1_spread"]
    row["best_model_by_token_f1"] = case["best_model_by_token_f1"]
    row["worst_model_by_token_f1"] = case["worst_model_by_token_f1"]
    row["error_tags"] = "; ".join(tags)
    row["notes"] = (
        f"Best by token F1: {case['best_model_by_token_f1']}; "
        f"worst: {case['worst_model_by_token_f1']}; "
        f"selected for {case['selection_reason']}."
    )
    return row


def write_case_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Error Analysis Case Samples\n\n")
        for row in rows:
            handle.write(
                f"## Case {row['case_id']} | test #{row['test_index']} | "
                f"{row['bucket']} | {row['error_tags']}\n\n"
            )
            handle.write(f"- Source: {row['source']}\n")
            handle.write(f"- Reference: {row['reference']}\n")
            handle.write(f"- LSTM no attention greedy: {row['lstm_no_attention_greedy']}\n")
            handle.write(f"- LSTM attention selected: {row['lstm_attention_selected']}\n")
            handle.write(f"- Gated attention selected: {row['gated_attention_selected']}\n")
            handle.write(f"- Small Transformer selected: {row['transformer_selected']}\n")
            handle.write(f"- Note: {row['notes']}\n\n")


def write_taxonomy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Error Taxonomy\n\n")
        handle.write("| Tag | Meaning | Report use |\n")
        handle.write("| --- | --- | --- |\n")
        handle.write("| rare word | Source contains low-frequency content words and models tend to omit or replace them. | Discuss vocabulary/data sparsity. |\n")
        handle.write("| action error | Main action is mistranslated, weakened, or replaced. | Discuss semantic faithfulness. |\n")
        handle.write("| modifier loss | Color, size, age, clothing, or other modifiers are dropped or altered. | Discuss detail retention. |\n")
        handle.write("| repetition | A hypothesis repeats words or local phrases. | Discuss decoding/model fluency failure. |\n")
        handle.write("| long sentence compression | Long source descriptions become shorter and lose clauses. | Discuss length sensitivity. |\n")
        handle.write("| fluent but incomplete | Output is grammatical-looking but misses key content. | Discuss BLEU/chrF and qualitative mismatch. |\n")
        handle.write("| correct/paraphrase | At least one model gives a strong faithful paraphrase. | Provide positive examples. |\n")


def plot_tag_counts(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(tag.strip() for tag in row["error_tags"].split(";"))
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, values, color="#4c78a8")
    axis.set_ylabel("Selected cases")
    axis.set_title("Error Tag Counts")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "error_tag_counts.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build representative qualitative error-analysis cases.")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--train-jsonl", default="data/processed/multi30k/train.jsonl")
    parser.add_argument("--sample-count", type=int, default=24)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = analysis_dir / "error_analysis"
    source_freq = load_source_frequencies(Path(args.train_jsonl))
    cases = build_case_pool(source_freq)
    selected = select_cases(cases, args.sample_count)
    rows = [flatten_case(index, case) for index, case in enumerate(selected, start=1)]

    write_csv(output_dir / "case_samples.csv", rows)
    write_json(output_dir / "case_samples.json", rows)
    write_case_markdown(output_dir / "case_samples.md", rows)
    write_taxonomy(output_dir / "error_taxonomy.md")
    plot_tag_counts(rows, analysis_dir / "figures")

    print(
        json.dumps(
            {
                "cases": len(rows),
                "source_frequency_tokens": len(source_freq),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
