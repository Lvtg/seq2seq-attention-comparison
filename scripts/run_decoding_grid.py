from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config
from seq2seq_project.train_loop import run_generation_evaluation


MODELS = [
    {
        "model_id": "lstm_attention",
        "label": "LSTM attention",
        "config": "configs/lstm_attention.json",
        "checkpoint": "checkpoints/lstm_attention/best.pt",
    },
    {
        "model_id": "gated_attention",
        "label": "Gated attention",
        "config": "configs/gated_attention.json",
        "checkpoint": "checkpoints/gated_attention/best.pt",
    },
    {
        "model_id": "transformer_small",
        "label": "Small Transformer",
        "config": "configs/transformer_small.json",
        "checkpoint": "checkpoints/transformer_small/best.pt",
    },
]
BEAM_SIZES = [3, 5]
LENGTH_PENALTIES = [0.0, 0.6, 1.0]
BLEU_TIE_TOLERANCE = 0.1


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


def decode_label(method: str, beam_size: int | None, length_penalty: float | None) -> str:
    if method == "greedy":
        return "greedy"
    penalty = str(length_penalty).replace(".", "p")
    return f"beam{beam_size}_lp{penalty}"


def decode_settings() -> list[dict[str, Any]]:
    settings = [{"method": "greedy", "beam_size": None, "length_penalty": None}]
    for beam_size in BEAM_SIZES:
        for length_penalty in LENGTH_PENALTIES:
            settings.append(
                {
                    "method": "beam",
                    "beam_size": beam_size,
                    "length_penalty": length_penalty,
                }
            )
    return settings


def row_from_result(model: dict[str, str], setting: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    label = decode_label(setting["method"], setting["beam_size"], setting["length_penalty"])
    return {
        "model_id": model["model_id"],
        "label": model["label"],
        "split": result["split"],
        "decode_label": label,
        "method": setting["method"],
        "beam_size": setting["beam_size"],
        "length_penalty": setting["length_penalty"],
        "bleu": result["bleu"],
        "chrf": result["chrf"],
        "examples": result["examples"],
        "translation_output": result["output"],
    }


def is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    if candidate["bleu"] > current["bleu"] + BLEU_TIE_TOLERANCE:
        return True
    if abs(candidate["bleu"] - current["bleu"]) <= BLEU_TIE_TOLERANCE and candidate["chrf"] > current["chrf"]:
        return True
    return False


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = rows[0]
    for row in rows[1:]:
        if is_better(row, best):
            best = row
    return best


def evaluate_one(
    model: dict[str, str],
    setting: dict[str, Any],
    split: str,
    output_dir: Path,
    limit_examples: int | None,
) -> dict[str, Any]:
    config = load_config(model["config"])
    label = decode_label(setting["method"], setting["beam_size"], setting["length_penalty"])
    output_path = output_dir / "translations" / split / f"{model['model_id']}_{split}_{label}.txt"
    result = run_generation_evaluation(
        config,
        checkpoint_path=model["checkpoint"],
        split=split,
        limit_examples=limit_examples,
        method=setting["method"],
        beam_size=setting["beam_size"],
        length_penalty=setting["length_penalty"],
        output_path=output_path,
        show_progress=False,
    )
    return row_from_result(model, setting, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation decoding grid and selected test decoding.")
    parser.add_argument("--output-dir", default="analysis/decoding")
    parser.add_argument("--limit-examples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    valid_rows = []
    selected_rows = []
    settings = decode_settings()

    for model in MODELS:
        model_valid_rows = []
        for setting in settings:
            row = evaluate_one(model, setting, "valid", output_dir, args.limit_examples)
            valid_rows.append(row)
            model_valid_rows.append(row)
        selected = select_best(model_valid_rows)
        selected_setting = {
            "method": selected["method"],
            "beam_size": selected["beam_size"],
            "length_penalty": selected["length_penalty"],
        }
        selected_rows.append(evaluate_one(model, selected_setting, "test", output_dir, args.limit_examples))

    write_csv(output_dir / "decoding_valid_grid.csv", valid_rows)
    write_json(output_dir / "decoding_valid_grid.json", valid_rows)
    write_csv(output_dir / "decoding_test_selected.csv", selected_rows)
    write_json(output_dir / "decoding_test_selected.json", selected_rows)
    write_json(
        output_dir / "selection_policy.json",
        {
            "valid_selection": "Select highest valid BLEU; if BLEU differs by at most 0.1, select higher chrF.",
            "bleu_tie_tolerance": BLEU_TIE_TOLERANCE,
            "beam_sizes": BEAM_SIZES,
            "length_penalties": LENGTH_PENALTIES,
            "limit_examples": args.limit_examples,
        },
    )
    print(
        json.dumps(
            {
                "valid_rows": len(valid_rows),
                "selected_test_rows": len(selected_rows),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
