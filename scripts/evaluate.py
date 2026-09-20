from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config
from seq2seq_project.train_loop import run_generation_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--output", default=None, help="Optional translation output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_generation_evaluation(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        limit_examples=args.limit_examples,
        method=args.method,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
