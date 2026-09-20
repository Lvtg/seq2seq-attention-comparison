from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config
from seq2seq_project.train_loop import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seq2seq model.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-valid-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs

    run_training(
        config,
        limit_train_batches=args.limit_train_batches,
        limit_valid_batches=args.limit_valid_batches,
    )


if __name__ == "__main__":
    main()
