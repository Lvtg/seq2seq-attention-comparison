from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config
from seq2seq_project.data import prepare_multi30k


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Multi30k En-De.")
    parser.add_argument("--config", default="configs/base.json", help="Path to config JSON.")
    parser.add_argument("--force", action="store_true", help="Recreate processed files.")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = prepare_multi30k(config, force=args.force)

    print("Prepared Multi30k:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
