from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a resolved experiment config.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
