from __future__ import annotations

import gzip
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from seq2seq_project.config import save_json
from seq2seq_project.vocab import Vocab

MULTI30K_BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
MULTI30K_FILES = {
    "train": {"en": "train.en.gz", "de": "train.de.gz"},
    "valid": {"en": "val.en.gz", "de": "val.de.gz"},
    "test": {"en": "test_2016_flickr.en.gz", "de": "test_2016_flickr.de.gz"},
}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass
class TranslationExample:
    src_tokens: list[str]
    tgt_tokens: list[str]
    src_ids: list[int]
    tgt_ids: list[int]


class TranslationDataset:
    def __init__(self, examples: list[TranslationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TranslationExample:
        return self.examples[index]


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text.strip())


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def decompress_gzip(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    with gzip.open(input_path, "rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(config["data"]["root"])
    return root, root / "raw" / "multi30k", root / "processed" / "multi30k"


def prepare_multi30k(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config["data"]
    _, raw_dir, processed_dir = _paths(config)
    src_lang = data_config["src_lang"]
    tgt_lang = data_config["tgt_lang"]
    lowercase = data_config["lowercase"]

    if processed_dir.exists() and force:
        shutil.rmtree(processed_dir)

    split_tokens: dict[str, list[tuple[list[str], list[str]]]] = {}
    split_counts: dict[str, int] = {}

    for split, files in MULTI30K_FILES.items():
        source_gz = raw_dir / files[src_lang]
        target_gz = raw_dir / files[tgt_lang]
        download_file(f"{MULTI30K_BASE_URL}/{files[src_lang]}", source_gz)
        download_file(f"{MULTI30K_BASE_URL}/{files[tgt_lang]}", target_gz)

        source_txt = raw_dir / source_gz.stem
        target_txt = raw_dir / target_gz.stem
        decompress_gzip(source_gz, source_txt)
        decompress_gzip(target_gz, target_txt)

        source_lines = read_lines(source_txt)
        target_lines = read_lines(target_txt)
        if len(source_lines) != len(target_lines):
            raise ValueError(f"Split {split} has mismatched source/target sizes.")

        pairs = [
            (tokenize(src, lowercase=lowercase), tokenize(tgt, lowercase=lowercase))
            for src, tgt in zip(source_lines, target_lines)
        ]
        split_tokens[split] = pairs
        split_counts[split] = len(pairs)

    source_vocab = Vocab.build(
        (src for src, _ in split_tokens["train"]),
        min_freq=data_config["min_freq"],
        max_size=data_config["max_vocab_size"],
    )
    target_vocab = Vocab.build(
        (tgt for _, tgt in split_tokens["train"]),
        min_freq=data_config["min_freq"],
        max_size=data_config["max_vocab_size"],
    )

    max_src_len = data_config["max_source_length"]
    max_tgt_len = data_config["max_target_length"]
    for split, pairs in split_tokens.items():
        records = []
        for src_tokens, tgt_tokens in pairs:
            if len(src_tokens) > max_src_len or len(tgt_tokens) > max_tgt_len:
                continue
            records.append(
                {
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "src_ids": source_vocab.encode(src_tokens, add_bos=False, add_eos=True),
                    "tgt_ids": target_vocab.encode(tgt_tokens, add_bos=True, add_eos=True),
                }
            )
        write_jsonl(processed_dir / f"{split}.jsonl", records)
        split_counts[f"{split}_kept"] = len(records)

    save_json(source_vocab.to_dict(), processed_dir / f"vocab.{src_lang}.json")
    save_json(target_vocab.to_dict(), processed_dir / f"vocab.{tgt_lang}.json")

    summary = {
        "processed_dir": str(processed_dir),
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        **split_counts,
    }
    save_json(summary, processed_dir / "summary.json")
    return summary


def load_vocabularies(config: dict[str, Any]) -> tuple[Vocab, Vocab]:
    _, _, processed_dir = _paths(config)
    src_lang = config["data"]["src_lang"]
    tgt_lang = config["data"]["tgt_lang"]
    with (processed_dir / f"vocab.{src_lang}.json").open("r", encoding="utf-8") as handle:
        src_vocab = Vocab.from_dict(json.load(handle))
    with (processed_dir / f"vocab.{tgt_lang}.json").open("r", encoding="utf-8") as handle:
        tgt_vocab = Vocab.from_dict(json.load(handle))
    return src_vocab, tgt_vocab


def load_split(config: dict[str, Any], split: str) -> TranslationDataset:
    _, _, processed_dir = _paths(config)
    path = processed_dir / f"{split}.jsonl"
    examples: list[TranslationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                TranslationExample(
                    src_tokens=record["src_tokens"],
                    tgt_tokens=record["tgt_tokens"],
                    src_ids=record["src_ids"],
                    tgt_ids=record["tgt_ids"],
                )
            )
    return TranslationDataset(examples)


def make_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def collate(examples: list[TranslationExample]) -> dict[str, Any]:
        src = [torch.tensor(example.src_ids, dtype=torch.long) for example in examples]
        tgt = [torch.tensor(example.tgt_ids, dtype=torch.long) for example in examples]
        src_lengths = torch.tensor([len(item) for item in src], dtype=torch.long)
        tgt_lengths = torch.tensor([len(item) for item in tgt], dtype=torch.long)
        return {
            "src": pad_sequence(src, batch_first=True, padding_value=src_pad_idx),
            "tgt": pad_sequence(tgt, batch_first=True, padding_value=tgt_pad_idx),
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "src_tokens": [example.src_tokens for example in examples],
            "tgt_tokens": [example.tgt_tokens for example in examples],
        }

    return collate
