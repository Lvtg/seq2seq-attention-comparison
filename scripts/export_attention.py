from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2seq_project.config import load_config
from seq2seq_project.data import TranslationDataset, load_split, load_vocabularies, make_collate_fn
from seq2seq_project.train_loop import build_model, move_batch
from seq2seq_project.utils import resolve_device


def trim_at_eos(ids: list[int], eos_idx: int) -> list[int]:
    if eos_idx in ids:
        return ids[: ids.index(eos_idx)]
    return ids


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export attention examples from an attention checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--min-source-length", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    src_vocab, tgt_vocab = load_vocabularies(config)
    dataset = load_split(config, args.split)
    selected = [
        example
        for example in dataset.examples
        if len(example.src_tokens) >= args.min_source_length
    ][: args.num_examples]
    if not selected:
        raise ValueError("No examples matched the requested length filter.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(config, len(src_vocab), len(tgt_vocab), src_vocab.pad_idx, tgt_vocab.pad_idx).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        TranslationDataset(selected),
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(src_vocab.pad_idx, tgt_vocab.pad_idx),
    )

    records = []
    for batch in loader:
        batch = move_batch(batch, device)
        output_ids, aux = model.greedy_decode(
            batch["src"],
            batch["src_lengths"],
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            max_length=config["decoding"]["max_length"],
        )
        if "attention" not in aux:
            raise ValueError("This checkpoint did not return attention weights.")

        source_length = int(batch["src_lengths"][0].item())
        source_tokens_for_attention = batch["src_tokens"][0] + [src_vocab.idx_to_token[src_vocab.eos_idx]]
        generated_ids = trim_at_eos(output_ids[0].cpu().tolist(), tgt_vocab.eos_idx)
        hypothesis_tokens = tgt_vocab.decode(generated_ids)
        attention = aux["attention"][0, : len(generated_ids), :source_length].cpu().tolist()
        record = {
            "source": " ".join(batch["src_tokens"][0]),
            "reference": " ".join(batch["tgt_tokens"][0]),
            "hypothesis": " ".join(hypothesis_tokens),
            "source_length": len(batch["src_tokens"][0]),
            "source_tokens": source_tokens_for_attention,
            "reference_tokens": batch["tgt_tokens"][0],
            "hypothesis_tokens": hypothesis_tokens,
            "attention": attention,
        }
        if "gate" in aux:
            gate_sequence = aux["gate"][0, : len(generated_ids)].cpu().tolist()
            record["gate_sequence"] = gate_sequence
            record["gate_mean"] = sum(gate_sequence) / max(len(gate_sequence), 1)
            record["gate_min"] = min(gate_sequence) if gate_sequence else None
            record["gate_max"] = max(gate_sequence) if gate_sequence else None
        records.append(record)

    if args.output is None:
        artifact_type = "gates" if "gated" in config["model"]["attention"] else "attention"
        suffix = "gate_examples" if artifact_type == "gates" else "attention_examples"
        output_path = (
            Path(config["paths"]["artifacts_dir"])
            / artifact_type
            / f"{config['training']['experiment_name']}_{args.split}_{suffix}.json"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(output_path), "examples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
