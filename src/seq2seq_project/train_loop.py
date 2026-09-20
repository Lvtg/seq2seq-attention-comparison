from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from seq2seq_project.config import save_json
from seq2seq_project.data import load_split, load_vocabularies, make_collate_fn
from seq2seq_project.models import LSTMSeq2Seq, TransformerSeq2Seq
from seq2seq_project.utils import Timer, append_jsonl, count_parameters, resolve_device, set_seed


def build_model(config: dict[str, Any], src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int):
    model_config = config["model"]
    if model_config["type"] == "lstm":
        return LSTMSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=src_pad_idx,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            attention=model_config["attention"],
            length_feature_norm=model_config.get("length_feature_norm", config["data"]["max_source_length"]),
        )
    if model_config["type"] == "transformer":
        return TransformerSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
        )
    raise ValueError(f"Unsupported model type: {model_config['type']}")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ["src", "tgt", "src_lengths", "tgt_lengths"]:
        moved[key] = batch[key].to(device)
    return moved


def make_loaders(config: dict[str, Any], src_pad_idx: int, tgt_pad_idx: int):
    training_config = config["training"]
    collate_fn = make_collate_fn(src_pad_idx, tgt_pad_idx)
    train_loader = DataLoader(
        load_split(config, "train"),
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        load_split(config, "valid"),
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    config: dict[str, Any],
    limit_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    teacher_forcing_ratio = config["training"]["teacher_forcing_ratio"]
    clip_grad_norm = config["training"]["clip_grad_norm"]

    iterator = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch["src"], batch["src_lengths"], batch["tgt"], teacher_forcing_ratio)
        target = batch["tgt"][:, 1 : 1 + logits.size(1)]
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        token_count = target.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, limit_batches: int | None = None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, desc="valid", leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break
        batch = move_batch(batch, device)
        logits, _ = model(batch["src"], batch["src_lengths"], batch["tgt"], teacher_forcing_ratio=1.0)
        target = batch["tgt"][:, 1 : 1 + logits.size(1)]
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        token_count = target.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def run_training(
    config: dict[str, Any],
    limit_train_batches: int | None = None,
    limit_valid_batches: int | None = None,
) -> None:
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])
    src_vocab, tgt_vocab = load_vocabularies(config)
    train_loader, valid_loader = make_loaders(config, src_vocab.pad_idx, tgt_vocab.pad_idx)

    model = build_model(config, len(src_vocab), len(tgt_vocab), src_vocab.pad_idx, tgt_vocab.pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    experiment_name = config["training"]["experiment_name"]
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"]) / experiment_name
    run_dir = Path(config["paths"]["runs_dir"]) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, run_dir / "resolved_config.json")

    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    best_valid = float("inf")
    patience = config["training"]["early_stopping_patience"]
    stale_epochs = 0

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        timer = Timer()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config,
            limit_batches=limit_train_batches,
        )
        valid_loss = evaluate_loss(
            model,
            valid_loader,
            criterion,
            device,
            limit_batches=limit_valid_batches,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "elapsed_seconds": timer.elapsed(),
        }
        append_jsonl(run_dir / "metrics.jsonl", record)
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            stale_epochs = 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break


@torch.no_grad()
def run_generation_evaluation(
    config: dict[str, Any],
    checkpoint_path: str,
    split: str = "test",
    limit_examples: int | None = None,
    method: str | None = None,
    beam_size: int | None = None,
    length_penalty: float | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    import sacrebleu

    decoding_config = config["decoding"]
    method = method or decoding_config.get("method", "greedy")
    beam_size = beam_size if beam_size is not None else decoding_config.get("beam_size", 5)
    length_penalty = length_penalty if length_penalty is not None else decoding_config.get("length_penalty", 0.6)
    if method not in {"greedy", "beam"}:
        raise ValueError(f"Unsupported decoding method: {method}")

    device = resolve_device(config["project"]["device"])
    src_vocab, tgt_vocab = load_vocabularies(config)
    dataset = load_split(config, split)
    collate_fn = make_collate_fn(src_vocab.pad_idx, tgt_vocab.pad_idx)
    batch_size = 1 if method == "beam" else config["training"]["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, len(src_vocab), len(tgt_vocab), src_vocab.pad_idx, tgt_vocab.pad_idx).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    max_length = config["decoding"]["max_length"]

    seen = 0
    for batch in tqdm(loader, desc=f"generate-{split}", disable=not show_progress):
        batch = move_batch(batch, device)
        if method == "greedy":
            output_ids, _ = model.greedy_decode(
                batch["src"],
                batch["src_lengths"],
                bos_idx=tgt_vocab.bos_idx,
                eos_idx=tgt_vocab.eos_idx,
                max_length=max_length,
            )
        else:
            output_ids, _ = model.beam_decode(
                batch["src"],
                batch["src_lengths"],
                bos_idx=tgt_vocab.bos_idx,
                eos_idx=tgt_vocab.eos_idx,
                max_length=max_length,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        for row, src_tokens, tgt_tokens in zip(output_ids.cpu().tolist(), batch["src_tokens"], batch["tgt_tokens"]):
            predictions.append(" ".join(tgt_vocab.decode(row)))
            references.append(" ".join(tgt_tokens))
            sources.append(" ".join(src_tokens))
            seen += 1
            if limit_examples is not None and seen >= limit_examples:
                break
        if limit_examples is not None and seen >= limit_examples:
            break

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    experiment_name = config["training"]["experiment_name"]
    if output_path is None:
        output_dir = Path(config["paths"]["artifacts_dir"]) / "translations"
        output_dir.mkdir(parents=True, exist_ok=True)
        if method == "greedy":
            suffix = "greedy"
        else:
            penalty_label = str(length_penalty).replace(".", "p")
            suffix = f"beam{beam_size}_lp{penalty_label}"
        output_path = output_dir / f"{experiment_name}_{split}_{suffix}.txt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for source, prediction, reference in zip(sources, predictions, references):
            handle.write(f"SRC\t{source}\n")
            handle.write(f"HYP\t{prediction}\n")
            handle.write(f"REF\t{reference}\n\n")
    result = {
        "split": split,
        "method": method,
        "beam_size": beam_size if method == "beam" else None,
        "length_penalty": length_penalty if method == "beam" else None,
        "bleu": bleu,
        "chrf": chrf,
        "examples": len(predictions),
        "output": str(output_path),
    }
    print(json.dumps(result, indent=2))
    return result
