from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.embedding_dim = embedding_dim
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=tgt_pad_idx)
        self.positional = PositionalEncoding(embedding_dim, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(embedding_dim, tgt_vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del src_lengths, teacher_forcing_ratio
        decoder_input = tgt[:, :-1]
        src_key_padding_mask = src.eq(self.src_pad_idx)
        tgt_key_padding_mask = decoder_input.eq(self.tgt_pad_idx)
        tgt_mask = self._causal_mask(decoder_input.size(1), decoder_input.device)

        src_emb = self.positional(self.src_embedding(src) * math.sqrt(self.embedding_dim))
        tgt_emb = self.positional(self.tgt_embedding(decoder_input) * math.sqrt(self.embedding_dim))
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output(output), {}

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del src_lengths
        src_key_padding_mask = src.eq(self.src_pad_idx)
        src_emb = self.positional(self.src_embedding(src) * math.sqrt(self.embedding_dim))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        generated = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_length):
            tgt_mask = self._causal_mask(generated.size(1), generated.device)
            tgt_emb = self.positional(self.tgt_embedding(generated) * math.sqrt(self.embedding_dim))
            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = self.output(output[:, -1]).argmax(dim=1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(eos_idx)):
                break
        return generated[:, 1:], {}

    @staticmethod
    def _length_penalized_score(log_prob: float, output_length: int, length_penalty: float) -> float:
        if length_penalty == 0.0:
            return log_prob
        norm = ((5.0 + max(output_length, 1)) / 6.0) ** length_penalty
        return log_prob / norm

    @torch.no_grad()
    def beam_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
        beam_size: int = 5,
        length_penalty: float = 0.6,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del src_lengths
        if src.size(0) != 1:
            raise ValueError("beam_decode currently expects batch size 1.")

        src_key_padding_mask = src.eq(self.src_pad_idx)
        src_emb = self.positional(self.src_embedding(src) * math.sqrt(self.embedding_dim))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        beams = [{"tokens": [bos_idx], "log_prob": 0.0, "done": False}]

        for _ in range(max_length):
            candidates = []
            for beam in beams:
                if beam["done"]:
                    candidates.append(beam)
                    continue
                generated = torch.tensor([beam["tokens"]], dtype=torch.long, device=src.device)
                tgt_mask = self._causal_mask(generated.size(1), generated.device)
                tgt_emb = self.positional(self.tgt_embedding(generated) * math.sqrt(self.embedding_dim))
                output = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                logits = self.output(output[:, -1])
                log_probs = torch.log_softmax(logits, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)
                for token_log_prob, token_idx in zip(top_log_probs[0], top_indices[0]):
                    token = int(token_idx.item())
                    candidates.append(
                        {
                            "tokens": [*beam["tokens"], token],
                            "log_prob": beam["log_prob"] + float(token_log_prob.item()),
                            "done": token == eos_idx,
                        }
                    )

            beams = sorted(
                candidates,
                key=lambda item: self._length_penalized_score(
                    item["log_prob"],
                    len(item["tokens"]) - 1,
                    length_penalty,
                ),
                reverse=True,
            )[:beam_size]
            if all(beam["done"] for beam in beams):
                break

        best = max(
            beams,
            key=lambda item: self._length_penalized_score(
                item["log_prob"],
                len(item["tokens"]) - 1,
                length_penalty,
            ),
        )
        output_tokens = best["tokens"][1:] or [eos_idx]
        return torch.tensor([output_tokens], dtype=torch.long, device=src.device), {
            "sequence_log_prob": best["log_prob"],
            "length_penalized_score": self._length_penalized_score(
                best["log_prob"],
                len(output_tokens),
                length_penalty,
            ),
        }
