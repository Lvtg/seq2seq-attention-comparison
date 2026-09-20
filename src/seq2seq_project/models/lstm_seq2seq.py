from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        attention: str = "none",
        length_feature_norm: float = 80.0,
    ) -> None:
        super().__init__()
        if attention not in {"none", "luong", "gated_luong"}:
            raise ValueError(f"Unsupported attention type: {attention}")
        self.attention = attention
        self.src_pad_idx = src_pad_idx
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.length_feature_norm = length_feature_norm

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        decoder_input_dim = embedding_dim if attention == "none" else embedding_dim + hidden_dim
        self.decoder = nn.LSTM(
            decoder_input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        if attention == "gated_luong":
            self.gate = nn.Linear(hidden_dim * 3 + 1, hidden_dim)
        self.output = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor):
        embedded = self.dropout(self.src_embedding(src))
        lengths_cpu = src_lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (hidden, cell) = self.encoder(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )
        return outputs, hidden, cell

    def attend(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(src.eq(self.src_pad_idx), -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

    def fuse_context(
        self,
        decoder_hidden: torch.Tensor,
        attention_context: torch.Tensor,
        global_context: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.attention != "gated_luong":
            return attention_context, None
        length_feature = (src_lengths.float() / self.length_feature_norm).clamp(max=1.0).unsqueeze(1)
        gate_input = torch.cat([decoder_hidden, attention_context, global_context, length_feature], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * attention_context + (1.0 - gate) * global_context
        return fused, gate.mean(dim=1)

    def decode_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        global_context: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        embedded = self.dropout(self.tgt_embedding(input_token)).unsqueeze(1)
        attention_weights = None
        gate_value = None
        if self.attention == "none":
            decoder_input = embedded
        else:
            decoder_hidden = hidden[-1]
            attention_context, attention_weights = self.attend(decoder_hidden, encoder_outputs, src)
            context, gate_value = self.fuse_context(
                decoder_hidden,
                attention_context,
                global_context,
                src_lengths,
            )
            decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        logits = self.output(output.squeeze(1))
        return logits, hidden, cell, attention_weights, gate_value

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        encoder_outputs, hidden, cell = self.encode(src, src_lengths)
        global_context = hidden[-1]
        input_token = tgt[:, 0]
        logits_steps = []
        attention_steps = []
        gate_steps = []

        for step in range(1, tgt.size(1)):
            logits, hidden, cell, attention_weights, gate_value = self.decode_step(
                input_token,
                hidden,
                cell,
                global_context,
                encoder_outputs,
                src,
                src_lengths,
            )
            logits_steps.append(logits)
            if attention_weights is not None:
                attention_steps.append(attention_weights)
            if gate_value is not None:
                gate_steps.append(gate_value)

            use_teacher = torch.rand(1, device=tgt.device).item() < teacher_forcing_ratio
            input_token = tgt[:, step] if use_teacher else logits.argmax(dim=1)

        aux: dict[str, Any] = {}
        if attention_steps:
            aux["attention"] = torch.stack(attention_steps, dim=1)
        if gate_steps:
            aux["gate"] = torch.stack(gate_steps, dim=1)
        return torch.stack(logits_steps, dim=1), aux

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        encoder_outputs, hidden, cell = self.encode(src, src_lengths)
        global_context = hidden[-1]
        input_token = torch.full((src.size(0),), bos_idx, dtype=torch.long, device=src.device)
        outputs = []
        attention_steps = []
        gate_steps = []

        for _ in range(max_length):
            logits, hidden, cell, attention_weights, gate_value = self.decode_step(
                input_token,
                hidden,
                cell,
                global_context,
                encoder_outputs,
                src,
                src_lengths,
            )
            input_token = logits.argmax(dim=1)
            outputs.append(input_token)
            if attention_weights is not None:
                attention_steps.append(attention_weights)
            if gate_value is not None:
                gate_steps.append(gate_value)
            if torch.all(input_token.eq(eos_idx)):
                break

        aux: dict[str, Any] = {}
        if attention_steps:
            aux["attention"] = torch.stack(attention_steps, dim=1)
        if gate_steps:
            aux["gate"] = torch.stack(gate_steps, dim=1)
        return torch.stack(outputs, dim=1), aux

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
        if src.size(0) != 1:
            raise ValueError("beam_decode currently expects batch size 1.")

        encoder_outputs, hidden, cell = self.encode(src, src_lengths)
        global_context = hidden[-1]
        beams = [
            {
                "tokens": [bos_idx],
                "log_prob": 0.0,
                "hidden": hidden,
                "cell": cell,
                "done": False,
            }
        ]

        for _ in range(max_length):
            candidates = []
            for beam in beams:
                if beam["done"]:
                    candidates.append(beam)
                    continue
                input_token = torch.tensor([beam["tokens"][-1]], dtype=torch.long, device=src.device)
                logits, next_hidden, next_cell, _, _ = self.decode_step(
                    input_token,
                    beam["hidden"],
                    beam["cell"],
                    global_context,
                    encoder_outputs,
                    src,
                    src_lengths,
                )
                log_probs = torch.log_softmax(logits, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)
                for token_log_prob, token_idx in zip(top_log_probs[0], top_indices[0]):
                    token = int(token_idx.item())
                    tokens = [*beam["tokens"], token]
                    candidates.append(
                        {
                            "tokens": tokens,
                            "log_prob": beam["log_prob"] + float(token_log_prob.item()),
                            "hidden": next_hidden.clone(),
                            "cell": next_cell.clone(),
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
