from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


@dataclass
class Vocab:
    idx_to_token: list[str]
    token_to_idx: dict[str, int]

    @classmethod
    def build(
        cls,
        tokenized_lines: Iterable[list[str]],
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in tokenized_lines:
            counter.update(tokens)

        tokens = list(SPECIALS)
        limit = None if max_size is None else max(max_size - len(SPECIALS), 0)
        for token, freq in counter.most_common(limit):
            if freq < min_freq:
                continue
            if token not in SPECIALS:
                tokens.append(token)
        return cls(tokens, {token: idx for idx, token in enumerate(tokens)})

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[PAD]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[UNK]

    @property
    def bos_idx(self) -> int:
        return self.token_to_idx[BOS]

    @property
    def eos_idx(self) -> int:
        return self.token_to_idx[EOS]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, tokens: list[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        ids.extend(self.token_to_idx.get(token, self.unk_idx) for token in tokens)
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.idx_to_token[int(idx)]
            if token == EOS:
                break
            if skip_specials and token in SPECIALS:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, list[str]]:
        return {"idx_to_token": self.idx_to_token}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "Vocab":
        tokens = data["idx_to_token"]
        return cls(tokens, {token: idx for idx, token in enumerate(tokens)})
