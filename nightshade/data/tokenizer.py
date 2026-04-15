from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Union

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class BPETokenizer:
    """Simple whitespace tokenizer with BPE-like API for smoke training."""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        self.token_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["</s>"]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def vocab_size_property(self) -> int:
        return len(self.token_to_id)

    def train(self, files: List[str]) -> None:
        counter: Counter[str] = Counter()
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    counter.update(line.strip().split())
        room = max(0, self.vocab_size - len(self.special_tokens))
        for token, _ in counter.most_common(room):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None, return_tensors: str | None = None):
        ids = [self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in text.strip().split() if tok]
        if add_special_tokens:
            ids = [self.token_to_id["<s>"]] + ids + [self.eos_token_id]
        if max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            if torch is None:
                raise RuntimeError("torch required for return_tensors='pt'")
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids

    def decode(self, ids: List[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        toks = []
        for i in ids:
            t = self.id_to_token.get(int(i), "<unk>")
            if skip_special_tokens and t in self.special_tokens:
                continue
            toks.append(t)
        return " ".join(toks).strip()

    def save(self, save_dir: Union[str, Path]) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "token_to_id": self.token_to_id}, f)

    @classmethod
    def load(cls, path_or_dir: Union[str, Path]) -> "BPETokenizer":
        p = Path(path_or_dir)
        if p.is_dir():
            p = p / "tokenizer.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("vocab_size", 30000))
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok
