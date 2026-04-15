from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 256, min_length: int = 2):
        self.examples = []
        for t in texts:
            ids = tokenizer.encode(t, max_length=max_length)
            if len(ids) >= min_length:
                self.examples.append(ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.examples[idx]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


class ConcatDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 256, eos_token_id: int = 3, shuffle_documents: bool = False):
        docs = [tokenizer.encode(t, add_special_tokens=False) + [eos_token_id] for t in texts]
        if shuffle_documents:
            import random

            random.shuffle(docs)
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        all_tokens = []
        for d in docs:
            all_tokens.extend(d)
        self.all_tokens = all_tokens
        self.num_seq = max(1, (len(all_tokens) + max_length - 1) // max_length)

    def __len__(self) -> int:
        return self.num_seq

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.max_length
        x = self.all_tokens[start : start + self.max_length]
        y = self.all_tokens[start + 1 : start + self.max_length + 1]
        if len(x) < self.max_length:
            x = x + [self.pad_id] * (self.max_length - len(x))
        if len(y) < self.max_length:
            y = y + [-100] * (self.max_length - len(y))
        return {"input_ids": torch.tensor(x, dtype=torch.long), "labels": torch.tensor(y, dtype=torch.long)}


def load_text_file(file_path: str) -> List[str]:
    p = Path(file_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
