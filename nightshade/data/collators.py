from __future__ import annotations

from typing import Any, Dict, List

import torch


class PreTrainingCollator:
    def __init__(self, tokenizer: Any, max_length: int = 256, label_pad_token_id: int = -100):
        self.pad_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids = [e["input_ids"].tolist() if hasattr(e["input_ids"], "tolist") else e["input_ids"] for e in examples]
        max_len = min(self.max_length, max(len(x) for x in ids))
        batch_in, batch_mask, batch_lbl = [], [], []
        for x in ids:
            x = x[:max_len]
            pad = max_len - len(x)
            inp = x + [self.pad_id] * pad
            lbl = x + [self.label_pad_token_id] * pad
            msk = [1] * len(x) + [0] * pad
            batch_in.append(inp)
            batch_mask.append(msk)
            batch_lbl.append(lbl)
        return {
            "input_ids": torch.tensor(batch_in, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_lbl, dtype=torch.long),
        }
