from __future__ import annotations

from typing import List, Union

import torch


class TextGenerator:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 32, do_sample: bool = True, temperature: float = 0.8, top_k: int = 50) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        out = []
        self.model.eval()
        for p in prompts:
            x = self.tokenizer.encode(p, return_tensors="pt")["input_ids"].to(self.device)
            y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, do_sample=do_sample)
            out.append(self.tokenizer.decode(y[0]))
        return out
