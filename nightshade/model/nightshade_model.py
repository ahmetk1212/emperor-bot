from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from ..core.config import ModelConfig
from .base import BaseModel
from .norms import RMSNorm
from .embeddings import CombinedEmbedding
from .blocks import MambaBlock, GatedAttentionBlock, FlashAttentionBlock, SwiGLUFeedForward


class NightShadeBlock(nn.Module):
    def __init__(self, layer_type: str, config: ModelConfig):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)

        if layer_type == "mamba":
            self.core = MambaBlock(config.hidden_size, config.dropout)
            self.ffn = None
        elif layer_type == "gated_attn":
            self.core = GatedAttentionBlock(config.hidden_size, max(1, config.num_heads // 2), config.attention_dropout)
            self.ffn = None
        else:
            self.core = FlashAttentionBlock(config.hidden_size, config.num_heads, config.head_dim, config.attention_dropout, config.use_bias)
            self.ffn = SwiGLUFeedForward(config.hidden_size, config.intermediate_size, config.dropout, config.use_bias)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = x + self.core(self.norm1(x), attention_mask)
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x


class NightShadeLM(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = CombinedEmbedding(config.vocab_size, config.hidden_size, config.max_seq_length, config.dropout)
        self.layers = nn.ModuleList([NightShadeBlock(layer_type, config) for layer_type in config.layer_types])
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0, top_k: int = 50, do_sample: bool = True) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out = self(input_ids)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def create_tiny_model() -> NightShadeLM:
    return NightShadeLM(ModelConfig())


def create_small_model() -> NightShadeLM:
    cfg = ModelConfig(vocab_size=40000, hidden_size=512, num_layers=8, intermediate_size=1360, max_seq_length=512, layer_types=["mamba", "mamba", "mamba", "gated_attn", "gated_attn", "flash_attn", "flash_attn", "flash_attn"])
    return NightShadeLM(cfg)


def create_medium_model() -> NightShadeLM:
    cfg = ModelConfig(vocab_size=50000, hidden_size=640, num_layers=10, intermediate_size=2048, max_seq_length=512, layer_types=["mamba"] * 4 + ["gated_attn"] * 2 + ["flash_attn"] * 4)
    return NightShadeLM(cfg)
