import torch
import torch.nn as nn


class GatedAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

    def forward(self, x: torch.Tensor, attention_mask=None):
        y, _ = self.attn(x, x, x, need_weights=False)
        return y * self.gate(x)
