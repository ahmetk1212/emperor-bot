import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    """Lightweight SSM-like block for tiny smoke training."""

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *_):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.proj(y))
        return y
