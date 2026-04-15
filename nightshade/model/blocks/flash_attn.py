import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.o_proj(y)
