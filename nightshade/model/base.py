from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        ...

    def get_num_params(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
