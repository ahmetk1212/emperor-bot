import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    params = [p for p in model.parameters() if p.requires_grad]
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=beta1, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
