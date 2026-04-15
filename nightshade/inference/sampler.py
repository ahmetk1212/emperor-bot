import torch


class GreedySampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class TopKSampler:
    def __init__(self, k: int = 50):
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.k, logits.size(-1))
        vals, idx = torch.topk(logits, k)
        probs = torch.softmax(vals, dim=-1)
        picked = torch.multinomial(probs, 1)
        return idx.gather(-1, picked)


class NucleusSampler:
    def __init__(self, p: float = 0.95):
        self.p = p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        mask = csum <= self.p
        mask[..., 0] = True
        keep = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        keep = keep / keep.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(keep, 1)
        return sorted_idx.gather(-1, sampled)
