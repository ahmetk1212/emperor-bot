import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, num_training_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.total = max(1, num_training_steps)
        self.warmup = max(0, warmup_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_idx = 0

    def step(self):
        self.step_idx += 1
        if self.step_idx <= self.warmup:
            ratio = self.step_idx / max(1, self.warmup)
        else:
            progress = (self.step_idx - self.warmup) / max(1, self.total - self.warmup)
            ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.base_lrs[i] * ratio

    def state_dict(self):
        return {"step_idx": self.step_idx}

    def load_state_dict(self, state):
        self.step_idx = state.get("step_idx", 0)


def create_scheduler(optimizer, scheduler_name: str = "warmup_cosine", num_training_steps: int = 100, warmup_steps: int = 0, min_lr_ratio: float = 0.1, **_):
    if scheduler_name == "constant":
        return WarmupCosineScheduler(optimizer, num_training_steps=1, warmup_steps=0, min_lr_ratio=1.0)
    return WarmupCosineScheduler(optimizer, num_training_steps=num_training_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio)
