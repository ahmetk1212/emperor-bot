from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def __init__(self, model, optimizer=None, scheduler=None, save_dir: str | Path = "./outputs/checkpoints", save_total_limit: int = 3):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit

    def save_checkpoint(self, step: int, metrics: Optional[Dict[str, float]] = None, best: bool = False) -> Path:
        name = "best_model.pt" if best else f"checkpoint_step_{step}.pt"
        path = self.save_dir / name
        payload: Dict[str, Any] = {"step": step, "model_state_dict": self.model.state_dict(), "metrics": metrics or {}}
        if self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, path)
        self._cleanup()
        return path

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

    def _cleanup(self):
        files = sorted(self.save_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        while len(files) > self.save_total_limit:
            old = files.pop(0)
            old.unlink(missing_ok=True)
