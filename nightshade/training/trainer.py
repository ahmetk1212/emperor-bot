from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..core.config import NightShadeConfig
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .checkpoint import CheckpointManager


class LMTrainer:
    def __init__(self, model, config: NightShadeConfig, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, tokenizer=None):
        self.model = model
        self.config = config
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = create_optimizer(
            model,
            optimizer_name=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta1=config.training.beta1,
            beta2=config.training.beta2,
            eps=config.training.eps,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_name=config.training.scheduler,
            num_training_steps=config.training.max_steps,
            warmup_steps=config.training.warmup_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )
        self.ckpt = CheckpointManager(
            model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_dir=Path(config.output_dir) / "checkpoints",
            save_total_limit=3,
        )

    def train(self, max_steps: Optional[int] = None):
        max_steps = max_steps or self.config.training.max_steps
        self.model.train()
        step = 0
        self.optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            for batch in self.train_loader:
                batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels", batch["input_ids"]))
                loss = out["loss"]
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                step += 1
                if step % self.config.training.save_steps == 0:
                    self.ckpt.save_checkpoint(step=step, metrics={"loss": float(loss.item())})
                if step >= max_steps:
                    break
            if len(self.train_loader) == 0:
                break
        self.ckpt.save_checkpoint(step=step, metrics={"loss": float(loss.item()) if 'loss' in locals() else 0.0}, best=True)
        return {"global_step": step, "loss": float(loss.item()) if 'loss' in locals() else 0.0}
