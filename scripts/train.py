#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nightshade.core.config import NightShadeConfig
from nightshade.model import NightShadeLM
from nightshade.data.tokenizer import BPETokenizer
from nightshade.data.dataset import ConcatDataset, load_text_file
from nightshade.data.collators import PreTrainingCollator
from nightshade.training import LMTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args()


def _sample_texts() -> list[str]:
    return [
        "hello world this is a tiny training sample",
        "the pipeline should be validated at tiny scale first",
        "short smoke training proves the end to end flow",
    ] * 200


def main():
    args = parse_args()
    cfg = NightShadeConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps

    train_path = Path(cfg.data.train_path)
    texts = load_text_file(str(train_path)) if train_path.exists() else _sample_texts()

    tok_path = Path(cfg.data.tokenizer_path)
    tok = BPETokenizer(vocab_size=cfg.data.tokenizer_vocab_size)
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
    else:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
            temp_file = f.name
        tok.train([temp_file])
        Path(temp_file).unlink(missing_ok=True)
        tok.save(tok_path.parent)

    ds = ConcatDataset(texts=texts, tokenizer=tok, max_length=cfg.data.tokenizer_max_length, eos_token_id=tok.eos_token_id, shuffle_documents=True)
    collator = PreTrainingCollator(tokenizer=tok, max_length=cfg.data.tokenizer_max_length)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collator, num_workers=cfg.data.num_workers)

    model = NightShadeLM(cfg.model)
    trainer = LMTrainer(model=model, config=cfg, train_dataloader=loader, tokenizer=tok)
    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
