#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nightshade.core.config import NightShadeConfig
from nightshade.model import NightShadeLM
from nightshade.data.tokenizer import BPETokenizer
from nightshade.inference import TextGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="merhaba")
    p.add_argument("--max_tokens", type=int, default=32)
    args = p.parse_args()

    cfg = NightShadeConfig.from_yaml(args.config)
    model = NightShadeLM(cfg.model)
    ckpt = torch.load(args.model, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    tok = BPETokenizer.load(args.tokenizer)
    gen = TextGenerator(model, tok)
    print(gen.generate(args.prompt, max_new_tokens=args.max_tokens, do_sample=False)[0])


if __name__ == "__main__":
    main()
