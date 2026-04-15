# emperor-bot / NightShade Tiny

This repository now splits the `llmtiny` draft into a runnable multi-file Python project:

- `nightshade/core`
- `nightshade/model`
- `nightshade/data`
- `nightshade/training`
- `nightshade/inference`
- `scripts`
- `configs`
- `tests`

## Quick start

```bash
python scripts/train.py --config configs/tiny.yaml --max_steps 10
python scripts/generate.py --config configs/tiny.yaml --model outputs/tiny/checkpoints/best_model.pt --tokenizer outputs/tiny/tokenizer/tokenizer.json --prompt "hello"
```

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
