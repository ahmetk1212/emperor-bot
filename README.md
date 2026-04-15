# emperor-bot / NightShade Tiny

Bu repo artık `llmtiny` taslağını çalışan çok dosyalı bir Python proje yapısına böler:

- `nightshade/core`
- `nightshade/model`
- `nightshade/data`
- `nightshade/training`
- `nightshade/inference`
- `scripts`
- `configs`
- `tests`

## Hızlı kullanım

```bash
python scripts/train.py --config configs/tiny.yaml --max_steps 10
python scripts/generate.py --config configs/tiny.yaml --model outputs/tiny/checkpoints/best_model.pt --tokenizer outputs/tiny/tokenizer/tokenizer.json --prompt "merhaba"
```

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
