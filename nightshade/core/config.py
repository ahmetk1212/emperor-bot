from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ModelConfig:
    vocab_size: int = 30000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 1024
    max_seq_length: int = 256
    layer_types: list[str] = field(default_factory=lambda: ["mamba", "mamba", "gated_attn", "flash_attn", "flash_attn", "flash_attn"])
    norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    init_std: float = 0.02
    use_bias: bool = False
    use_swiglu: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.head_dim != 0:
            raise ValueError("hidden_size head_dim ile bölünebilir olmalı")
        self.num_heads = self.hidden_size // self.head_dim
        if len(self.layer_types) != self.num_layers:
            self.layer_types = ["mamba"] * max(1, self.num_layers // 3) + ["gated_attn"] * max(1, self.num_layers // 6)
            self.layer_types = (self.layer_types + ["flash_attn"] * self.num_layers)[: self.num_layers]


@dataclass
class DataConfig:
    train_path: str = "./data/train.txt"
    val_path: str = "./data/val.txt"
    tokenizer_path: str = "./data/tokenizer.json"
    tokenizer_vocab_size: int = 30000
    tokenizer_max_length: int = 256
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 50
    eval_steps: int = 10
    save_steps: int = 20
    logging_steps: int = 5
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    scheduler: Literal["constant", "warmup_cosine"] = "warmup_cosine"
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    seed: int = 42


@dataclass
class InferenceConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 64
    repetition_penalty: float = 1.0
    do_sample: bool = True


@dataclass
class NightShadeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    project_name: str = "nightshade-llm"
    experiment_name: str = "tiny"
    output_dir: str = "./outputs"

    @classmethod
    def from_dict(cls, data: dict) -> "NightShadeConfig":
        cfg = cls()
        for section, dc in [("model", cfg.model), ("data", cfg.data), ("training", cfg.training), ("inference", cfg.inference)]:
            for k, v in data.get(section, {}).items():
                if hasattr(dc, k):
                    setattr(dc, k, v)
        for k in ["project_name", "experiment_name", "output_dir"]:
            if k in data:
                setattr(cfg, k, data[k])
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NightShadeConfig":
        if yaml is None:
            raise RuntimeError("pyyaml gerekli")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise RuntimeError("pyyaml gerekli")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def get_tiny_config() -> NightShadeConfig:
    return NightShadeConfig()


def get_small_config() -> NightShadeConfig:
    cfg = NightShadeConfig()
    cfg.model.hidden_size = 512
    cfg.model.num_layers = 8
    cfg.model.intermediate_size = 1360
    cfg.model.max_seq_length = 512
    cfg.model.vocab_size = 40000
    cfg.model.layer_types = ["mamba", "mamba", "mamba", "gated_attn", "gated_attn", "flash_attn", "flash_attn", "flash_attn"]
    cfg.model.__post_init__()
    cfg.training.batch_size = 2
    cfg.training.gradient_accumulation_steps = 8
    cfg.training.max_steps = 100
    return cfg


def get_medium_config() -> NightShadeConfig:
    cfg = get_small_config()
    cfg.model.hidden_size = 640
    cfg.model.num_layers = 10
    cfg.model.intermediate_size = 2048
    cfg.model.layer_types = ["mamba"] * 4 + ["gated_attn"] * 2 + ["flash_attn"] * 4
    cfg.model.__post_init__()
    return cfg
