"""NightShade tiny training pipeline."""

from .core.config import NightShadeConfig, ModelConfig, DataConfig, TrainingConfig

__all__ = ["NightShadeConfig", "ModelConfig", "DataConfig", "TrainingConfig"]

try:
    from .model import NightShadeLM, create_tiny_model, create_small_model, create_medium_model

    __all__.extend(
        [
            "NightShadeLM",
            "create_tiny_model",
            "create_small_model",
            "create_medium_model",
        ]
    )
except ModuleNotFoundError:
    # if torch is not installed, keep config imports available
    pass
