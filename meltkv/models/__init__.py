import importlib

from .hooks import ModelHooks

_REGISTRY = {}


def get_model_hooks(model_type: str) -> ModelHooks:
    if model_type in _REGISTRY:
        return _REGISTRY[model_type]
    if model_type != "llama":
        raise ImportError(
            f"MeltKV public release currently supports LLaMA only (got '{model_type}')."
        )
    module = importlib.import_module(".llama", package=__package__)
    hooks = getattr(module, "model_hooks")
    _REGISTRY[model_type] = hooks
    return hooks
