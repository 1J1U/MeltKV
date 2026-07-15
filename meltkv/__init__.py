

__version__ = "0.1.0"

__all__ = ["MeltKVCache", "MeltKVConfig", "load_config", "nbits_to_dtype"]


def __getattr__(name):
    if name in ("MeltKVCache", "nbits_to_dtype"):
        from .cache import MeltKVCache, nbits_to_dtype
        return MeltKVCache if name == "MeltKVCache" else nbits_to_dtype
    if name in ("MeltKVConfig", "load_config"):
        from .config import MeltKVConfig, load_config
        return MeltKVConfig if name == "MeltKVConfig" else load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
