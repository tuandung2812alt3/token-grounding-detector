"""Model wrapper factory."""

from models.llava_wrapper import LLaVAWrapper
from models.internvl_wrapper import InternVLWrapper
from models.qwen_wrapper import QwenVLWrapper

_REGISTRY = {
    "llava_1_5_7b": LLaVAWrapper,
    "internvl_2_5_8b": InternVLWrapper,
    "qwen2_5_vl_7b": QwenVLWrapper,
}


def build_model(model_key: str, cfg: dict, device: str = "cuda"):
    if model_key not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Valid: {list(_REGISTRY.keys())}")
    return _REGISTRY[model_key](cfg, device=device)
