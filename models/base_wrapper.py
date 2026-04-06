"""Base class for LVLM wrappers. Defines the interface for generation and feature extraction."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image


@dataclass
class ModelOutput:
    """Structured output of a single prefix forward pass."""

    token_id: int
    token_str: str

    text_to_patch_attn: torch.Tensor

    text_to_text_attn: torch.Tensor

    token_hidden_states: torch.Tensor

    patch_hidden_states: torch.Tensor

    response_token_idx: int

    token_logits: Optional[torch.Tensor] = None


@dataclass
class GenerationOutput:
    """Full generation result for one image."""
    image_id: int
    generated_text: str
    response_token_ids: List[int]
    response_tokens: List[str]


class BaseLVLMWrapper(ABC):
    """Abstract wrapper for Large Vision-Language Models."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()


    @abstractmethod
    def _load_model(self) -> None:
        """Load model, tokenizer/processor onto self.device."""
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        """Generate a description for `image` using greedy decoding"""
        ...

    @abstractmethod
    def extract_token_features(
        self,
        image: Image.Image,
        prefix_token_ids: List[int],
        response_token_idx: int,
    ) -> ModelOutput:
        """Run ONE forward pass using `prefix_token_ids` as input and"""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers in the LM backbone."""
        ...

    @property
    @abstractmethod
    def num_visual_tokens(self) -> int:
        """Number of visual patch tokens in the LM sequence."""
        ...


    def _ids_to_str(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @torch.no_grad()
    def _safe_forward(self, **kwargs) -> dict:
        """Wrapper around model(**kwargs) that always disables gradients"""
        try:
            return self.model(**kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU OOM during forward pass. Try reducing image resolution "
                "or processing fewer object tokens at once."
            ) from e

