"""LLaVA-1.5 wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput

IMAGE_TOKEN_INDEX = -200
NUM_VISUAL_TOKENS = 576


class LLaVAWrapper(BaseLVLMWrapper):
    """Wrapper for LLaVA-1.5-7B (HF transformers implementation)."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[LLaVAWrapper] Loading model from {hf_name} …")
        self.processor = AutoProcessor.from_pretrained(hf_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print("[LLaVAWrapper] Model loaded.")


    @property
    def num_layers(self) -> int:
        return self.model.language_model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return NUM_VISUAL_TOKENS


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        if prompt is None:
            prompt = self.cfg["prompt_template"]

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=256,
            )

        prompt_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response_tokens = [
            self.tokenizer.decode([tid], skip_special_tokens=False)
            for tid in response_ids
        ]

        return GenerationOutput(
            image_id=-1,
            generated_text=generated_text,
            response_token_ids=response_ids,
            response_tokens=response_tokens,
        )


    def extract_token_features(
        self,
        image: Image.Image,
        prefix_token_ids: List[int],
        response_token_idx: int,
    ) -> ModelOutput:
        """Runs a forward pass with prompt+partial_response and returns"""
        prompt_text = self.cfg["prompt_template"]
        partial_text = self.tokenizer.decode(prefix_token_ids, skip_special_tokens=True)
        full_prompt = prompt_text + partial_text

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        input_ids = inputs["input_ids"]

        img_placeholder_mask = (input_ids[0] == IMAGE_TOKEN_INDEX)
        if img_placeholder_mask.any():
            img_placeholder_pos = img_placeholder_mask.nonzero(as_tuple=True)[0][0].item()
            img_start = img_placeholder_pos
            img_end = img_start + NUM_VISUAL_TOKENS
        else:
            img_start, img_end = self._find_img_range_from_embeds(inputs)

        with torch.no_grad():
            out = self._safe_forward(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        if (
            out.attentions is None
            or len(out.attentions) == 0
            or out.attentions[0] is None
        ):
            raise RuntimeError(
                "out.attentions is empty or None. This usually means flash attention "
                "is active and suppressing attention output. "
                "Fix: load the model with attn_implementation='eager':\n"
                "  LlavaForConditionalGeneration.from_pretrained(..., "
                "attn_implementation='eager')"
            )

        expanded_seq_len = out.attentions[0].shape[-1]
        text_to_patch_attn, text_to_text_attn = self._extract_attention_features(
            out.attentions, img_start, img_end, expanded_seq_len
        )

        token_hidden_states, patch_hidden_states = self._extract_hidden_states(
            out.hidden_states, img_start, img_end
        )

        pred_token_id = out.logits[0, -1].argmax().item()
        pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
        last_logits = out.logits[0, -1].float().cpu()

        return ModelOutput(
            token_id=pred_token_id,
            token_str=pred_token_str,
            text_to_patch_attn=text_to_patch_attn.cpu(),
            text_to_text_attn=text_to_text_attn.cpu(),
            token_hidden_states=token_hidden_states.cpu(),
            patch_hidden_states=patch_hidden_states.cpu(),
            response_token_idx=response_token_idx,
            token_logits=last_logits,
        )


    def _find_img_range_from_embeds(self, inputs: dict) -> Tuple[int, int]:
        """Fallback: estimate img_start by counting non-image prompt tokens."""
        return 4, 4 + NUM_VISUAL_TOKENS

    @staticmethod
    def _extract_attention_features(
        attentions: tuple,
        img_start: int,
        img_end: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract two attention tensors from the last token across all layers:"""
        visual_set = set(range(img_start, img_end))
        last_pos   = seq_len - 1
        text_indices = [
            i for i in range(seq_len)
            if i not in visual_set and i != last_pos
        ]
        text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

        patch_layers = []
        text_layers  = []
        for layer_attn in attentions:
            row = layer_attn[0, :, last_pos, :]
            patch_layers.append(row[:, img_start:img_end])
            text_layers.append(row[:, text_idx_tensor])

        return (
            torch.stack(patch_layers, dim=0),
            torch.stack(text_layers,  dim=0),
        )

    @staticmethod
    def _extract_hidden_states(
        hidden_states: tuple,
        img_start: int,
        img_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: tuple of [1, seq_len, hidden_dim], length = num_layers+1"""
        token_list, patch_list = [], []
        for hs in hidden_states[1:]:
            token_list.append(hs[0, -1, :])
            patch_list.append(hs[0, img_start:img_end, :])
        token_hs = torch.stack(token_list, dim=0)
        patch_hs = torch.stack(patch_list, dim=0)
        return token_hs, patch_hs


from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

class LLaVANextWrapper(LLaVAWrapper):
    """Wrapper for LLaVA-Next (1.6) — dynamic resolution variant of LLaVA."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[LLaVANextWrapper] Loading model from {hf_name} …")
        self.processor = LlavaNextProcessor.from_pretrained(hf_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print("[LLaVANextWrapper] Model loaded.")
