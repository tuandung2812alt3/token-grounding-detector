"""Qwen2.5-VL wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput


class QwenVLWrapper(BaseLVLMWrapper):
    """Wrapper for Qwen2.5-VL-7B-Instruct."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[QwenVLWrapper] Loading model from {hf_name} …")
        self.processor = AutoProcessor.from_pretrained(
            hf_name, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        self._vision_start_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_start_token", "<|vision_start|>")
        )
        self._vision_end_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_end_token", "<|vision_end|>")
        )
        print(
            f"[QwenVLWrapper] Loaded. "
            f"vision_start={self._vision_start_id}, "
            f"vision_end={self._vision_end_id}"
        )

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.get("num_visual_tokens") or 256


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        if prompt is None:
            prompt = "Describe this image."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=256,
            )

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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        partial_response = self.tokenizer.decode(
            prefix_token_ids, skip_special_tokens=True
        )
        full_text = text + partial_response

        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"][0]
        img_start, img_end = self._find_vision_token_range(input_ids)

        with torch.no_grad():
            out = self._safe_forward(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        expanded_seq_len = out.attentions[0].shape[-1]
        text_to_patch_attn, text_to_text_attn = _extract_attention_features(
            out.attentions, img_start, img_end, expanded_seq_len
        )
        token_hidden_states, patch_hidden_states = _extract_hidden_states(
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


    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Locate <|vision_start|> and <|vision_end|> in input_ids and"""
        ids = input_ids.tolist()
        try:
            vs_pos = ids.index(self._vision_start_id)
            ve_pos = ids.index(self._vision_end_id)
            return vs_pos + 1, ve_pos
        except ValueError:
            return 5, 5 + 256


def _extract_text_to_patch_attn(
    attentions: tuple, img_start: int, img_end: int
) -> torch.Tensor:
    layers = []
    for attn in attentions:
        patch_attn = attn[0, :, -1, img_start:img_end]
        layers.append(patch_attn)
    return torch.stack(layers, dim=0)


def _extract_hidden_states(
    hidden_states: tuple, img_start: int, img_end: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, -1, :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)


def _extract_attention_features(attentions, img_start, img_end, seq_len):
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [i for i in range(seq_len) if i not in visual_set and i != last_pos]
    text_idx_tensor = __import__('torch').tensor(text_indices, dtype=__import__('torch').long)
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])
    import torch
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _extract_attention_features(attentions, img_start, img_end, seq_len):
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [i for i in range(seq_len) if i not in visual_set and i != last_pos]
    text_idx_tensor = __import__('torch').tensor(text_indices, dtype=__import__('torch').long)
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])
    import torch
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)
