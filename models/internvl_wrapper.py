"""InternVL-2.5 wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


class InternVLWrapper(BaseLVLMWrapper):
    """Wrapper for InternVL2.5-8B."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[InternVLWrapper] Loading model from {hf_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self._img_ctx_id

        from transformers import GenerationMixin, GenerationConfig
        lm = self.model.language_model
        lm_cls = type(lm)
        if GenerationMixin not in lm_cls.__mro__:
            lm_cls.__bases__ = (GenerationMixin,) + lm_cls.__bases__
        if lm.generation_config is None:
            lm.generation_config = GenerationConfig(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        _original_prepare = lm_cls.prepare_inputs_for_generation
        def _safe_prepare(self_lm, input_ids, past_key_values=None, **kwargs):
            if past_key_values is None:
                kwargs.pop("past_key_values", None)
                return {
                    "input_ids": input_ids,
                    "inputs_embeds": kwargs.get("inputs_embeds"),
                    "attention_mask": kwargs.get("attention_mask"),
                    "past_key_values": None,
                    "use_cache": kwargs.get("use_cache", True),
                }
            return _original_prepare(self_lm, input_ids, past_key_values=past_key_values, **kwargs)
        lm_cls.prepare_inputs_for_generation = _safe_prepare

        print(f"[InternVLWrapper] Loaded. IMG_CONTEXT token id = {self._img_ctx_id}")

    @property
    def num_layers(self) -> int:
        return self.model.language_model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.get("num_visual_tokens", 256)


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        if prompt is None:
            prompt = "Describe this image."

        pixel_values = self._preprocess_image(image)
        input_ids, img_start, img_end = self._build_input_ids_with_image(
            pixel_values, prefix_token_ids=[], user_prompt=prompt
        )
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)

            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            img_mask = (input_ids == self._img_ctx_id).squeeze(0)
            input_embeds[0][img_mask] = vit_embeds.reshape(-1, vit_embeds.shape[-1])

            response_ids = []
            past_key_values = None
            cur_embeds = input_embeds
            eos_token_id = self.tokenizer.eos_token_id

            for _ in range(256):
                out = self.model.language_model(
                    inputs_embeds=cur_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                next_token_id = int(out.logits[0, -1].argmax())
                response_ids.append(next_token_id)
                if next_token_id == eos_token_id:
                    break
                cur_embeds = self.model.language_model.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=self.device)
                )

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
        """Build a full input sequence that mirrors what InternVL's chat"""
        pixel_values = self._preprocess_image(image)

        input_ids, img_start, img_end = self._build_input_ids_with_image(
            pixel_values, prefix_token_ids
        )

        attention_mask = torch.ones_like(input_ids)

        image_flags = torch.ones(
            pixel_values.shape[0], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            out = self._safe_forward(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                pixel_values=pixel_values.to(self.device),
                image_flags=image_flags,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        seq_len = out.attentions[0].shape[-1]
        text_to_patch_attn, text_to_text_attn = _extract_attention_features(
            out.attentions, img_start, img_end, seq_len
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


    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Use InternVL's dynamic resolution preprocessing."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(
                (self.cfg["image_size"], self.cfg["image_size"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(torch.bfloat16).to(self.device)

    def _build_input_ids_with_image(
        self,
        pixel_values: torch.Tensor,
        prefix_token_ids: List[int],
        user_prompt: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Assemble the token id sequence that InternVL's LM backbone sees."""
        num_tiles = pixel_values.shape[0]
        tokens_per_tile = self.cfg.get("num_visual_tokens", 256)
        num_img_tokens = num_tiles * tokens_per_tile

        if user_prompt is None:
            user_prompt = "Describe this image."

        sys_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        sys_ids = self.tokenizer.encode(sys_prompt, add_special_tokens=False)

        img_start_ids = self.tokenizer.encode(IMG_START_TOKEN, add_special_tokens=False)
        img_ctx_ids = [self._img_ctx_id] * num_img_tokens
        img_end_ids = self.tokenizer.encode(IMG_END_TOKEN, add_special_tokens=False)

        user_suffix_ids = self.tokenizer.encode(
            f"\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )

        full_ids = (
            sys_ids
            + img_start_ids
            + img_ctx_ids
            + img_end_ids
            + user_suffix_ids
            + prefix_token_ids
        )

        img_start = len(sys_ids) + len(img_start_ids)
        img_end = img_start + num_img_tokens

        input_ids = torch.tensor([full_ids], dtype=torch.long)
        return input_ids, img_start, img_end


def _extract_attention_features(
    attentions: tuple,
    img_start: int,
    img_end: int,
    seq_len: int,
) -> tuple:
    """Extract text_to_patch_attn [L, n_heads, n_patches] and"""
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [
        i for i in range(seq_len)
        if i not in visual_set and i != last_pos
    ]
    text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])

    return (
        torch.stack(patch_layers, dim=0),
        torch.stack(text_layers,  dim=0),
    )


def _extract_hidden_states(
    hidden_states: tuple,
    img_start: int,
    img_end: int,
) -> tuple:
    """Returns token_hs [L, hidden_dim] and patch_hs [L, n_patches, hidden_dim]."""
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, -1, :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)
