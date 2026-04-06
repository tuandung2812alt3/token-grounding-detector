"""Feature extraction for POPE yes/no object probing evaluation."""

from __future__ import annotations
import os
import re
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from features.ads import compute_ads
from features.cgc import compute_cgc
from models.base_wrapper import BaseLVLMWrapper
from utils.io_utils import load_pkl, save_pkl


def extract_pope_features_for_dataset(
    model_wrapper: BaseLVLMWrapper,
    pope_samples: List[dict],
    cfg_ads: dict,
    cfg_cgc: dict,
    output_path: str,
    pope_prompt_template: str = "{question}\nAnswer the question using a single word or phrase.",
    yes_only: bool = True,
    resume: bool = True,
) -> List[dict]:
    """Main entry point.  For each POPE question:"""
    all_features: List[dict] = []
    done_qids = set()
    if resume and os.path.exists(output_path):
        all_features = load_pkl(output_path)
        done_qids = {f["question_id"] for f in all_features}
        print(f"[POPE Extractor] Resuming — {len(done_qids)} questions already done.")

    for sample in tqdm(pope_samples, desc="POPE features"):
        qid = sample["question_id"]
        if qid in done_qids:
            continue

        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            print(f"[POPE Extractor] Image not found: {image_path}, skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        question = sample["question"]
        object_word = sample["object_word"]
        gt_label = sample["gt_label"]

        prompt = pope_prompt_template.format(question=question)
        try:
            gen_out = model_wrapper.generate(image, prompt=prompt)
        except Exception as e:
            print(f"[POPE Extractor] Generation failed for qid={qid}: {e}")
            continue

        model_answer = _parse_yes_no(gen_out.generated_text)

        if model_answer == "yes":
            label = 1 if gt_label == "no" else 0
        elif model_answer == "no":
            if yes_only:
                label = -1
            else:
                label = 1 if gt_label == "yes" else 0
        else:
            label = -1

        try:
            raw = _extract_pope_forward(
                model_wrapper, image, prompt, object_word
            )
        except Exception as e:
            print(
                f"[POPE Extractor] Forward pass failed for qid={qid}, "
                f"object='{object_word}': {e}"
            )
            traceback.print_exc()
            continue

        if raw is None:
            print(
                f"[POPE Extractor] Could not locate '{object_word}' in "
                f"input for qid={qid}, skipping."
            )
            continue

        ads_score, ads_per_layer = compute_ads(
            raw["text_to_patch_attn"],
            top_patch_pct=cfg_ads.get("top_patch_pct", 0.10),
            connectivity=cfg_ads.get("connectivity", 8),
            min_blob_area=cfg_ads.get("min_blob_area", 3),
            top_k_layers=cfg_ads.get("top_k_layers", 10),
            per_head_min=cfg_ads.get("per_head_min", False),
            top_k_heads=cfg_ads.get("top_k_heads", 0),
        )
        cgc_score, cgc_per_layer = compute_cgc(
            raw["object_hidden_states"],
            raw["patch_hidden_states"],
            top_k_patches=cfg_cgc.get("top_k_patches", 5),
            top_k_pct=cfg_cgc.get("top_k_pct", 0.0),
            text_to_patch_attn=(
                raw["text_to_patch_attn"]
                if cfg_cgc.get("use_attn_weighting", False) else None
            ),
            mid_layer_pct=tuple(cfg_cgc.get("mid_layer_pct", [0.25, 0.75])),
        )

        last_logits = raw["last_logits"].numpy()
        logits_shifted = last_logits - last_logits.max()
        probs = np.exp(logits_shifted)
        probs = probs / probs.sum()

        pred_id = raw["pred_token_id"]
        token_log_prob = float(np.log(max(probs[pred_id], 1e-12)))

        p_valid = probs[probs > 1e-12]
        token_entropy = float(-np.sum(p_valid * np.log(p_valid)) / np.log(len(probs)))

        token_nll = -token_log_prob

        attn_np = raw["text_to_patch_attn"].numpy()
        var_per_layer = attn_np.sum(axis=-1).mean(axis=-1)
        n_layers = attn_np.shape[0]
        svar_ls = max(0, int(n_layers * 0.15))
        svar_le = min(n_layers, int(n_layers * 0.55))
        svar_score = float(var_per_layer[svar_ls:svar_le].sum())

        mid_l = n_layers // 2
        attn_per_head_mid = attn_np[mid_l].mean(axis=-1).tolist()

        feat = {
            "question_id":           qid,
            "image_id":              sample["image_id"],
            "question":              question,
            "object_word":           object_word,
            "gt_label":              gt_label,
            "model_answer":          model_answer,
            "generated_text":        gen_out.generated_text,
            "label":                 label,
            "token_str":             object_word,
            "token_id":              pred_id,
            "response_token_idx":    0,
            "ads_score":             ads_score,
            "cgc_score":             cgc_score,
            "ads_per_layer":         ads_per_layer.tolist(),
            "cgc_per_layer":         cgc_per_layer.tolist(),
            "token_logits":          last_logits.astype(np.float16),
            "token_log_prob":        token_log_prob,
            "token_entropy":         token_entropy,
            "token_nll":             token_nll,
            "svar_score":            svar_score,
            "attn_per_head_mid":     attn_per_head_mid,
        }
        all_features.append(feat)

        save_pkl(all_features, output_path)

    n_labeled = sum(1 for f in all_features if f["label"] in (0, 1))
    n_hallu = sum(1 for f in all_features if f["label"] == 1)
    n_true = sum(1 for f in all_features if f["label"] == 0)
    n_skip = sum(1 for f in all_features if f["label"] == -1)
    print(
        f"[POPE Extractor] Done. {len(all_features)} total, "
        f"{n_labeled} labeled (true={n_true}, hallu={n_hallu}), "
        f"{n_skip} skipped (no answers)."
    )
    return all_features


def _extract_pope_forward(
    wrapper: BaseLVLMWrapper,
    image: Image.Image,
    prompt: str,
    object_word: str,
) -> Optional[dict]:
    """Run ONE forward pass on [image + prompt] and extract:"""
    wrapper_type = type(wrapper).__name__

    if wrapper_type in ("LLaVAWrapper", "LLaVANextWrapper"):
        return _forward_llava(wrapper, image, prompt, object_word)
    if wrapper_type == "QwenVLWrapper":
        return _forward_qwen(wrapper, image, prompt, object_word)
    if wrapper_type == "InternVLWrapper":
        return _forward_internvl(wrapper, image, prompt, object_word)
    if wrapper_type == "Gemma3Wrapper":
        return _forward_gemma(wrapper, image, prompt, object_word)

    raise NotImplementedError(
        f"POPE feature extraction not implemented for {wrapper_type}. "
        f"Add a _forward_<model>() function in pope_extractor.py."
    )


def _parse_yes_no(text: str) -> str:
    """Parse model output to 'yes', 'no', or 'unknown'."""
    text = text.strip().lower()
    first = text.split()[0] if text.split() else ""
    first = re.sub(r"[^a-z]", "", first)
    if first == "yes":
        return "yes"
    if first == "no":
        return "no"
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "unknown"


def _find_object_position_in_text(
    input_ids_list: List[int],
    tokenizer,
    object_word: str,
    search_start: int,
) -> Optional[int]:
    """Find the first-token position of `object_word` within"""
    text_ids = input_ids_list[search_start:]
    if not text_ids:
        return None

    text_str = tokenizer.decode(text_ids, skip_special_tokens=False)
    text_lower = text_str.lower()

    for pattern in [
        r"\b" + re.escape(object_word.lower()) + r"\b",
        re.escape(object_word.lower()),
    ]:
        m = re.search(pattern, text_lower)
        if m is not None:
            prefix_str = text_str[: m.start()]
            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            return search_start + len(prefix_ids)

    return None


def _extract_from_outputs(
    out,
    tokenizer,
    img_start: int,
    img_end: int,
    obj_pos: int,
) -> dict:
    """Extract feature tensors from a forward-pass output."""
    if (
        out.attentions is None
        or len(out.attentions) == 0
        or out.attentions[0] is None
    ):
        raise RuntimeError(
            "out.attentions is empty/None — flash attention is likely active. "
            "Load model with attn_implementation='eager'."
        )

    num_layers = len(out.attentions)
    expanded_seq_len = out.attentions[0].shape[-1]
    last_pos = expanded_seq_len - 1

    if obj_pos < 0 or obj_pos >= expanded_seq_len:
        raise ValueError(
            f"Object position {obj_pos} out of bounds "
            f"(seq_len={expanded_seq_len}). "
            f"img_start={img_start}, img_end={img_end}."
        )

    patch_attn_layers = []
    for layer_attn in out.attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_attn_layers.append(row[:, img_start:img_end])

    text_to_patch_attn = torch.stack(patch_attn_layers, dim=0).cpu()

    obj_hs_layers = []
    patch_hs_layers = []
    for l in range(num_layers):
        hs = out.hidden_states[l + 1]
        obj_hs_layers.append(hs[0, obj_pos, :])
        patch_hs_layers.append(hs[0, img_start:img_end, :])

    object_hidden_states = torch.stack(obj_hs_layers, dim=0).cpu()
    patch_hidden_states = torch.stack(patch_hs_layers, dim=0).cpu()

    last_logits = out.logits[0, -1].float().cpu()
    pred_id = last_logits.argmax().item()
    pred_str = tokenizer.decode([pred_id], skip_special_tokens=True).strip()

    del out
    torch.cuda.empty_cache()

    return {
        "text_to_patch_attn":   text_to_patch_attn,
        "object_hidden_states": object_hidden_states,
        "patch_hidden_states":  patch_hidden_states,
        "last_logits":          last_logits,
        "pred_token_id":        pred_id,
        "pred_token_str":       pred_str,
    }


def _forward_llava(
    wrapper,
    image: Image.Image,
    prompt: str,
    object_word: str,
) -> Optional[dict]:

    from models.llava_wrapper import IMAGE_TOKEN_INDEX, NUM_VISUAL_TOKENS

    inputs = wrapper.processor(
        text=prompt, images=image, return_tensors="pt",
    ).to(wrapper.device, torch.float16)

    input_ids = inputs["input_ids"][0]
    input_ids_list = input_ids.tolist()

    img_mask = (input_ids == IMAGE_TOKEN_INDEX)
    if img_mask.any():
        img_placeholder_pos = img_mask.nonzero(as_tuple=True)[0][0].item()
        num_placeholders = int(img_mask.sum().item())
    else:
        img_placeholder_pos = 1
        num_placeholders = 1

    img_start_exp = img_placeholder_pos
    img_end_exp = img_placeholder_pos + NUM_VISUAL_TOKENS
    expansion_offset = NUM_VISUAL_TOKENS - num_placeholders

    text_start_original = img_placeholder_pos + num_placeholders
    obj_pos_original = _find_object_position_in_text(
        input_ids_list, wrapper.tokenizer, object_word, text_start_original
    )
    if obj_pos_original is None:
        return None

    obj_pos_expanded = obj_pos_original + expansion_offset

    with torch.no_grad():
        out = wrapper._safe_forward(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    return _extract_from_outputs(
        out, wrapper.tokenizer,
        img_start_exp, img_end_exp, obj_pos_expanded
    )


def _forward_qwen(
    wrapper,
    image: Image.Image,
    prompt: str,
    object_word: str,
) -> Optional[dict]:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = wrapper.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = wrapper.processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(wrapper.device)

    input_ids = inputs["input_ids"][0]
    input_ids_list = input_ids.tolist()

    vis_start_id = wrapper._vision_start_id
    vis_end_id = wrapper._vision_end_id

    img_start = None
    img_end = None
    for i, tid in enumerate(input_ids_list):
        if tid == vis_start_id and img_start is None:
            img_start = i + 1
        if tid == vis_end_id and img_start is not None:
            img_end = i
            break

    if img_start is None or img_end is None:
        raise RuntimeError("Could not find vision start/end markers in Qwen input.")

    text_search_start = img_end + 1
    obj_pos = _find_object_position_in_text(
        input_ids_list, wrapper.tokenizer, object_word, text_search_start
    )
    if obj_pos is None:
        return None

    with torch.no_grad():
        out = wrapper._safe_forward(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    return _extract_from_outputs(
        out, wrapper.tokenizer, img_start, img_end, obj_pos
    )


def _forward_internvl(
    wrapper,
    image: Image.Image,
    prompt: str,
    object_word: str,
) -> Optional[dict]:

    pixel_values = wrapper._preprocess_image(image)

    input_ids, img_start, img_end = wrapper._build_input_ids_with_image(
        pixel_values, prefix_token_ids=[], user_prompt=prompt,
    )
    input_ids_list = input_ids[0].tolist()

    obj_pos = _find_object_position_in_text(
        input_ids_list, wrapper.tokenizer, object_word, img_end
    )
    if obj_pos is None:
        return None

    attention_mask = torch.ones_like(input_ids)
    image_flags = torch.ones(
        pixel_values.shape[0], dtype=torch.long, device=wrapper.device
    )

    with torch.no_grad():
        out = wrapper._safe_forward(
            input_ids=input_ids.to(wrapper.device),
            attention_mask=attention_mask.to(wrapper.device),
            pixel_values=pixel_values.to(wrapper.device),
            image_flags=image_flags,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    return _extract_from_outputs(
        out, wrapper.tokenizer, img_start, img_end, obj_pos
    )


def _forward_gemma(
    wrapper,
    image: Image.Image,
    prompt: str,
    object_word: str,
) -> Optional[dict]:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = wrapper.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(wrapper.device, dtype=wrapper.model.dtype)

    input_ids_list = inputs["input_ids"][0].tolist()

    img_start, img_end = wrapper._find_image_token_range(input_ids_list)

    text_search_start = img_end
    obj_pos = _find_object_position_in_text(
        input_ids_list, wrapper.tokenizer, object_word, text_search_start
    )
    if obj_pos is None:
        return None

    with torch.no_grad():
        out = wrapper._safe_forward(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    return _extract_from_outputs(
        out, wrapper.tokenizer, img_start, img_end, obj_pos
    )
