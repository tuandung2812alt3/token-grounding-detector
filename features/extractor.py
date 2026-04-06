"""Feature extraction for MS-COCO image captioning. Runs prefix forward passes and computes ADS + CGC per token."""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput
from features.ads import compute_ads
from features.cgc import compute_cgc
from features.attention import compute_alpha_img_alpha_text
from utils.io_utils import load_pkl, save_pkl


def extract_features_for_dataset(
    model_wrapper: BaseLVLMWrapper,
    coco_samples: List[dict],
    labeling_results: Dict[int, dict],
    cfg_ads: dict,
    cfg_cgc: dict,
    output_path: str,
    resume: bool = True,
) -> List[dict]:
    """Main entry point.  Iterates over `coco_samples`, extracts features"""
    all_features: List[dict] = []
    done_image_ids = set()
    if resume and os.path.exists(output_path):
        all_features = load_pkl(output_path)
        done_image_ids = {f["image_id"] for f in all_features}
        print(f"[Extractor] Resuming — {len(done_image_ids)} images already done.")

    for sample in tqdm(coco_samples, desc="Extracting features"):
        image_id = sample["image_id"]
        if image_id in done_image_ids:
            continue

        label_info = labeling_results.get(image_id)
        if label_info is None:
            continue

        labeling_generated_text = label_info.get("generated_text", "")
        if not labeling_generated_text:
            continue
        labeling_response_ids = model_wrapper.tokenizer.encode(
            labeling_generated_text, add_special_tokens=False
        )
        gen_out = GenerationOutput(
            image_id=image_id,
            generated_text=labeling_generated_text,
            response_token_ids=labeling_response_ids,
            response_tokens=[
                model_wrapper.tokenizer.decode([tid], skip_special_tokens=False)
                for tid in labeling_response_ids
            ],
        )

        image = Image.open(sample["image_path"]).convert("RGB")

        object_token_spans = label_info.get("object_token_spans", [])
        if not object_token_spans:
            continue

        prompt_token_ids = _get_prompt_token_ids(model_wrapper, gen_out)

        image_features = []
        for span in object_token_spans:
            first_idx = span["token_indices"][0]

            prefix_ids = prompt_token_ids + gen_out.response_token_ids[:first_idx]

            try:
                model_out = model_wrapper.extract_token_features(
                    image=image,
                    prefix_token_ids=prefix_ids,
                    response_token_idx=first_idx,
                )
            except Exception as e:
                import traceback
                print(f"[Extractor] Warning — forward pass failed for "
                      f"image {image_id}, token '{span['word']}' "
                      f"(first_idx={first_idx}, prefix_len={len(prefix_ids)}): {e}")
                traceback.print_exc()
                continue

            ads_score, ads_per_layer = compute_ads(
                model_out.text_to_patch_attn,
                top_patch_pct=cfg_ads.get("top_patch_pct", 0.10),
                connectivity=cfg_ads.get("connectivity", 8),
                min_blob_area=cfg_ads.get("min_blob_area", 3),
                top_k_layers=cfg_ads.get("top_k_layers", 10),
                per_head_min=cfg_ads.get("per_head_min", False),
                top_k_heads=cfg_ads.get("top_k_heads", 0),
                token_hidden_states=model_out.token_hidden_states,
                patch_hidden_states=model_out.patch_hidden_states,
            )
            cgc_score, cgc_per_layer = compute_cgc(
                model_out.token_hidden_states,
                model_out.patch_hidden_states,
                top_k_patches=cfg_cgc.get("top_k_patches", 5),
                top_k_pct=cfg_cgc.get("top_k_pct", 0.0),
                text_to_patch_attn=(
                    model_out.text_to_patch_attn
                    if cfg_cgc.get("use_attn_weighting", False) else None
                ),
                mid_layer_pct=tuple(cfg_cgc.get("mid_layer_pct", [0.25, 0.75])),
            )
            alpha_img_per_layer, alpha_text_per_layer = compute_alpha_img_alpha_text(
                text_to_patch_attn=model_out.text_to_patch_attn,
                text_to_text_attn=model_out.text_to_text_attn,
            )

            baseline = _compute_baseline_features(model_out)

            feat = {
                "image_id":              image_id,
                "token_str":             span["word"],
                "token_id":              model_out.token_id,
                "response_token_idx":    first_idx,
                "label":                 span["label"],
                "ads_score":             ads_score,
                "cgc_score":             cgc_score,
                "ads_per_layer":         ads_per_layer.tolist(),
                "cgc_per_layer":         cgc_per_layer.tolist(),
                "alpha_img_per_layer":   alpha_img_per_layer.tolist(),
                "alpha_text_per_layer":  alpha_text_per_layer.tolist(),
                **baseline,
            }
            image_features.append(feat)

        all_features.extend(image_features)

        save_pkl(all_features, output_path)

    print(f"[Extractor] Done. {len(all_features)} object tokens saved to {output_path}.")
    return all_features


def _get_prompt_token_ids(
    wrapper: BaseLVLMWrapper,
    gen_out: GenerationOutput,
) -> List[int]:
    """Returns an EMPTY list because the wrapper's extract_token_features"""
    return []


def _compute_baseline_features(model_out) -> dict:
    """Compute baseline method features from a ModelOutput."""
    result = {}

    if model_out.token_logits is not None:
        logits = model_out.token_logits.numpy().astype(np.float32)
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs = probs / probs.sum()

        token_id = model_out.token_id
        token_log_prob = float(np.log(max(probs[token_id], 1e-12)))

        p_valid = probs[probs > 1e-12]
        token_entropy = float(-np.sum(p_valid * np.log(p_valid)) / np.log(len(probs)))

        result["token_logits"] = logits.astype(np.float16)
        result["token_log_prob"] = token_log_prob
        result["token_entropy"] = token_entropy
        result["token_nll"] = -token_log_prob

    attn_np = model_out.text_to_patch_attn.float().numpy()
    n_layers = attn_np.shape[0]

    var_per_layer = attn_np.sum(axis=-1).mean(axis=-1)
    svar_ls = max(0, int(n_layers * 0.15))
    svar_le = min(n_layers, int(n_layers * 0.55))
    result["svar_score"] = float(var_per_layer[svar_ls:svar_le].sum())

    mid_l = n_layers // 2
    result["attn_per_head_mid"] = attn_np[mid_l].mean(axis=-1).tolist()

    return result

