"""Attention Dispersion Score (ADS) — mass-weighted background entropy."""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure


def compute_ads(
    text_to_patch_attn: torch.Tensor,
    top_patch_pct: float = 0.10,
    connectivity: int = 8,
    min_blob_area: int = 3,
    top_k_layers: int = 10,
    per_head_min: bool = False,
    top_k_heads: int = 0,
    scale_blob_area: bool = True,
    token_hidden_states: Optional[torch.Tensor] = None,
    patch_hidden_states: Optional[torch.Tensor] = None,
) -> tuple[float, torch.Tensor]:
    num_layers, num_heads, num_patches = text_to_patch_attn.shape
    attn_np = text_to_patch_attn.float().numpy()
    grid_H, grid_W = _find_hw(num_patches)

    k_heads = 1 if per_head_min else top_k_heads

    if connectivity == 8:
        struct = generate_binary_structure(2, 2)
    else:
        struct = generate_binary_structure(2, 1)

    layer_maps = np.zeros((num_layers, num_patches))
    for n in range(num_layers):
        la = attn_np[n]
        if k_heads > 0:
            S_img = la.sum(axis=1)
            topk = np.argsort(S_img)[::-1][:k_heads]
            layer_maps[n] = la[topk].mean(axis=0)
        else:
            layer_maps[n] = la.mean(axis=0)

    per_layer_ads = torch.zeros(num_layers)
    for n in range(num_layers):
        per_layer_ads[n] = _compute_mass_weighted_bg_entropy(
            layer_maps[n], num_patches, grid_H, grid_W,
            top_patch_pct, struct, min_blob_area,
        )

    mid_start = max(0, int(num_layers * 0.25))
    mid_end = min(num_layers, int(num_layers * 0.75))
    ads_score = per_layer_ads[mid_start:mid_end].mean().item()

    return ads_score, per_layer_ads


def _compute_mass_weighted_bg_entropy(
    A_np: np.ndarray,
    num_patches: int,
    grid_H: int,
    grid_W: int,
    top_patch_pct: float,
    struct: np.ndarray,
    min_blob_area: int,
) -> float:
    """ADS = (1 - fg_blob_mass) × (bg_entropy / log(N))"""
    A_np = A_np - A_np.min()
    total = A_np.sum()
    if total <= 0:
        return 1.0
    A_np = A_np / total

    n_retain = max(1, int(round(num_patches * top_patch_pct)))
    threshold_idx = np.argpartition(A_np, -n_retain)[-n_retain:]
    fg_binary = np.zeros(num_patches, dtype=bool)
    fg_binary[threshold_idx] = True

    bg_mask = ~fg_binary
    n_bg = int(bg_mask.sum())
    if n_bg < 2:
        bg_entropy_norm = 0.0
    else:
        bg = A_np[bg_mask]
        bg_sum = bg.sum()
        if bg_sum <= 0:
            bg_entropy_norm = 0.0
        else:
            bg = bg / bg_sum
            bg = np.clip(bg, 1e-12, None)
            H_bg = float(-np.sum(bg * np.log(bg)))
            bg_entropy_norm = H_bg / np.log(num_patches)

    fg_2d = fg_binary.reshape(grid_H, grid_W)
    labeled, n_components = label(fg_2d, structure=struct)
    labeled_flat = labeled.reshape(-1)

    valid_blob_mask = np.zeros(num_patches, dtype=bool)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled_flat == comp_id
        if comp_mask.sum() >= min_blob_area:
            valid_blob_mask |= comp_mask

    fg_blob_mass = float(A_np[valid_blob_mask].sum())

    return float((1.0 - fg_blob_mass) * bg_entropy_norm)


def _find_hw(num_patches: int) -> tuple[int, int]:
    """Find H, W such that H * W == num_patches, as close to square as possible."""
    for h in range(int(num_patches ** 0.5), 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    return 1, num_patches