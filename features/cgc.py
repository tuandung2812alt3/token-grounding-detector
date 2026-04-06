"""Cross-modal Grounding Consistency (CGC) — top-k cosine similarity between token and patch embeddings."""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_cgc(
    token_hidden_states: torch.Tensor,
    patch_hidden_states: torch.Tensor,
    top_k_patches: int = 5,
    top_k_pct: float = 0.0,
    text_to_patch_attn: Optional[torch.Tensor] = None,
    mid_layer_pct: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[float, torch.Tensor]:
    """Compute Cross-modal Grounding Consistency for one object token."""
    num_layers = token_hidden_states.shape[0]
    per_layer_cgc = torch.zeros(num_layers)

    token_hs = token_hidden_states.float()
    patch_hs  = patch_hidden_states.float()

    for n in range(num_layers):
        h_t = token_hs[n]
        v_p = patch_hs[n]

        h_norm = F.normalize(h_t.unsqueeze(0), dim=-1)
        v_norm = F.normalize(v_p, dim=-1)
        similarities = (h_norm * v_norm).sum(dim=-1)

        if text_to_patch_attn is not None:
            attn_weights = text_to_patch_attn[n].mean(dim=0).float()
            attn_weights = attn_weights / (attn_weights.sum() + 1e-12)
            per_layer_cgc[n] = (attn_weights * similarities).sum()
        else:
            num_p = similarities.shape[0]
            if top_k_pct > 0:
                k = max(1, min(int(round(num_p * top_k_pct)), num_p))
            else:
                k = min(top_k_patches, num_p)
            top_k_vals, _ = torch.topk(similarities, k=k)
            per_layer_cgc[n] = top_k_vals.mean()

    mid_start = max(0, int(num_layers * mid_layer_pct[0]))
    mid_end   = min(num_layers, int(num_layers * mid_layer_pct[1]))
    if mid_end <= mid_start:
        mid_start, mid_end = 0, num_layers
    cgc_score = per_layer_cgc[mid_start:mid_end].mean().item()

    return cgc_score, per_layer_cgc
