"""Find token positions of hallucinated words in the generated sequence."""

from __future__ import annotations
import re
from typing import List, Optional


def find_object_token_spans(
    generated_text: str,
    response_token_ids: List[int],
    hallucinated_words: List[str],
    coco_objects: List[str],
    tokenizer,
) -> List[dict]:
    """Find token positions of all object words (both true and hallucinated)"""
    hallucinated_set = {w.lower() for w in hallucinated_words}
    coco_set = {w.lower() for w in coco_objects}

    candidate_words = {}
    for w in hallucinated_set:
        candidate_words[w] = 1
    for w in coco_set:
        if w not in candidate_words:
            candidate_words[w] = 0

    spans = []
    text_lower = generated_text.lower()

    for word, label in candidate_words.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        for m in re.finditer(pattern, text_lower):
            char_start = m.start()
            char_end = m.end()

            prefix_text = generated_text[:char_start]

            prefix_ids = tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            first_token_idx = len(prefix_ids)

            word_surface = generated_text[char_start:char_end]
            word_ids = tokenizer.encode(word_surface, add_special_tokens=False)
            n_toks = len(word_ids)

            last_idx = first_token_idx + n_toks - 1
            if last_idx >= len(response_token_ids):
                continue

            token_indices = list(range(first_token_idx, first_token_idx + n_toks))

            spans.append({
                "word": word,
                "token_indices": token_indices,
                "label": label,
            })
            break

    spans.sort(key=lambda x: x["token_indices"][0])
    return spans
