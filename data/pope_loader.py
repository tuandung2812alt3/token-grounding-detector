"""Load POPE benchmark questions for hallucination evaluation."""

from __future__ import annotations
import json
import os
import re
from typing import List, Optional

_OBJECT_RE = re.compile(
    r"Is there (?:a|an)\s+(.+?)\s+in the image",
    re.IGNORECASE,
)


def load_pope_questions(
    pope_dir: str,
    split: str = "random",
    coco_image_dir: str = "",
    max_questions: Optional[int] = None,
) -> List[dict]:
    """Load POPE questions from a JSONL file."""
    candidates = [
        f"coco_pope_{split}.json",
        f"pope_{split}.json",
        f"coco_pope_{split}.jsonl",
        f"pope_{split}.jsonl",
    ]
    pope_file = None
    for name in candidates:
        path = os.path.join(pope_dir, name)
        if os.path.exists(path):
            pope_file = path
            break
    if pope_file is None:
        raise FileNotFoundError(
            f"No POPE file found for split '{split}' in {pope_dir}. "
            f"Expected one of: {candidates}"
        )

    samples = []
    with open(pope_file, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            question = item["text"]
            image_file = item["image"]
            gt_label = item["label"].strip().lower()

            m = _OBJECT_RE.search(question)
            object_word = m.group(1).strip().lower() if m else ""
            if not object_word:
                continue

            image_id = _parse_image_id(image_file)
            image_path = (
                os.path.join(coco_image_dir, image_file)
                if coco_image_dir
                else image_file
            )

            samples.append({
                "question_id": item.get("question_id", line_idx),
                "image_id": image_id,
                "image_file": image_file,
                "image_path": image_path,
                "question": question,
                "object_word": object_word,
                "gt_label": gt_label,
            })

            if max_questions and len(samples) >= max_questions:
                break

    print(
        f"[POPE] Loaded {len(samples)} questions from {pope_file} "
        f"(yes={sum(1 for s in samples if s['gt_label']=='yes')}, "
        f"no={sum(1 for s in samples if s['gt_label']=='no')})"
    )
    return samples


def _parse_image_id(filename: str) -> int:
    """Extract numeric COCO image ID from filename."""
    m = re.search(r"(\d{12})", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", os.path.splitext(filename)[0])
    if m:
        return int(m.group(1))
    return abs(hash(filename)) % (10 ** 9)
