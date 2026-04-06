"""GPT-4o based labeling of hallucinated object tokens in generated captions."""

from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI

from labeling.token_finder import find_object_token_spans
from utils.io_utils import save_json, load_json


SYSTEM_PROMPT = (
    "You are a precise hallucination detector. "
    "Follow the instructions exactly and output ONLY a JSON list."
)

USER_TEMPLATE = """\
You are given:
- A list of ground truth object classes (from COCO).
- A detailed description of an image.
- Several captions of the same image.

Your task:
Find all object classes that are mentioned in the description, but are NOT mentioned in any of the captions, and are NOT present in the ground truth list.

Output the result as a list as in the examples. Do NOT add any extra text or provide any explanations.

Examples:
Objects: ["bowl", "broccoli", "carrot"]
Description: There are two bowls of food, one containing a mix of vegetables, such as broccoli and carrots, and the other containing meat. Captions:
- A bowl with broccoli and carrots.
→ Output: ["meat"]

Objects: ["bowl", "broccoli"]
Description: - A bowl full of broccoli.
Captions: - A bowl of green vegetables.
→ Output: []

Now answer:
Objects: {objects}
Description: {description}
Captions:
{captions_formatted}
→ Output:"""


def _format_captions(captions: List[str]) -> str:
    """Format captions as a bullet list, matching the paper's template."""
    return "\n".join(f"- {cap.strip()}" for cap in captions)


def _call_gpt4o(
    client: OpenAI,
    objects: List[str],
    description: str,
    captions: List[str],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[str]:
    """
    Call GPT-4o with the paper's exact prompt and parse the output as a
    Python list of hallucinated object class strings.

    Returns [] on parse failure (conservative: assume no hallucinations).
    """
    captions_formatted = _format_captions(captions)
    user_msg = USER_TEMPLATE.format(
        objects=json.dumps(objects),
        description=description,
        captions_formatted=captions_formatted,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            raw = raw.replace("```json", "").replace("```", "").strip()
            hallucinated = json.loads(raw)
            if isinstance(hallucinated, list):
                return [str(w).lower() for w in hallucinated]
            return []

        except json.JSONDecodeError:
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[GPT4Labeler] Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"[GPT4Labeler] Failed after {max_retries} retries: {e}")
                return []


def label_dataset(
    samples: List[dict],
    generation_results: Dict[int, str],   # image_id → generated_text
    generation_token_ids: Dict[int, List[int]],  # image_id → response_token_ids
    tokenizer,                            # the model's tokenizer (for token finding)
    output_path: str,
    openai_api_key: Optional[str] = None,
    resume: bool = True,
    sleep_between_calls: float = 0.5,
) -> Dict[int, dict]:
    """
    Run GPT-4o labeling on all samples and save results.

    Args:
        samples:               COCO sample list (from coco_loader).
        generation_results:    Dict mapping image_id → generated description string.
        generation_token_ids:  Dict mapping image_id → list of response token ids.
        tokenizer:             The LVLM tokenizer (used for token-position finding).
        output_path:           Path to save/resume the JSON results file.
        openai_api_key:        API key (falls back to OPENAI_API_KEY env var).
        resume:                Skip already-labeled images if output_path exists.
        sleep_between_calls:   Seconds to sleep between API calls.

    Returns:
        Dict mapping image_id → labeling result dict.
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)

    results: Dict[int, dict] = {}
    if resume and os.path.exists(output_path):
        raw = load_json(output_path)
        results = {int(k): v for k, v in raw.items()}
        print(f"[GPT4Labeler] Resuming — {len(results)} images already labeled.")

    for sample in samples:
        image_id = sample["image_id"]
        if image_id in results:
            continue
        if image_id not in generation_results:
            continue

        generated_text = generation_results[image_id]
        coco_objects = sample["coco_objects"]       # list[str], ground-truth classes
        captions = sample["captions"]               # list[str], GT captions

        hallucinated_words = _call_gpt4o(
            client,
            objects=coco_objects,
            description=generated_text,
            captions=captions,
        )
        time.sleep(sleep_between_calls)

        response_token_ids = generation_token_ids.get(image_id, [])
        object_token_spans = find_object_token_spans(
            generated_text=generated_text,
            response_token_ids=response_token_ids,
            hallucinated_words=hallucinated_words,
            coco_objects=coco_objects,
            tokenizer=tokenizer,
        )

        results[image_id] = {
            "image_id": image_id,
            "generated_text": generated_text,
            "hallucinated_words": hallucinated_words,
            "object_token_spans": object_token_spans,
        }

        save_json({str(k): v for k, v in results.items()}, output_path)

    print(f"[GPT4Labeler] Done. {len(results)} images labeled.")
    return results
