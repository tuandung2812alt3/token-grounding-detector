#!/usr/bin/env python3
"""Generate image descriptions with an LVLM and label hallucinated tokens via GPT-4o."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from models import build_model
from data.coco_loader import load_coco_samples, train_val_split
from labeling.gpt4_labeler import label_dataset
from utils.config_utils import load_config, get_model_cfg, get_dataset_cfg
from utils.io_utils import save_json, load_json
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   help="Model key, e.g. llava_1_5_7b")
    p.add_argument("--config",      default="configs/model_configs.yaml")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--openai-key",  default=None,
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--num-images",  type=int, default=None,
                   help="Override num_images from config")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--resume",      action="store_true",
                   help="Skip already-processed images")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)

    num_images = args.num_images or dataset_cfg["num_images"]

    images_dir = os.path.join(dataset_cfg["coco_root"], "val2014")
    samples = load_coco_samples(
        images_dir=images_dir,
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=num_images,
        seed=dataset_cfg["seed"],
    )

    splits_path = os.path.join(args.output_dir, "image_splits.json")
    if os.path.exists(splits_path):
        splits = load_json(splits_path)
        train_samples = [s for s in samples if s["image_id"] in set(splits["train"])]
        val_samples   = [s for s in samples if s["image_id"] in set(splits["val"])]
        print(f"[Generate] Loaded existing split: "
              f"{len(splits['train'])} train, {len(splits['val'])} val.")
    else:
        train_samples, val_samples = train_val_split(
            samples,
            train_ratio=dataset_cfg["train_ratio"],
            seed=dataset_cfg["seed"],
        )
        splits = {
            "train": [s["image_id"] for s in train_samples],
            "val":   [s["image_id"] for s in val_samples],
            "test":  [s["image_id"] for s in val_samples],
        }
        save_json(splits, splits_path)
        print(f"[Generate] Split saved: "
              f"{len(train_samples)} train, {len(val_samples)} val/test.")

    gen_path = os.path.join(args.output_dir, "generations.json")
    generation_results = {}
    generation_token_ids = {}

    if args.resume and os.path.exists(gen_path):
        raw = load_json(gen_path)
        generation_results   = {int(k): v["generated_text"]    for k, v in raw.items()}
        generation_token_ids = {int(k): v["response_token_ids"] for k, v in raw.items()}
        print(f"[Generate] Loaded {len(generation_results)} existing generations.")

    pending = [s for s in samples if s["image_id"] not in generation_results]

    if pending:
        print(f"[Generate] Loading model '{args.model}' for generation …")
        wrapper = build_model(args.model, model_cfg, device=args.device)

        for sample in tqdm(pending, desc="Generating"):
            image_id = sample["image_id"]
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image)

            generation_results[image_id] = gen_out.generated_text
            generation_token_ids[image_id] = gen_out.response_token_ids

            save_json(
                {
                    str(k): {
                        "generated_text":    generation_results[k],
                        "response_token_ids": generation_token_ids[k],
                    }
                    for k in generation_results
                },
                gen_path,
            )

        print(f"[Generate] Generations done. Saved to {gen_path}")
    else:
        print("[Generate] All generations already complete.")
        wrapper = build_model(args.model, model_cfg, device=args.device)

    label_path = os.path.join(args.output_dir, "labeling.json")
    print("[Generate] Running GPT-4o labeling …")
    label_dataset(
        samples=samples,
        generation_results=generation_results,
        generation_token_ids=generation_token_ids,
        tokenizer=wrapper.tokenizer,
        output_path=label_path,
        openai_api_key=args.openai_key,
        resume=args.resume,
    )
    print(f"[Generate] Labeling saved to {label_path}")


if __name__ == "__main__":
    main()
