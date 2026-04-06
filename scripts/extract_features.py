#!/usr/bin/env python3
"""Extract ADS and CGC features for all labeled object tokens."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coco_loader import load_coco_samples
from features.extractor import extract_features_for_dataset
from models import build_model
from utils.config_utils import (
    load_config, get_model_cfg, get_dataset_cfg, get_ads_cfg, get_cgc_cfg
)
from utils.io_utils import load_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--config",     default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    model_cfg   = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    ads_cfg     = get_ads_cfg(config)
    cgc_cfg     = get_cgc_cfg(config)

    images_dir = os.path.join(dataset_cfg["coco_root"], "val2014")
    samples = load_coco_samples(
        images_dir=images_dir,
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=dataset_cfg["num_images"],
        seed=dataset_cfg["seed"],
    )

    label_path = os.path.join(args.output_dir, "labeling.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Labeling file not found: {label_path}\n"
            "Run generate_and_label.py first."
        )
    raw_labels = load_json(label_path)
    labeling_results = {int(k): v for k, v in raw_labels.items()}
    print(f"[Extract] Loaded labeling for {len(labeling_results)} images.")

    print(f"[Extract] Loading model '{args.model}' …")
    wrapper = build_model(args.model, model_cfg, device=args.device)

    output_path = os.path.join(args.output_dir, "features.pkl")
    extract_features_for_dataset(
        model_wrapper=wrapper,
        coco_samples=samples,
        labeling_results=labeling_results,
        cfg_ads=ads_cfg,
        cfg_cgc=cgc_cfg,
        output_path=output_path,
        resume=args.resume,
    )
    print(f"[Extract] Features saved to {output_path}")


if __name__ == "__main__":
    main()
