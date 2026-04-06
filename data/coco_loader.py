import json
import os
import random
from pathlib import Path
from typing import Optional


def load_coco_samples(
    images_dir: str,
    instances_file: str,
    captions_file: str,
    num_images: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load COCO samples with image paths, instances, and captions.

    Args:
        images_dir:      Path to image folder, e.g. 'data/coco/val2014'.
        instances_file:  Path to instances JSON, e.g. 'data/coco/annotations/instances_val2014.json'.
        captions_file:   Path to captions JSON, e.g. 'data/coco/annotations/captions_val2014.json'.
        num_images:      If set, randomly subsample to this many images.
        seed:            Random seed for reproducibility.

    Returns:
        List of dicts with keys: image_id, image_path, file_name, captions, annotations.
    """
    images_dir = Path(images_dir)
    instances_file = Path(instances_file)
    captions_file = Path(captions_file)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not instances_file.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_file}")
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    with open(instances_file, "r") as f:
        instances_data = json.load(f)

    with open(captions_file, "r") as f:
        captions_data = json.load(f)

    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in instances_data["images"]
    }

    id_to_category: dict[int, str] = {
        cat["id"]: cat["name"] for cat in instances_data.get("categories", [])
    }

    id_to_annotations: dict[int, list[dict]] = {}
    for ann in instances_data.get("annotations", []):
        iid = ann["image_id"]
        id_to_annotations.setdefault(iid, []).append({
            "id": ann["id"],
            "bbox": ann.get("bbox"),
            "area": ann.get("area"),
            "category_id": ann.get("category_id"),
            "category_name": id_to_category.get(ann.get("category_id"), ""),
            "iscrowd": ann.get("iscrowd", 0),
        })

    id_to_captions: dict[int, list[str]] = {}
    for ann in captions_data.get("annotations", []):
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])

    samples = []
    for image_id, file_name in id_to_filename.items():
        image_path = images_dir / file_name
        if not image_path.exists():
            continue
        samples.append({
            "image_id": image_id,
            "image_path": str(image_path.resolve()),
            "file_name": file_name,
            "captions": id_to_captions.get(image_id, []),
            "annotations": id_to_annotations.get(image_id, []),
        })

    if num_images is not None and num_images < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_images)

    print(f"[coco_loader] Loaded {len(samples)} samples from '{images_dir.name}'.")
    return samples


def train_val_split(
    samples: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train and val subsets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    data = samples.copy()
    random.seed(seed)
    random.shuffle(data)

    n_train = max(1, int(len(data) * train_ratio))
    train_samples = data[:n_train]
    val_samples   = data[n_train:]

    print(
        f"[coco_loader] Split → train: {len(train_samples)}, val: {len(val_samples)}"
    )
    return train_samples, val_samples