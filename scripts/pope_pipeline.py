#!/usr/bin/env python3
"""End-to-end POPE evaluation: extract features, train, and evaluate."""

import argparse
import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pope_loader import load_pope_questions
from features.pope_extractor import extract_pope_features_for_dataset
from detection.train import (
    build_feature_matrix,
    train_and_evaluate,
)
from detection.evaluate import (
    evaluate_ads_threshold,
    evaluate_cgc_threshold,
)
from utils.config_utils import (
    load_config, get_model_cfg, get_classifier_cfgs, get_ads_cfg, get_cgc_cfg
)
from utils.io_utils import save_pkl, load_pkl, save_json, load_json
from models import build_model


POPE_SPLITS = ["random", "popular", "adversarial"]

POPE_PROMPT_TEMPLATE = "{question}\nAnswer the question using a single word or phrase."

_POPE_PROMPT_TEMPLATES = {
    "llava_1_5_7b": "USER: <image>\n{question}\nAnswer the question using a single word or phrase.\nASSISTANT:",
    "llava_next_7b": "[INST] <image>\n{question}\nAnswer the question using a single word or phrase. [/INST]",
    "_default": "{question}\nAnswer the question using a single word or phrase.",
}


def get_pope_prompt_template(model_key: str) -> str:
    """Return the POPE prompt template for the given model."""
    return _POPE_PROMPT_TEMPLATES.get(
        model_key, _POPE_PROMPT_TEMPLATES["_default"]
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="POPE hallucination detection pipeline"
    )
    p.add_argument("--model", required=True,
                   help="Model key from model_configs.yaml")
    p.add_argument("--config", default="configs/model_configs.yaml")
    p.add_argument("--pope-dir", required=True,
                   help="Directory containing POPE JSONL files")
    p.add_argument("--coco-image-dir", required=True,
                   help="Path to COCO val2014 images")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["random"],
                   choices=POPE_SPLITS,
                   help="POPE splits to evaluate (default: random)")
    p.add_argument("--prompt-template", default=None,
                   help="Prompt template with {question} placeholder. "
                        "Auto-detected per model if not set.")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Cap per split (for debugging)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of images for training (rest split 50/50 val/test)")
    p.add_argument("--include-no", action="store_true",
                   help="Also label 'no' answers (default: yes-only)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shap", action="store_true",
                   help="Compute SHAP feature importance (slow)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    clf_cfgs = get_classifier_cfgs(config)
    cfg_ads = get_ads_cfg(config)
    cfg_cgc = get_cgc_cfg(config)

    print(f"\n{'='*60}")
    print(f"  Loading model: {args.model}")
    print(f"{'='*60}")
    wrapper = build_model(args.model, model_cfg, device=args.device)

    prompt_template = args.prompt_template or get_pope_prompt_template(args.model)
    print(f"  Prompt template: {prompt_template!r}")

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  POPE split: {split}")
        print(f"{'='*60}")

        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        results_dir = os.path.join(split_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        feat_path = os.path.join(split_dir, "pope_features.pkl")
        splits_path = os.path.join(split_dir, "image_splits.json")

        pope_samples = load_pope_questions(
            args.pope_dir,
            split=split,
            coco_image_dir=args.coco_image_dir,
            max_questions=args.max_questions,
        )

        all_features = extract_pope_features_for_dataset(
            model_wrapper=wrapper,
            pope_samples=pope_samples,
            cfg_ads=cfg_ads,
            cfg_cgc=cfg_cgc,
            output_path=feat_path,
            pope_prompt_template=prompt_template,
            yes_only=(not args.include_no),
            resume=args.resume,
        )

        pope_acc = _compute_pope_accuracy(all_features)
        pope_acc_path = os.path.join(split_dir, "pope_accuracy.json")
        save_json(pope_acc, pope_acc_path)
        print(f"\n  Standard POPE accuracy ({split}):")
        _print_pope_acc(pope_acc)

        if not os.path.exists(splits_path) or not args.resume:
            image_ids = list({f["image_id"] for f in all_features})
            random.shuffle(image_ids)
            n = len(image_ids)
            n_train = int(n * args.train_ratio)
            n_val = (n - n_train) // 2
            train_ids = set(image_ids[:n_train])
            val_ids = set(image_ids[n_train : n_train + n_val])
            test_ids = set(image_ids[n_train + n_val :])
            save_json(
                {
                    "train": sorted(train_ids),
                    "val": sorted(val_ids),
                    "test": sorted(test_ids),
                },
                splits_path,
            )
        else:
            splits_data = load_json(splits_path)
            train_ids = set(splits_data["train"])
            val_ids = set(splits_data["val"])
            test_ids = set(splits_data["test"])

        print(
            f"\n  Image splits: train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}"
        )

        test_feats = [f for f in all_features if f["image_id"] in test_ids]
        labeled_test = [f for f in test_feats if f["label"] in (0, 1)]

        if len(labeled_test) > 0:
            print(f"\n  Single-metric threshold detectors (test set, n={len(labeled_test)}):")
            print("  ADS-only:")
            ads_m = evaluate_ads_threshold(labeled_test)
            _print_metrics(ads_m)
            print("  CGC-only:")
            cgc_m = evaluate_cgc_threshold(labeled_test)
            _print_metrics(cgc_m)
            save_json(
                {"ads": ads_m, "cgc": cgc_m},
                os.path.join(results_dir, f"{args.model}_{split}_single_metric.json"),
            )
        else:
            print("  [WARN] No labeled test samples — skipping baselines.")

        print(f"\n  Training classifiers ({split})…")
        try:
            clf_results = train_and_evaluate(
                feature_path=feat_path,
                train_image_ids=train_ids,
                val_image_ids=val_ids,
                test_image_ids=test_ids,
                clf_configs=clf_cfgs,
                output_dir=results_dir,
                model_key=f"{args.model}_{split}",
            )
        except ValueError as e:
            print(f"  [WARN] Training failed: {e}")
            print("  This usually means not enough labeled samples in a split.")
            clf_results = {}

        train_feats_abl = [f for f in all_features if f["image_id"] in train_ids and f["label"] in (0,1)]
        val_feats_abl   = [f for f in all_features if f["image_id"] in val_ids   and f["label"] in (0,1)]
        if len(train_feats_abl) > 0 and len(labeled_test) > 0:
            from analysis.ablation import run_feature_ablation, print_feature_ablation_table
            print(f"\n  Feature-type ablation ({split})…")
            abl_results = run_feature_ablation(
                train_features=train_feats_abl,
                val_features=val_feats_abl if val_feats_abl else train_feats_abl,
                test_features=labeled_test,
            )
            abl_txt = print_feature_ablation_table(abl_results)
            with open(os.path.join(results_dir, f"{args.model}_{split}_feature_ablation.txt"), "w") as fh:
                fh.write(abl_txt + "\n")
            save_json(
                {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in abl_results.items()},
                os.path.join(results_dir, f"{args.model}_{split}_feature_ablation.json"),
            )

        if args.shap and clf_results:
            _run_shap(results_dir, args.model, split, all_features, test_ids)

        print(f"\n  [POPE {split}] Done. Results in {results_dir}")

    print(f"\n{'='*60}")
    print(f"  All splits complete for {args.model}")
    print(f"{'='*60}")


def _compute_pope_accuracy(features: list) -> dict:
    """Compute standard POPE metrics from model answers vs ground truth."""
    n_total = len(features)
    if n_total == 0:
        return {}

    n_correct = sum(
        1 for f in features
        if f["model_answer"] == f["gt_label"]
    )
    n_yes_answer = sum(1 for f in features if f["model_answer"] == "yes")
    n_gt_yes = sum(1 for f in features if f["gt_label"] == "yes")

    tp = sum(1 for f in features if f["model_answer"] == "yes" and f["gt_label"] == "yes")
    fp = sum(1 for f in features if f["model_answer"] == "yes" and f["gt_label"] == "no")
    fn = sum(1 for f in features if f["model_answer"] == "no" and f["gt_label"] == "yes")
    tn = sum(1 for f in features if f["model_answer"] == "no" and f["gt_label"] == "no")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_ratio = n_yes_answer / n_total if n_total > 0 else 0.0

    return {
        "accuracy": n_correct / n_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "n_total": n_total,
        "n_correct": n_correct,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _print_pope_acc(m: dict) -> None:
    if not m:
        print("    (no data)")
        return
    print(
        f"    ACC={m['accuracy']:.3f}  PR={m['precision']:.3f}  "
        f"RC={m['recall']:.3f}  F1={m['f1']:.3f}  "
        f"yes_ratio={m['yes_ratio']:.3f}  "
        f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})"
    )


def _print_metrics(m: dict) -> None:
    print(
        f"    PR={m.get('precision', 0):.3f}  "
        f"RC={m.get('recall', 0):.3f}  "
        f"F1={m.get('f1', 0):.3f}  "
        f"ACC={m.get('accuracy', 0):.3f}  "
        f"AUC={m.get('auc', 0):.3f}"
    )


def _run_shap(results_dir, model, split, all_feats, test_ids):
    """Run SHAP importance on the best XGB model."""
    import pickle
    from detection.evaluate import compute_shap_importance

    xgb_path = os.path.join(results_dir, f"{model}_{split}_xgb.pkl")
    if not os.path.exists(xgb_path):
        return
    with open(xgb_path, "rb") as f:
        xgb_clf = pickle.load(f)

    test_feats = [f for f in all_feats if f["image_id"] in test_ids and f["label"] in (0, 1)]
    sample = next((x for x in all_feats if x.get("label") in (0, 1)), None)
    if sample:
        num_layers = len(sample["ads_per_layer"])
        print(f"\n  [SHAP] Computing importance ({model} {split} XGB)…")
        shap_vals = compute_shap_importance(xgb_clf, test_feats, num_layers=num_layers)
        if shap_vals:
            save_json(
                {k: v.tolist() for k, v in shap_vals.items()},
                os.path.join(results_dir, f"{model}_{split}_shap.json"),
            )
            print("  [SHAP] Saved.")


if __name__ == "__main__":
    main()
