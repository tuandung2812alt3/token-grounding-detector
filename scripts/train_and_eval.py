#!/usr/bin/env python3
"""Train classifiers on extracted features and evaluate on the test set."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.train import train_and_evaluate
from detection.evaluate import (
    evaluate_ads_threshold,
    evaluate_cgc_threshold,
    layerwise_analysis,
    compute_shap_importance,
)
from utils.config_utils import load_config, get_classifier_cfgs
from utils.io_utils import load_pkl, load_json, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--config",     default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shap",       action="store_true",
                   help="Compute SHAP feature importance (slow)")
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    clf_cfgs = get_classifier_cfgs(config)

    feat_path   = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for p in [feat_path, splits_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    splits = load_json(splits_path)
    train_ids = set(splits["train"])
    val_ids   = set(splits["val"])
    test_ids  = set(splits["test"])

    all_feats = load_pkl(feat_path)

    test_feats = [f for f in all_feats if f["image_id"] in test_ids]
    val_feats  = [f for f in all_feats if f["image_id"] in val_ids]

    print("\n" + "=" * 60)
    print("  Single-metric threshold evaluators (test set)")
    print("=" * 60)

    ads_tau_candidates = [f["ads_score"] for f in val_feats if f.get("label") in (0,1)]
    cgc_tau_candidates = [f["cgc_score"] for f in val_feats if f.get("label") in (0,1)]

    print("\nADS-only detector:")
    ads_metrics = evaluate_ads_threshold(test_feats)
    _print_metrics(ads_metrics)

    print("\nCGC-only detector:")
    cgc_metrics = evaluate_cgc_threshold(test_feats)
    _print_metrics(cgc_metrics)

    save_json(
        {"ads": ads_metrics, "cgc": cgc_metrics},
        os.path.join(results_dir, f"{args.model}_single_metric.json"),
    )

    print("\n" + "=" * 60)
    print("  Layer-wise analysis — ADS (attention entropy)")
    print("=" * 60)
    lw_ads = layerwise_analysis(all_feats, feature_key="ads_per_layer")
    save_json(lw_ads, os.path.join(results_dir, "layerwise_ads.json"))

    print("\n" + "=" * 60)
    print("  Layer-wise analysis — CGC (feature similarity)")
    print("=" * 60)
    lw_cgc = layerwise_analysis(all_feats, feature_key="cgc_per_layer")
    save_json(lw_cgc, os.path.join(results_dir, "layerwise_cgc.json"))

    print("\n" + "=" * 60)
    print("  Training classifiers (XGB / RF / MLP)")
    print("=" * 60)
    all_results = train_and_evaluate(
        feature_path=feat_path,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        test_image_ids=test_ids,
        clf_configs=clf_cfgs,
        output_dir=results_dir,
        model_key=args.model,
    )

    print("\n" + "=" * 60)
    print("  Feature-type ablation")
    print("=" * 60)

    from analysis.ablation import run_feature_ablation, print_feature_ablation_table

    val_feats  = [f for f in all_feats if f["image_id"] in val_ids]
    ablation_results = run_feature_ablation(
        train_feats=[f for f in all_feats if f["image_id"] in train_ids],
        val_features=val_feats if val_feats else [f for f in all_feats if f["image_id"] in train_ids],
        test_features=test_feats,
    )
    ablation_txt = print_feature_ablation_table(ablation_results)

    with open(os.path.join(results_dir, f"{args.model}_feature_ablation.txt"), "w") as fh:
        fh.write(ablation_txt + "\n")
    save_json(
        {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ablation_results.items()},
        os.path.join(results_dir, f"{args.model}_feature_ablation.json"),
    )

    if args.shap:
        import pickle
        xgb_path = os.path.join(results_dir, f"{args.model}_xgb.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                xgb_clf = pickle.load(f)

            sample_feat = next(
                (x for x in all_feats if x.get("label") in (0, 1)), None
            )
            if sample_feat:
                num_layers = len(sample_feat["ads_per_layer"])
                print(f"\n[SHAP] Computing importance for {args.model} (XGB) …")
                shap_vals = compute_shap_importance(
                    xgb_clf, test_feats, num_layers=num_layers
                )
                if shap_vals:
                    save_json(
                        {k: v.tolist() for k, v in shap_vals.items()},
                        os.path.join(results_dir, f"{args.model}_shap.json"),
                    )
                    print("[SHAP] Saved.")

    print("\n" + "=" * 60)
    print("  Baseline comparison")
    print("=" * 60)
    import subprocess
    baseline_cmd = [
        sys.executable, "scripts/eval_baselines.py",
        "--features",   feat_path,
        "--splits",     splits_path,
        "--output-dir", results_dir,
        "--model-name", args.model,
    ]
    try:
        subprocess.run(baseline_cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [WARN] Baselines failed: {e}")
        print("  (This is expected if features.pkl lacks token_logits — "
              "re-run step2 to populate them.)")

    print(f"\n[Train] All done. Results in {results_dir}")


def _print_metrics(m: dict) -> None:
    print(
        f"  PR={m.get('precision', 0):.3f}  "
        f"RC={m.get('recall', 0):.3f}  "
        f"F1={m.get('f1', 0):.3f}  "
        f"ACC={m.get('accuracy', 0):.3f}  "
        f"AUC={m.get('auc', 0):.3f}"
    )


if __name__ == "__main__":
    main()
