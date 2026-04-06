"""YAML config loading helpers."""
from __future__ import annotations
import yaml


def load_config(path: str = "configs/model_configs.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_cfg(config: dict, model_key: str) -> dict:
    cfg = config["models"].get(model_key)
    if cfg is None:
        raise ValueError(
            f"Model '{model_key}' not found in config. "
            f"Available: {list(config['models'].keys())}"
        )
    return cfg


def get_ads_cfg(config: dict) -> dict:
    return config["feature_extraction"]["ads"]


def get_cgc_cfg(config: dict) -> dict:
    return config["feature_extraction"]["cgc"]


def get_dataset_cfg(config: dict) -> dict:
    return config["dataset"]


def get_classifier_cfgs(config: dict) -> dict:
    return config["classifiers"]
