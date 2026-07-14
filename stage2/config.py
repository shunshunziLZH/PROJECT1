from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "basicsr/data/DATA",
        "val_ratio": 0.1,
        "patch_size": 64,
        "patches_per_image": 8,
        "use_hflip": True,
        "use_vflip": True,
        "use_rot90": True,
        "cache_data": "none",
        "validate_tensors": True,
        "range_tolerance": 1.0e-4,
        "physical_error_warn": 0.10,
        "physical_error_fail": None,
        "physical_warning_limit": 5,
        "decode_cache_size": 2,
        "ram_cache_max_gib": 4.0,
        "fingerprint_mode": "stat",
        "train_limit": None,
        "val_limit": None,
    },
    "model": {
        "key_dim": 64,
        "value_dim": 128,
        "projection_dim": 64,
        "base_channels": 32,
    },
    "stage1": {
        "checkpoint": None,
        "base_channels": None,
    },
    "loss": {
        "lambda_grad": 0.1,
        "lambda_ssim": 0.2,
        "lambda_fft": 0.05,
        "lambda_restore": 1.0,
        "lambda_key": 0.5,
        "lambda_align": 0.1,
        "lambda_query": 0.2,
        "lambda_inv": 0.1,
        "lambda_rank": 0.0,
        "rank_margin": 0.05,
        "tau_physical": 0.2,
        "tau_embedding": 0.1,
        "physical_relation_size": 8,
    },
    "training": {
        "output_dir": "outputs/bank_stage2",
        "batch_size": 32,
        "epochs": 60,
        "warmup_epochs": 10,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "use_amp": True,
        "num_workers": 8,
        "val_frequency": 1,
        "diagnostic_frequency": 1,
        "log_frequency": 20,
        "resume": None,
    },
    "extraction": {
        "output_dir": "artifacts/bank_embeddings",
        "batch_size": 64,
        "num_workers": 8,
        "chunk_size": 4096,
        "patches_per_image": 8,
    },
    "bank": {
        "output": "artifacts/neural_physics_bank_v0.pt",
        "num_prototypes": 64,
        "trim_fraction": 0.2,
        "max_iterations": 100,
        "batch_size": 4096,
        "seed": 42,
    },
    "retrieval": {
        "top_k": 4,
        "temperature": 0.1,
    },
}


def _deep_merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _validate(config: Mapping[str, Any]) -> None:
    patch_size = int(config["data"]["patch_size"])
    if patch_size <= 0 or patch_size % 8:
        raise ValueError("data.patch_size must be a positive multiple of 8")
    for name in ("key_dim", "value_dim", "projection_dim", "base_channels"):
        if int(config["model"][name]) <= 0:
            raise ValueError(f"model.{name} must be positive")
    if int(config["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive")
    warmup = int(config["training"]["warmup_epochs"])
    if not 0 <= warmup < int(config["training"]["epochs"]):
        raise ValueError("training.warmup_epochs must be in [0, epochs); at least one joint epoch is required")
    if not 0 < float(config["data"]["val_ratio"]) < 1:
        raise ValueError("data.val_ratio must be in (0, 1) to keep train and validation disjoint")
    if int(config["training"]["val_frequency"]) <= 0:
        raise ValueError("training.val_frequency must be positive for best-checkpoint selection")
    if int(config["data"]["decode_cache_size"]) < 0:
        raise ValueError("data.decode_cache_size cannot be negative")
    if config["data"]["fingerprint_mode"] not in {"content", "stat"}:
        raise ValueError("data.fingerprint_mode must be 'content' or 'stat'")
    if float(config["loss"]["lambda_rank"]) > 0:
        if int(config["training"]["batch_size"]) < 2 * int(config["data"]["patches_per_image"]):
            raise ValueError(
                "lambda_rank > 0 requires training.batch_size >= 2 * data.patches_per_image "
                "so cross-image wrong values can be formed"
            )
    if int(config["bank"]["num_prototypes"]) <= 0:
        raise ValueError("bank.num_prototypes must be positive")
    if not 0 <= float(config["bank"]["trim_fraction"]) < 1:
        raise ValueError("bank.trim_fraction must be in [0, 1)")
    if float(config["retrieval"]["temperature"]) <= 0:
        raise ValueError("retrieval.temperature must be positive")


def _reject_unknown_keys(loaded: Mapping[str, Any], defaults: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in loaded.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in defaults:
            raise KeyError(f"Unknown Stage 2 configuration key: {dotted}")
        if isinstance(value, Mapping):
            if not isinstance(defaults[key], Mapping):
                raise TypeError(f"Configuration key {dotted} must not be a mapping")
            _reject_unknown_keys(value, defaults[key], dotted)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage 2 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("Stage 2 YAML root must be a mapping")
    _reject_unknown_keys(loaded, DEFAULT_CONFIG)
    config = deepcopy(DEFAULT_CONFIG)
    _deep_merge(config, loaded)
    config["_config_path"] = str(path.resolve())
    _validate(config)
    return config


def save_config(config: Mapping[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
