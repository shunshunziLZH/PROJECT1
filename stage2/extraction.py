from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from torch.utils.data import DataLoader

from .bank import EmbeddingWriter
from .data import build_data_manifest, dataset_from_config
from .models import make_physical_key_input, make_value_input
from .runtime import build_stage2_model, load_module_state_dicts
from .utils import load_checkpoint, resolve_device, seed_worker, set_seed


def _metadata_value(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _metadata_rows(batch: Mapping[str, Any], count: int) -> List[Dict[str, Any]]:
    fields = (
        "sample_id",
        "image_path",
        "patch_index",
        "crop_top",
        "crop_left",
        "original_height",
        "original_width",
        "hflip",
        "vflip",
        "rotation_k",
        "physical_error",
        "mean_T",
        "mean_B",
    )
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        row = {field: _metadata_value(batch[field], index) for field in fields if field in batch}
        row["split"] = "train"
        rows.append(row)
    return rows


def extract_embeddings(
    config: Mapping[str, Any],
    checkpoint_path: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Extract trained neural key/value pairs from the training split only."""

    set_seed(int(config["seed"]), deterministic=True)
    device = resolve_device(str(config["device"]))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Stage 2 checkpoint must be a mapping")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise KeyError("Stage 2 checkpoint is missing its configuration snapshot")
    for section, field in (
        ("model", "key_dim"),
        ("model", "value_dim"),
        ("model", "projection_dim"),
        ("model", "base_channels"),
        ("data", "patch_size"),
    ):
        if config[section].get(field) != checkpoint_config.get(section, {}).get(field):
            raise ValueError(f"Extraction config {section}.{field} is incompatible with the checkpoint")
    current_manifest = build_data_manifest(config)
    checkpoint_manifest = checkpoint.get("data_manifest")
    if not isinstance(checkpoint_manifest, Mapping):
        raise KeyError("Stage 2 checkpoint is missing data_manifest; training-only provenance is unverified")
    if checkpoint_manifest.get("format_version") != 1 or checkpoint_manifest.get("training_split_only") is not True:
        raise ValueError("Stage 2 checkpoint data_manifest lacks valid training-only provenance")
    for field in (
        "fingerprint_mode", "train_ids", "val_ids", "train_fingerprint", "val_fingerprint"
    ):
        if current_manifest.get(field) != checkpoint_manifest.get(field):
            raise ValueError(f"Extraction dataset does not match checkpoint data_manifest field {field}")
    model = build_stage2_model(config).to(device)
    load_module_state_dicts(model, checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = dataset_from_config(config, "train", extraction=True)
    extraction = config["extraction"]
    num_workers = int(extraction["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_arguments: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(extraction["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_arguments.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_arguments)

    destination = Path(output_dir or extraction["output_dir"])
    model_config = config["model"]
    writer = EmbeddingWriter(
        destination,
        key_dim=int(model_config["key_dim"]),
        value_dim=int(model_config["value_dim"]),
        chunk_size=int(extraction["chunk_size"]),
        overwrite=overwrite,
    )
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                I = batch["I"].to(device, non_blocking=True)
                J = batch["J"].to(device, non_blocking=True)
                T = batch["T"].to(device, non_blocking=True)
                B = batch["B"].to(device, non_blocking=True)
                _, keys = model.key_encoder(make_physical_key_input(T, B))
                values = model.value_encoder(make_value_input(I, J))
                if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite embedding at extraction batch {batch_index}")
                writer.append(keys, values, _metadata_rows(batch, I.shape[0]))
                if batch_index % 100 == 0 or batch_index == len(loader):
                    print(f"Extracted {writer.count}/{len(dataset)} neural key-value samples")
        extraction_snapshot = {
            **{key: value for key, value in config.items() if not str(key).startswith("_")},
            "split": "train",
            "dataset_split": "train",
            "training_split_only": True,
            "stage2_checkpoint": str(Path(checkpoint_path).resolve()),
        }
        manifest = writer.close(extraction_snapshot)
    except BaseException:
        writer.abort()
        raise
    print(f"Saved training-only embedding shards to {destination}")
    return manifest


__all__ = ["extract_embeddings"]
