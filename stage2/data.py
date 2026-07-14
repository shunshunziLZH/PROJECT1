from __future__ import annotations

import random
import warnings
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
FIELDS: Tuple[str, ...] = ("I", "J", "T", "B")


def _field_paths(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing required dataset folder: {folder}")
    result: Dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise RuntimeError(f"Duplicate image id {path.stem!r} in {folder}")
            result[path.stem] = path
    return result


def discover_aligned_samples(root: Union[str, Path]) -> List[Dict[str, Path]]:
    root = Path(root)
    by_field = {field: _field_paths(root / field) for field in FIELDS}
    all_ids = set().union(*(paths.keys() for paths in by_field.values()))
    missing: Dict[str, List[str]] = {}
    for field, paths in by_field.items():
        absent = sorted(all_ids - paths.keys())
        if absent:
            missing[field] = absent
    if missing:
        summary = "; ".join(
            f"{field}: {len(ids)} missing (e.g. {', '.join(ids[:3])})" for field, ids in missing.items()
        )
        raise RuntimeError(f"Unpaired I/J/T/B dataset under {root}: {summary}")
    if not all_ids:
        raise RuntimeError(f"No aligned I/J/T/B samples found under {root}")
    return [{"sample_id": image_id, **{field: by_field[field][image_id] for field in FIELDS}} for image_id in sorted(all_ids)]


def split_samples(
    samples: Sequence[Mapping[str, Path]], split: str, val_ratio: float
) -> List[Mapping[str, Path]]:
    if split not in {"train", "val", "all"}:
        raise ValueError("split must be one of: train, val, all")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    samples = list(samples)
    if split == "all":
        return samples
    if val_ratio == 0 or len(samples) <= 1:
        return samples if split == "train" else []
    val_count = max(1, int(round(len(samples) * val_ratio)))
    train_count = len(samples) - val_count
    return samples[:train_count] if split == "train" else samples[train_count:]


def load_rgb(path: Union[str, Path]) -> torch.Tensor:
    with Image.open(path) as image:
        source = np.asarray(image)
        if source.dtype != np.uint8:
            raise TypeError(f"Stage 2 currently requires 8-bit image files, got {source.dtype}: {path}")
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0)


def physical_reconstruction_error(I: torch.Tensor, J: torch.Tensor, T: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (I - (J * T + B)).abs().mean()


def validate_aligned_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    range_tolerance: float = 1.0e-4,
    physical_error_warn: Optional[float] = 0.10,
    physical_error_fail: Optional[float] = None,
    sample_id: str = "unknown",
) -> float:
    missing = [field for field in FIELDS if field not in tensors]
    if missing:
        raise KeyError(f"Sample {sample_id} is missing fields: {missing}")
    shapes = {field: tuple(tensors[field].shape) for field in FIELDS}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Sample {sample_id} has inconsistent shapes: {shapes}")
    shape = next(iter(shapes.values()))
    if len(shape) != 3 or shape[0] != 3:
        raise ValueError(f"Sample {sample_id} must use RGB CHW tensors, got {shape}")
    for field in FIELDS:
        tensor = tensors[field]
        if not tensor.is_floating_point():
            raise TypeError(f"Sample {sample_id} field {field} is not floating point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Sample {sample_id} field {field} contains NaN/Inf")
        low, high = float(tensor.min()), float(tensor.max())
        if low < -range_tolerance or high > 1.0 + range_tolerance:
            raise ValueError(
                f"Sample {sample_id} field {field} is outside [0,1]: min={low:.6g}, max={high:.6g}"
            )
    error = float(physical_reconstruction_error(*(tensors[field] for field in FIELDS)))
    if physical_error_fail is not None and error > physical_error_fail:
        raise ValueError(
            f"Sample {sample_id} physical reconstruction error {error:.6f} exceeds failure threshold "
            f"{physical_error_fail:.6f}"
        )
    if physical_error_warn is not None and error > physical_error_warn:
        warnings.warn(
            f"Sample {sample_id} has high physical reconstruction error: {error:.6f} > {physical_error_warn:.6f}",
            RuntimeWarning,
            stacklevel=2,
        )
    return error


def _pad(tensors: Sequence[torch.Tensor], patch_size: int) -> List[torch.Tensor]:
    import torch.nn.functional as F

    height, width = tensors[0].shape[-2:]
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if pad_h == 0 and pad_w == 0:
        return list(tensors)
    return [F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate") for tensor in tensors]


class AlignedBankDataset(Dataset):
    """Aligned I/J/T/B patches for Stage 2.

    A logical image is repeated ``patches_per_image`` times, so each epoch (or
    deterministic extraction pass) observes multiple degradation regions.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: str = "train",
        val_ratio: float = 0.1,
        patch_size: int = 64,
        patches_per_image: int = 8,
        use_hflip: bool = True,
        use_vflip: bool = True,
        use_rot90: bool = True,
        validate_tensors: bool = True,
        range_tolerance: float = 1.0e-4,
        physical_error_warn: Optional[float] = 0.10,
        physical_error_fail: Optional[float] = None,
        deterministic: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        cache_data: str = "none",
        decode_cache_size: int = 2,
        ram_cache_max_gib: float = 4.0,
        physical_warning_limit: int = 5,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive")
        if cache_data not in {"none", "ram"}:
            raise ValueError("cache_data must be 'none' or 'ram'")
        if decode_cache_size < 0 or physical_warning_limit < 0:
            raise ValueError("decode_cache_size and physical_warning_limit cannot be negative")
        self.root = Path(root)
        selected = split_samples(discover_aligned_samples(self.root), split, val_ratio)
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            raise RuntimeError(f"No samples selected for split={split}")
        self.samples = selected
        self.split = split
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.use_hflip = use_hflip and not deterministic
        self.use_vflip = use_vflip and not deterministic
        self.use_rot90 = use_rot90 and not deterministic
        self.validate_tensors = validate_tensors
        self.range_tolerance = range_tolerance
        self.physical_error_warn = physical_error_warn
        self.physical_error_fail = physical_error_fail
        self.physical_warning_limit = int(physical_warning_limit)
        self._physical_warning_count = 0
        self.deterministic = deterministic
        self.seed = int(seed)
        self.cache: Optional[List[Dict[str, torch.Tensor]]] = None
        self.decode_cache_size = int(decode_cache_size)
        self._decode_cache: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()
        if cache_data == "ram":
            first = self._load_sample(self.samples[0])
            bytes_per_sample = sum(tensor.numel() * tensor.element_size() for tensor in first.values())
            estimated_gib = bytes_per_sample * len(self.samples) / (1024**3)
            if estimated_gib > float(ram_cache_max_gib):
                raise MemoryError(
                    f"RAM cache would require approximately {estimated_gib:.1f} GiB, exceeding "
                    f"data.ram_cache_max_gib={ram_cache_max_gib}. Use cache_data=none; the grouped "
                    "sampler and bounded decode cache avoid repeated patch decoding."
                )
            self.cache = [self._load_sample(sample) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _load_sample(self, sample: Mapping[str, Path]) -> Dict[str, torch.Tensor]:
        tensors = {field: load_rgb(sample[field]) for field in FIELDS}
        shapes = {field: tuple(tensor.shape) for field, tensor in tensors.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Sample {sample['sample_id']} has inconsistent source sizes: {shapes}")
        return tensors

    def _get_loaded(self, image_index: int) -> Dict[str, torch.Tensor]:
        if self.cache is not None:
            return self.cache[image_index]
        cached = self._decode_cache.pop(image_index, None)
        if cached is not None:
            self._decode_cache[image_index] = cached
            return cached
        loaded = self._load_sample(self.samples[image_index])
        if self.decode_cache_size > 0:
            self._decode_cache[image_index] = loaded
            while len(self._decode_cache) > self.decode_cache_size:
                self._decode_cache.popitem(last=False)
        return loaded

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        image_index, patch_index = divmod(index, self.patches_per_image)
        sample = self.samples[image_index]
        loaded = self._get_loaded(image_index)
        tensors = [loaded[field].clone() for field in FIELDS]
        original_height, original_width = tensors[0].shape[-2:]
        tensors = _pad(tensors, self.patch_size)
        height, width = tensors[0].shape[-2:]
        rng = random.Random(self.seed + index * 1_000_003) if self.deterministic else random
        top = rng.randint(0, height - self.patch_size)
        left = rng.randint(0, width - self.patch_size)
        tensors = [tensor[:, top : top + self.patch_size, left : left + self.patch_size] for tensor in tensors]

        hflip = bool(self.use_hflip and rng.random() < 0.5)
        vflip = bool(self.use_vflip and rng.random() < 0.5)
        rotation = rng.randint(0, 3) if self.use_rot90 else 0
        if hflip:
            tensors = [tensor.flip(-1) for tensor in tensors]
        if vflip:
            tensors = [tensor.flip(-2) for tensor in tensors]
        if rotation:
            tensors = [torch.rot90(tensor, rotation, (-2, -1)) for tensor in tensors]

        result_tensors = dict(zip(FIELDS, tensors))
        physical_error = (
            validate_aligned_tensors(
                result_tensors,
                range_tolerance=self.range_tolerance,
                physical_error_warn=None,
                physical_error_fail=self.physical_error_fail,
                sample_id=str(sample["sample_id"]),
            )
            if self.validate_tensors
            else float(physical_reconstruction_error(*tensors))
        )
        if self.physical_error_warn is not None and physical_error > float(self.physical_error_warn):
            self._physical_warning_count += 1
            if self._physical_warning_count <= self.physical_warning_limit:
                suffix = (
                    " Further warnings from this dataset worker are suppressed."
                    if self._physical_warning_count == self.physical_warning_limit
                    else ""
                )
                warnings.warn(
                    f"Sample {sample['sample_id']} has high physical reconstruction error "
                    f"{physical_error:.6f} > {float(self.physical_error_warn):.6f}.{suffix}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return {
            **result_tensors,
            "sample_id": str(sample["sample_id"]),
            "image_path": str(sample["I"]),
            "patch_index": patch_index,
            "crop_top": top,
            "crop_left": left,
            "original_height": original_height,
            "original_width": original_width,
            "hflip": hflip,
            "vflip": vflip,
            "rotation_k": rotation,
            "physical_error": physical_error,
            "mean_T": float(result_tensors["T"].mean()),
            "mean_B": float(result_tensors["B"].mean()),
        }


def dataset_from_config(config: Mapping[str, object], split: str, *, extraction: bool = False) -> AlignedBankDataset:
    data = config["data"]
    assert isinstance(data, Mapping)
    extraction_config = config.get("extraction", {})
    assert isinstance(extraction_config, Mapping)
    patches = extraction_config.get("patches_per_image", data["patches_per_image"]) if extraction else data["patches_per_image"]
    limit = data.get("train_limit") if split == "train" else data.get("val_limit")
    return AlignedBankDataset(
        data["root"],
        split=split,
        val_ratio=float(data["val_ratio"]),
        patch_size=int(data["patch_size"]),
        patches_per_image=int(patches),
        use_hflip=bool(data["use_hflip"]),
        use_vflip=bool(data["use_vflip"]),
        use_rot90=bool(data["use_rot90"]),
        validate_tensors=bool(data["validate_tensors"]),
        range_tolerance=float(data["range_tolerance"]),
        physical_error_warn=data.get("physical_error_warn"),
        physical_error_fail=data.get("physical_error_fail"),
        deterministic=extraction or split != "train",
        seed=int(config["seed"]),
        limit=None if limit is None else int(limit),
        cache_data=str(data["cache_data"]),
        decode_cache_size=int(data["decode_cache_size"]),
        ram_cache_max_gib=float(data["ram_cache_max_gib"]),
        physical_warning_limit=int(data["physical_warning_limit"]),
    )


def build_data_manifest(config: Mapping[str, object]) -> Dict[str, object]:
    """Record disjoint split IDs and a lightweight file fingerprint for provenance."""

    data = config["data"]
    assert isinstance(data, Mapping)
    samples = discover_aligned_samples(data["root"])
    train_samples = split_samples(samples, "train", float(data["val_ratio"]))
    val_samples = split_samples(samples, "val", float(data["val_ratio"]))
    train_limit, val_limit = data.get("train_limit"), data.get("val_limit")
    if train_limit is not None:
        train_samples = train_samples[: int(train_limit)]
    if val_limit is not None:
        val_samples = val_samples[: int(val_limit)]
    train_ids = [str(sample["sample_id"]) for sample in train_samples]
    val_ids = [str(sample["sample_id"]) for sample in val_samples]
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise RuntimeError(f"Train/validation split overlap detected: {overlap[:5]}")

    fingerprint_mode = str(data.get("fingerprint_mode", "content"))
    if fingerprint_mode not in {"content", "stat"}:
        raise ValueError("data.fingerprint_mode must be 'content' or 'stat'")

    def fingerprint(selected: Sequence[Mapping[str, Path]]) -> str:
        digest = hashlib.sha256()
        for sample in selected:
            digest.update(str(sample["sample_id"]).encode("utf-8"))
            for field in FIELDS:
                path = Path(sample[field])
                stat = path.stat()
                digest.update(field.encode("ascii"))
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                if fingerprint_mode == "content":
                    with path.open("rb") as handle:
                        while True:
                            chunk = handle.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                else:
                    digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()

    return {
        "format_version": 1,
        "fingerprint_mode": fingerprint_mode,
        "all_sample_count": len(samples),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_fingerprint": fingerprint(train_samples),
        "val_fingerprint": fingerprint(val_samples),
        "training_split_only": True,
    }
