from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_ids(folder: Path) -> Set[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return {path.stem for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def paired_ids(root: Union[str, Path]) -> List[str]:
    root = Path(root)
    ids = _list_ids(root / "I") & _list_ids(root / "T") & _list_ids(root / "B")
    if not ids:
        raise RuntimeError(f"No paired I/T/B images found under {root}")
    return sorted(ids)


def image_path(root: Path, subset: str, image_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / subset / f"{image_id}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {subset} image for id {image_id} under {root}")


def load_rgb_uint8(path: Union[str, Path]) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8, copy=True)


def load_rgb(path: Union[str, Path]) -> np.ndarray:
    return load_rgb_uint8(path).astype(np.float32) / 255.0


def save_rgb(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().cpu().clamp(0, 1).numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))
    image.save(path)


def _pad_to_min_size(images: List[np.ndarray], min_size: int) -> List[np.ndarray]:
    height, width = images[0].shape[:2]
    pad_h = max(0, min_size - height)
    pad_w = max(0, min_size - width)
    if pad_h == 0 and pad_w == 0:
        return images
    pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
    return [np.pad(image, pad_spec, mode="edge") for image in images]


def _sync_crop(images: List[np.ndarray], patch_size: int) -> List[np.ndarray]:
    images = _pad_to_min_size(images, patch_size)
    height, width = images[0].shape[:2]
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    return [image[top : top + patch_size, left : left + patch_size, :] for image in images]


def _sync_augment(images: List[np.ndarray], use_flip: bool, use_rot: bool) -> List[np.ndarray]:
    if use_flip and random.random() < 0.5:
        images = [np.flip(image, axis=1) for image in images]
    if use_rot:
        k = random.randint(0, 3)
        if k:
            images = [np.rot90(image, k, axes=(0, 1)) for image in images]
    return images


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    return tensor.float()


class TBDataset(Dataset):
    """File-name paired dataset for I -> T/B prediction."""

    def __init__(
        self,
        root: Union[str, Path] = "basicsr/data/DATA",
        split: str = "train",
        val_ratio: float = 0.1,
        patch_size: Union[int, None] = 512,
        use_flip: bool = True,
        use_rot: bool = True,
        limit: Union[int, None] = None,
        cache_data: str = "none",
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.cache_data = cache_data
        self.cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        if cache_data not in {"none", "ram"}:
            raise ValueError("cache_data must be one of: none, ram")

        ids = paired_ids(self.root)
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")

        if split != "all" and len(ids) > 1 and val_ratio > 0:
            val_count = max(1, int(round(len(ids) * val_ratio)))
            train_count = len(ids) - val_count
            ids = ids[:train_count] if split == "train" else ids[train_count:]

        if limit is not None:
            ids = ids[:limit]
        if not ids:
            raise RuntimeError(f"No images selected for split={split}")
        self.ids = ids
        if self.cache_data == "ram":
            self.cache = self._build_ram_cache()

    def _build_ram_cache(self) -> Dict[str, Dict[str, np.ndarray]]:
        cache: Dict[str, Dict[str, np.ndarray]] = {"I": {}, "T": {}, "B": {}}
        print(f"Building RAM cache for split={self.split}: {len(self.ids)} ids")
        for index, image_id in enumerate(self.ids, start=1):
            for subset in cache:
                cache[subset][image_id] = load_rgb_uint8(image_path(self.root, subset, image_id))
            if index % 1000 == 0 or index == len(self.ids):
                print(f"  cached {index}/{len(self.ids)} ids for split={self.split}")
        return cache

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id = self.ids[index]
        i_path = image_path(self.root, "I", image_id)
        t_path = image_path(self.root, "T", image_id)
        b_path = image_path(self.root, "B", image_id)

        if self.cache is not None:
            images = [self.cache[subset][image_id] for subset in ("I", "T", "B")]
        else:
            images = [load_rgb_uint8(i_path), load_rgb_uint8(t_path), load_rgb_uint8(b_path)]
        if self.split == "train":
            if self.patch_size is not None:
                images = _sync_crop(images, self.patch_size)
            images = _sync_augment(images, self.use_flip, self.use_rot)

        image_i, image_t, image_b = [_to_tensor(image) for image in images]
        return {
            "I": image_i,
            "T": image_t,
            "B": image_b,
            "id": image_id,
            "I_path": str(i_path),
            "T_path": str(t_path),
            "B_path": str(b_path),
        }


def describe_dataset(root: Union[str, Path] = "basicsr/data/DATA", val_ratio: float = 0.1) -> Dict[str, Union[int, str]]:
    ids = paired_ids(root)
    val_count = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 and val_ratio > 0 else 0
    return {
        "root": str(Path(root)),
        "paired_count": len(ids),
        "train_count": len(ids) - val_count,
        "val_count": val_count,
        "first_id": ids[0],
        "last_id": ids[-1],
    }
