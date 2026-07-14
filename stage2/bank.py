from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .utils import atomic_torch_save, load_checkpoint


ArrayLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, os.PathLike]


def _as_float32_matrix(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N,D], got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Metadata/configuration cannot contain NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON/YAML serializable")


def _metadata_rows(metadata: Any, count: int) -> List[Dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(count)]
    if isinstance(metadata, Mapping):
        if count == 1:
            return [dict(_jsonable(metadata))]
        rows: List[Dict[str, Any]] = [{} for _ in range(count)]
        for key, raw_value in metadata.items():
            value = _jsonable(raw_value)
            if isinstance(value, list) and len(value) == count:
                for index, item in enumerate(value):
                    rows[index][str(key)] = item
            else:
                for row in rows:
                    row[str(key)] = value
        return rows
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise TypeError("metadata must be a mapping or a sequence of mappings")
    if len(metadata) != count:
        raise ValueError(f"metadata has {len(metadata)} rows but embeddings have {count}")
    rows = []
    for index, row in enumerate(metadata):
        if not isinstance(row, Mapping):
            raise TypeError(f"metadata row {index} must be a mapping")
        rows.append(dict(_jsonable(row)))
    return rows


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_yaml_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    serializable = _jsonable(payload)
    try:
        import yaml

        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
    except ImportError:
        # JSON is valid YAML 1.2 and keeps extraction usable without PyYAML.
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    os.replace(temporary, path)


def _ensure_training_only(
    configuration: Optional[Mapping[str, Any]], *, require_evidence: bool = False
) -> bool:
    if configuration is None:
        if require_evidence:
            raise ValueError("Training-only extraction evidence is required but configuration is missing")
        return False
    flag = configuration.get("training_split_only")
    if flag is not None and not isinstance(flag, bool):
        raise TypeError("training_split_only must be an explicit boolean")
    flag_is_true = flag is True
    if flag is False:
        raise ValueError("Bank extraction explicitly declares training_split_only=False")
    candidates: List[Any] = [configuration.get("split"), configuration.get("dataset_split")]
    for section_name in ("data", "dataset", "extraction"):
        section = configuration.get(section_name)
        if isinstance(section, Mapping):
            candidates.extend((section.get("split"), section.get("dataset_split")))
            section_flag = section.get("training_split_only")
            if section_flag is not None and not isinstance(section_flag, bool):
                raise TypeError(
                    f"{section_name}.training_split_only must be an explicit boolean"
                )
            if section_flag is False:
                raise ValueError(
                    f"Bank extraction section {section_name!r} declares training_split_only=False"
                )
            flag_is_true = flag_is_true or section_flag is True
    train_split_seen = False
    for split in candidates:
        if split is None:
            continue
        if str(split).lower() not in {"train", "training"}:
            raise ValueError(
                f"Refusing to build a bank from split={split!r}; only the training split is allowed"
            )
        train_split_seen = True
    evidence = flag_is_true and train_split_seen
    if require_evidence and not evidence:
        raise ValueError(
            "Extraction configuration must explicitly set training_split_only=True and split='train' "
            "(or dataset_split='train')"
        )
    return evidence


_FINGERPRINT_MODULES = ("key_encoder", "value_encoder", "key_projector", "value_projector")


def stage2_checkpoint_fingerprint(
    checkpoint: Union[PathLike, Mapping[str, Any]]
) -> str:
    """Hash embedding-space state tensors independent of ``torch.save`` serialization."""
    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        payload = load_checkpoint(Path(checkpoint), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Stage 2 checkpoint root must be a mapping")
    digest = hashlib.sha256()
    digest.update(b"neural-physics-bank-stage2-fingerprint-v1\0")
    for module_name in _FINGERPRINT_MODULES:
        state = _cpu_state_dict(payload.get(module_name), module_name)
        digest.update(module_name.encode("utf-8") + b"\0")
        for parameter_name in sorted(state):
            value = state[parameter_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Checkpoint state {module_name}.{parameter_name} is not a tensor and cannot be fingerprinted"
                )
            tensor = value.detach().cpu().contiguous()
            if tensor.layout != torch.strided:
                raise TypeError(
                    f"Checkpoint state {module_name}.{parameter_name} uses unsupported layout {tensor.layout}"
                )
            digest.update(parameter_name.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
            digest.update(b"\0")
    return digest.hexdigest()


def _verify_embedding_checkpoint(
    extraction_config: Mapping[str, Any],
    stage2_checkpoint: Union[PathLike, Mapping[str, Any]],
    *,
    allow_mismatch: bool,
) -> None:
    recorded = extraction_config.get("stage2_checkpoint_fingerprint")
    problem: Optional[str] = None
    if not isinstance(recorded, str) or len(recorded) != 64:
        problem = "Embedding extraction is missing a valid Stage 2 checkpoint fingerprint"
    else:
        try:
            int(recorded, 16)
        except ValueError as error:
            raise ValueError("Embedding Stage 2 checkpoint fingerprint is not hexadecimal") from error
        actual = stage2_checkpoint_fingerprint(stage2_checkpoint)
        if not hmac.compare_digest(recorded.lower(), actual.lower()):
            problem = (
                "The Stage 2 checkpoint does not match the embedding-space weights used for extraction "
                f"(expected fingerprint={recorded}, got {actual})"
            )
    if problem is not None:
        if not allow_mismatch:
            raise ValueError(problem + "; re-extract embeddings or pass allow_checkpoint_mismatch=True")
        warnings.warn(
            "UNSAFE checkpoint provenance override: " + problem,
            RuntimeWarning,
            stacklevel=3,
        )


class EmbeddingWriter:
    """Incrementally write aligned key/value embeddings in bounded RAM.

    Shards and metadata are durable after each flush. ``manifest.json`` is
    written only by :meth:`close`, so readers never mistake an interrupted
    extraction for a complete one.
    """

    def __init__(
        self,
        output_dir: PathLike,
        key_dim: int,
        value_dim: int,
        chunk_size: int = 4096,
        *,
        overwrite: bool = False,
    ) -> None:
        if int(key_dim) <= 0 or int(value_dim) <= 0:
            raise ValueError("key_dim and value_dim must be positive")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.chunk_size = int(chunk_size)
        self._closed = False
        self._count = 0
        self._buffer_count = 0
        self._key_parts: List[np.ndarray] = []
        self._value_parts: List[np.ndarray] = []
        self._shards: List[Dict[str, Any]] = []

        managed = [
            *self.output_dir.glob("keys_*.npy"),
            *self.output_dir.glob("values_*.npy"),
            self.output_dir / "metadata.jsonl",
            self.output_dir / "manifest.json",
            self.output_dir / "extraction_config.yaml",
        ]
        existing = [path for path in managed if path.exists()]
        if existing and not overwrite:
            names = ", ".join(path.name for path in existing[:5])
            raise FileExistsError(
                f"Embedding output already contains extraction files ({names}); "
                "choose a new directory or pass overwrite=True"
            )
        if overwrite:
            for path in existing:
                if path.is_file():
                    path.unlink()
        self._metadata_handle = (self.output_dir / "metadata.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, keys: ArrayLike, values: ArrayLike, metadata: Any = None) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed EmbeddingWriter")
        key_array = _as_float32_matrix(keys, "keys")
        value_array = _as_float32_matrix(values, "values")
        if key_array.shape[0] != value_array.shape[0]:
            raise ValueError(
                f"keys and values must have the same sample count, got "
                f"{key_array.shape[0]} and {value_array.shape[0]}"
            )
        if key_array.shape[1] != self.key_dim:
            raise ValueError(f"Expected key_dim={self.key_dim}, got {key_array.shape[1]}")
        if value_array.shape[1] != self.value_dim:
            raise ValueError(f"Expected value_dim={self.value_dim}, got {value_array.shape[1]}")
        rows = _metadata_rows(metadata, key_array.shape[0])
        # Validate every metadata row before changing any output state.
        encoded_rows = [
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for row in rows
        ]

        offset = 0
        while offset < key_array.shape[0]:
            take = min(self.chunk_size - self._buffer_count, key_array.shape[0] - offset)
            end = offset + take
            self._key_parts.append(key_array[offset:end].copy())
            self._value_parts.append(value_array[offset:end].copy())
            self._buffer_count += take
            offset = end
            if self._buffer_count == self.chunk_size:
                self._flush()
        for encoded in encoded_rows:
            self._metadata_handle.write(encoded + "\n")
        self._metadata_handle.flush()
        self._count += key_array.shape[0]

    # Common name used by generic streaming writers.
    write = append

    def _flush(self) -> None:
        if self._buffer_count == 0:
            return
        keys = np.concatenate(self._key_parts, axis=0)
        values = np.concatenate(self._value_parts, axis=0)
        if keys.shape[0] != self._buffer_count or values.shape[0] != self._buffer_count:
            raise RuntimeError("Internal embedding buffer count mismatch")
        shard_index = len(self._shards)
        key_name = f"keys_{shard_index:05d}.npy"
        value_name = f"values_{shard_index:05d}.npy"
        key_path, value_path = self.output_dir / key_name, self.output_dir / value_name
        key_temp = key_path.with_name(f".{key_name}.{os.getpid()}.tmp")
        value_temp = value_path.with_name(f".{value_name}.{os.getpid()}.tmp")
        with key_temp.open("wb") as handle:
            np.save(handle, keys, allow_pickle=False)
        with value_temp.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(key_temp, key_path)
        os.replace(value_temp, value_path)
        self._shards.append(
            {"index": shard_index, "keys": key_name, "values": value_name, "count": int(keys.shape[0])}
        )
        self._key_parts.clear()
        self._value_parts.clear()
        self._buffer_count = 0

    def close(self, extraction_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if self._closed:
            return {
                "format_version": 1,
                "count": self._count,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "chunk_size": self.chunk_size,
                "shards": list(self._shards),
            }
        _ensure_training_only(extraction_config, require_evidence=True)
        if self._count == 0:
            self._metadata_handle.close()
            self._closed = True
            raise ValueError("Cannot finalize an empty embedding extraction")
        snapshot: Dict[str, Any] = dict(_jsonable(extraction_config or {}))
        checkpoint_reference = snapshot.get("stage2_checkpoint")
        declared_fingerprint = snapshot.get("stage2_checkpoint_fingerprint")
        if checkpoint_reference is not None:
            computed_fingerprint = stage2_checkpoint_fingerprint(Path(str(checkpoint_reference)))
            if declared_fingerprint is not None and declared_fingerprint != computed_fingerprint:
                raise ValueError(
                    "Declared stage2_checkpoint_fingerprint does not match stage2_checkpoint"
                )
            snapshot["stage2_checkpoint_fingerprint"] = computed_fingerprint
        elif declared_fingerprint is not None:
            if not isinstance(declared_fingerprint, str) or len(declared_fingerprint) != 64:
                raise ValueError(
                    "stage2_checkpoint_fingerprint must be a 64-character SHA-256 hex digest"
                )
            try:
                int(declared_fingerprint, 16)
            except ValueError as error:
                raise ValueError("stage2_checkpoint_fingerprint must be hexadecimal") from error
        self._flush()
        self._metadata_handle.close()
        manifest = {
            "format_version": 1,
            "count": self._count,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dtype": "float32",
            "chunk_size": self.chunk_size,
            "metadata": "metadata.jsonl",
            "extraction_config": "extraction_config.yaml",
            "training_split_only": True,
            "shards": list(self._shards),
        }
        if snapshot.get("stage2_checkpoint_fingerprint") is not None:
            manifest["stage2_checkpoint_fingerprint"] = snapshot["stage2_checkpoint_fingerprint"]
        snapshot["training_split_only"] = True
        snapshot["embedding_count"] = self._count
        snapshot["key_dim"] = self.key_dim
        snapshot["value_dim"] = self.value_dim
        snapshot["chunk_size"] = self.chunk_size
        _write_yaml_atomic(snapshot, self.output_dir / "extraction_config.yaml")
        _write_json_atomic(manifest, self.output_dir / "manifest.json")
        self._closed = True
        return manifest

    finalize = close

    def abort(self) -> None:
        if not self._closed:
            self._metadata_handle.close()
            self._closed = True

    def __enter__(self) -> "EmbeddingWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# Descriptive alias retained for callers that prefer the longer name.
IncrementalEmbeddingWriter = EmbeddingWriter


class EmbeddingReader:
    def __init__(self, input_dir: PathLike) -> None:
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Embedding directory not found: {self.input_dir}")
        manifest_path = self.input_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Embedding extraction is incomplete: missing {manifest_path}. "
                "Finalize EmbeddingWriter before reading."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, Mapping):
            raise TypeError("Embedding manifest root must be a mapping")
        if int(manifest.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported embedding format version: {manifest.get('format_version')}")
        self.manifest = dict(manifest)
        self.count = int(manifest.get("count", -1))
        self.key_dim = int(manifest.get("key_dim", -1))
        self.value_dim = int(manifest.get("value_dim", -1))
        if min(self.count, self.key_dim, self.value_dim) <= 0:
            raise ValueError("Embedding manifest contains invalid count or dimensions")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("Embedding manifest contains no shards")
        self.shards = shards
        declared_count = sum(int(shard.get("count", -1)) for shard in shards if isinstance(shard, Mapping))
        if declared_count != self.count:
            raise ValueError(
                f"Embedding shard counts sum to {declared_count}, manifest declares {self.count}"
            )

    def __len__(self) -> int:
        return self.count

    def iter_shards(self, mmap_mode: Optional[str] = "r") -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        seen = 0
        for index, shard in enumerate(self.shards):
            if not isinstance(shard, Mapping):
                raise TypeError(f"Manifest shard {index} is not a mapping")
            key_path = self.input_dir / str(shard.get("keys", ""))
            value_path = self.input_dir / str(shard.get("values", ""))
            if not key_path.is_file() or not value_path.is_file():
                raise FileNotFoundError(f"Missing embedding shard pair: {key_path}, {value_path}")
            keys = np.load(key_path, mmap_mode=mmap_mode, allow_pickle=False)
            values = np.load(value_path, mmap_mode=mmap_mode, allow_pickle=False)
            expected = int(shard["count"])
            if keys.shape != (expected, self.key_dim):
                raise ValueError(f"Invalid key shard shape in {key_path}: {keys.shape}")
            if values.shape != (expected, self.value_dim):
                raise ValueError(f"Invalid value shard shape in {value_path}: {values.shape}")
            if not np.issubdtype(keys.dtype, np.floating) or not np.issubdtype(values.dtype, np.floating):
                raise TypeError("Embedding shards must be floating point")
            if not np.isfinite(keys).all() or not np.isfinite(values).all():
                raise ValueError(f"Embedding shard {index} contains NaN or Inf")
            seen += expected
            yield keys, values
        if seen != self.count:
            raise RuntimeError(f"Read {seen} embeddings, expected {self.count}")

    def load_metadata(self) -> List[Dict[str, Any]]:
        path = self.input_dir / str(self.manifest.get("metadata", "metadata.jsonl"))
        if not path.is_file():
            raise FileNotFoundError(f"Embedding metadata not found: {path}")
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid metadata JSON on line {line_number}: {error}") from error
                if not isinstance(row, Mapping):
                    raise TypeError(f"Metadata row {line_number} must be a mapping")
                rows.append(dict(row))
        if len(rows) != self.count:
            raise ValueError(f"Metadata has {len(rows)} rows, embeddings have {self.count}")
        return rows

    def load_config(self) -> Dict[str, Any]:
        path = self.input_dir / str(
            self.manifest.get("extraction_config", "extraction_config.yaml")
        )
        if not path.is_file():
            raise FileNotFoundError(f"Extraction configuration not found: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            import yaml

            result = yaml.safe_load(text) or {}
        except ImportError:
            result = json.loads(text)
        if not isinstance(result, Mapping):
            raise TypeError("Extraction configuration root must be a mapping")
        result = dict(result)
        _ensure_training_only(result, require_evidence=True)
        return result

    def load(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        key_parts: List[np.ndarray] = []
        value_parts: List[np.ndarray] = []
        for keys, values in self.iter_shards(mmap_mode="r"):
            key_parts.append(np.asarray(keys, dtype=np.float32))
            value_parts.append(np.asarray(values, dtype=np.float32))
        keys = np.concatenate(key_parts, axis=0)
        values = np.concatenate(value_parts, axis=0)
        return keys, values, self.load_metadata()


def load_embeddings(input_dir: PathLike) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    return EmbeddingReader(input_dir).load()


def load_embedding_bundle(
    input_dir: PathLike,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    reader = EmbeddingReader(input_dir)
    configuration = reader.load_config()
    keys, values, metadata = reader.load()
    return keys, values, metadata, configuration


def _normalize_keys(keys: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    bad = np.flatnonzero(norms[:, 0] <= epsilon)
    if bad.size:
        preview = ", ".join(map(str, bad[:8]))
        raise ValueError(f"Cannot normalize {bad.size} zero-length keys (indices: {preview})")
    normalized = keys / norms
    if not np.isfinite(normalized).all():
        raise ValueError("Key normalization produced NaN or Inf")
    return np.asarray(normalized, dtype=np.float32)


def _fit_sklearn_kmeans(
    keys: np.ndarray,
    num_prototypes: int,
    batch_size: int,
    max_iterations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans

    effective_batch = min(keys.shape[0], max(num_prototypes, int(batch_size)))
    estimator = MiniBatchKMeans(
        n_clusters=num_prototypes,
        batch_size=effective_batch,
        max_iter=max_iterations,
        random_state=seed,
        n_init=3,
        reassignment_ratio=0.01,
        compute_labels=True,
    )
    labels = estimator.fit_predict(keys)
    return np.asarray(labels, dtype=np.int64), np.asarray(estimator.cluster_centers_, dtype=np.float32)


def _squared_distances_to_centers(
    points: torch.Tensor, centers: torch.Tensor, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, points.shape[0], batch_size):
        chunk = points[start : start + batch_size]
        distance = (
            chunk.square().sum(dim=1, keepdim=True)
            + center_norm
            - 2.0 * chunk.matmul(centers.t())
        ).clamp_min_(0.0)
        minimum, label = distance.min(dim=1)
        labels.append(label)
        distances.append(minimum)
    return torch.cat(labels), torch.cat(distances)


def _torch_kmeans_plus_plus(points: torch.Tensor, clusters: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    first = int(torch.randint(points.shape[0], (1,), generator=generator))
    selected = torch.zeros(points.shape[0], dtype=torch.bool)
    selected[first] = True
    centers = [points[first].clone()]
    minimum = (points - centers[0]).square().sum(dim=1)
    for _ in range(1, clusters):
        weights = minimum.clone()
        weights[selected] = 0
        total = float(weights.sum())
        if total > 0 and math.isfinite(total):
            index = int(torch.multinomial(weights, 1, generator=generator))
        else:
            available = torch.nonzero(~selected, as_tuple=False).flatten()
            if available.numel() == 0:
                raise RuntimeError("K-means++ ran out of distinct sample indices")
            choice = int(torch.randint(available.numel(), (1,), generator=generator))
            index = int(available[choice])
        selected[index] = True
        center = points[index].clone()
        centers.append(center)
        minimum = torch.minimum(minimum, (points - center).square().sum(dim=1))
    return torch.stack(centers)


def _fit_torch_kmeans(
    keys: np.ndarray,
    num_prototypes: int,
    batch_size: int,
    max_iterations: int,
    seed: int,
    tolerance: float = 1.0e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    points = torch.from_numpy(np.ascontiguousarray(keys))
    centers = _torch_kmeans_plus_plus(points, num_prototypes, seed)
    batch_size = max(1, int(batch_size))
    for _ in range(max_iterations):
        labels, distances = _squared_distances_to_centers(points, centers, batch_size)
        counts = torch.bincount(labels, minlength=num_prototypes)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, points)
        updated = centers.clone()
        nonempty = counts > 0
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            # Seed empty centers with high-error samples from non-singleton clusters.
            order = torch.argsort(distances, descending=True, stable=True)
            used = torch.zeros(points.shape[0], dtype=torch.bool)
            mutable_counts = counts.clone()
            for empty in torch.nonzero(~nonempty, as_tuple=False).flatten().tolist():
                candidate = None
                for index in order.tolist():
                    donor = int(labels[index])
                    if not used[index] and int(mutable_counts[donor]) > 1:
                        candidate = index
                        mutable_counts[donor] -= 1
                        used[index] = True
                        break
                if candidate is None:
                    raise RuntimeError("Unable to re-seed an empty K-means cluster")
                updated[empty] = points[candidate]
        shift = float((updated - centers).square().sum(dim=1).max().sqrt())
        centers = updated
        if shift <= tolerance:
            break
    labels, _ = _squared_distances_to_centers(points, centers, batch_size)
    return labels.numpy().astype(np.int64, copy=False), centers.numpy().astype(np.float32, copy=False)


def _repair_empty_clusters(
    keys: np.ndarray, labels: np.ndarray, centers: np.ndarray, num_prototypes: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    if labels.shape != (keys.shape[0],):
        raise RuntimeError(f"K-means returned invalid labels shape: {labels.shape}")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= num_prototypes:
        raise RuntimeError("K-means returned out-of-range cluster labels")
    labels = labels.copy()
    counts = np.bincount(labels, minlength=num_prototypes)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        unique_count = int(np.unique(keys, axis=0).shape[0])
        if unique_count < num_prototypes:
            raise ValueError(
                f"K-means requested {num_prototypes} prototypes but normalized keys contain only "
                f"{unique_count} distinct vectors; the embedding space is collapsed"
            )
        warnings.warn(
            f"K-means returned {empty.size} empty clusters; repairing them with deterministic "
            "high-error donor samples.",
            RuntimeWarning,
            stacklevel=3,
        )
        assigned_distances = np.square(keys - centers[labels]).sum(axis=1)
        # mergesort makes equal-distance repair deterministic by original index.
        order = np.argsort(-assigned_distances, kind="mergesort")
        used = np.zeros(keys.shape[0], dtype=bool)
        for cluster in empty:
            candidate = next(
                (
                    int(index)
                    for index in order
                    if not used[index] and counts[labels[index]] > 1
                ),
                None,
            )
            if candidate is None:
                raise RuntimeError(
                    f"K-means produced {empty.size} empty clusters and they could not be repaired"
                )
            donor = labels[candidate]
            counts[donor] -= 1
            labels[candidate] = cluster
            counts[cluster] += 1
            centers[cluster] = keys[candidate]
            used[candidate] = True
    if (counts == 0).any():
        raise RuntimeError(f"K-means still has {int((counts == 0).sum())} empty clusters after repair")
    repaired_centers = np.empty_like(centers)
    for cluster in range(num_prototypes):
        repaired_centers[cluster] = keys[labels == cluster].mean(axis=0)
    return labels, repaired_centers, int(empty.size)


def _aggregate_metadata(
    metadata: Optional[Sequence[Mapping[str, Any]]], retained: Sequence[np.ndarray]
) -> Dict[str, torch.Tensor]:
    if metadata is None:
        return {}
    maximum = max((int(group.max()) for group in retained if group.size), default=-1)
    if maximum >= len(metadata):
        raise ValueError("Metadata is shorter than embedding indices")
    common_keys = set(metadata[0]) if metadata else set()
    for row in metadata[1:]:
        common_keys.intersection_update(row)
    result: Dict[str, torch.Tensor] = {}
    for key in sorted(common_keys):
        try:
            arrays = [np.asarray(row[key], dtype=np.float64) for row in metadata]
        except (TypeError, ValueError):
            continue
        shape = arrays[0].shape
        if any(array.shape != shape or not np.isfinite(array).all() for array in arrays):
            continue
        stacked = np.stack(arrays)
        values = np.stack([stacked[indices].mean(axis=0) for indices in retained]).astype(np.float32)
        result[str(key)] = torch.from_numpy(values)
    return result


def build_bank(
    keys: ArrayLike,
    values: ArrayLike,
    num_prototypes: int = 64,
    *,
    trim_fraction: float = 0.2,
    batch_size: int = 4096,
    max_iterations: int = 100,
    seed: int = 42,
    backend: str = "auto",
    metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    max_iter: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Cluster normalized keys and robustly aggregate their paired values."""
    key_array = _as_float32_matrix(keys, "keys")
    value_array = _as_float32_matrix(values, "values")
    if key_array.shape[0] != value_array.shape[0]:
        raise ValueError("keys and values must contain the same number of samples")
    if isinstance(num_prototypes, bool) or int(num_prototypes) <= 0:
        raise ValueError("num_prototypes must be a positive integer")
    num_prototypes = int(num_prototypes)
    if key_array.shape[0] < num_prototypes:
        raise ValueError(
            f"Cannot build {num_prototypes} prototypes from only {key_array.shape[0]} samples"
        )
    if not 0.0 <= float(trim_fraction) < 1.0:
        raise ValueError("trim_fraction must be in [0,1)")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if max_iter is not None:
        max_iterations = max_iter
    if random_state is not None:
        seed = random_state
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    if backend not in {"auto", "sklearn", "torch"}:
        raise ValueError("backend must be one of: auto, sklearn, torch")
    if metadata is not None and len(metadata) != key_array.shape[0]:
        raise ValueError(f"metadata has {len(metadata)} rows, expected {key_array.shape[0]}")
    if metadata is not None and any(not isinstance(row, Mapping) for row in metadata):
        raise TypeError("Every metadata row must be a mapping")

    normalized = _normalize_keys(key_array)
    selected_backend = backend
    if backend in {"auto", "sklearn"}:
        try:
            labels, centers = _fit_sklearn_kmeans(
                normalized, num_prototypes, int(batch_size), int(max_iterations), int(seed)
            )
            selected_backend = "sklearn_minibatch"
        except ImportError as error:
            if backend == "sklearn":
                raise RuntimeError("scikit-learn is required for backend='sklearn'") from error
            warnings.warn(
                "scikit-learn is unavailable; using the deterministic PyTorch K-means fallback",
                RuntimeWarning,
                stacklevel=2,
            )
            labels, centers = _fit_torch_kmeans(
                normalized, num_prototypes, int(batch_size), int(max_iterations), int(seed)
            )
            selected_backend = "torch"
    else:
        labels, centers = _fit_torch_kmeans(
            normalized, num_prototypes, int(batch_size), int(max_iterations), int(seed)
        )
        selected_backend = "torch"
    labels, centers, repaired_empty = _repair_empty_clusters(
        normalized, labels, centers, num_prototypes
    )

    prototype_keys = np.empty((num_prototypes, normalized.shape[1]), dtype=np.float32)
    prototype_values = np.empty((num_prototypes, value_array.shape[1]), dtype=np.float32)
    counts = np.bincount(labels, minlength=num_prototypes).astype(np.int64)
    variances = np.empty(num_prototypes, dtype=np.float32)
    retained_groups: List[np.ndarray] = []
    for cluster in range(num_prototypes):
        members = np.flatnonzero(labels == cluster)
        if members.size == 0:
            raise RuntimeError(f"Cluster {cluster} is empty after K-means repair")
        distances = np.square(normalized[members] - centers[cluster]).sum(axis=1)
        order = np.argsort(distances, kind="mergesort")
        discard = min(int(math.floor(members.size * float(trim_fraction))), members.size - 1)
        retained_indices = members[order[: members.size - discard]]
        retained_groups.append(retained_indices)
        mean_key = normalized[retained_indices].mean(axis=0)
        norm = float(np.linalg.norm(mean_key))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            nearest = retained_indices[int(np.argmin(distances[order[: members.size - discard]]))]
            mean_key = normalized[nearest]
            norm = float(np.linalg.norm(mean_key))
        prototype = mean_key / norm
        prototype_keys[cluster] = prototype
        prototype_values[cluster] = value_array[retained_indices].mean(axis=0)
        variances[cluster] = np.square(normalized[retained_indices] - prototype).sum(axis=1).mean()

    if (counts <= 0).any():
        raise RuntimeError("Bank construction produced an empty cluster")
    if not np.isfinite(prototype_keys).all() or not np.isfinite(prototype_values).all():
        raise RuntimeError("Bank construction produced NaN or Inf prototypes")
    result: Dict[str, Any] = {
        "keys": torch.from_numpy(prototype_keys),
        "values": torch.from_numpy(prototype_values),
        "cluster_count": torch.from_numpy(counts),
        "cluster_variance": torch.from_numpy(variances),
        "clustering_backend": selected_backend,
        "trim_fraction": float(trim_fraction),
        "repaired_empty_clusters": repaired_empty,
    }
    prototype_metadata = _aggregate_metadata(metadata, retained_groups)
    if prototype_metadata:
        result["prototype_metadata"] = prototype_metadata
    return result


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _infer_projector_dim(state_dict: Mapping[str, Any]) -> Optional[int]:
    candidates = [
        (name, value)
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.ndim == 2 and name.endswith("weight")
    ]
    if not candidates:
        return None
    return int(sorted(candidates, key=lambda item: item[0])[-1][1].shape[0])


def _cpu_state_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise KeyError(f"Stage 2 checkpoint field {name!r} must be a non-empty state dict")
    if not any(isinstance(item, torch.Tensor) for item in value.values()):
        raise TypeError(f"Stage 2 checkpoint state dict {name!r} contains no tensors")
    return {
        str(key): item.detach().cpu() if isinstance(item, torch.Tensor) else item
        for key, item in value.items()
    }


def _model_config_signature(configuration: Mapping[str, Any], label: str) -> Dict[str, int]:
    fields = {
        "patch_size": (("data", "patch_size"), "patch_size"),
        "key_dim": (("model", "key_dim"), "key_dim"),
        "value_dim": (("model", "value_dim"), "value_dim"),
        "projection_dim": (("model", "projection_dim"), "projection_dim"),
        "base_channels": (("model", "base_channels"), "base_channels"),
    }
    result: Dict[str, int] = {}
    for name, (nested_path, flat_name) in fields.items():
        value = _nested_get(configuration, nested_path, configuration.get(flat_name))
        if value is None:
            raise KeyError(f"{label} is missing model-defining field {name!r}")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{label} field {name!r} must be positive")
        result[name] = value
    return result


def _strict_validate_stage2_checkpoint(
    stage2: Mapping[str, Any], checkpoint_config: Mapping[str, Any]
) -> None:
    try:
        from .runtime import build_stage2_model, load_module_state_dicts

        model = build_stage2_model(checkpoint_config)
        load_module_state_dicts(model, stage2, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "Stage 2 checkpoint state dictionaries are incompatible with its embedded config"
        ) from error
    finally:
        if "model" in locals():
            del model


def assemble_bank_checkpoint(
    bank: Mapping[str, Any],
    stage2_checkpoint: Union[PathLike, Mapping[str, Any]],
    configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine prototypes with trained encoders in the required portable format."""
    if isinstance(stage2_checkpoint, Mapping):
        stage2 = dict(stage2_checkpoint)
        checkpoint_name = "<in-memory>"
    else:
        checkpoint_path = Path(stage2_checkpoint)
        stage2 = load_checkpoint(checkpoint_path, map_location="cpu")
        if not isinstance(stage2, Mapping):
            raise TypeError("Stage 2 checkpoint root must be a mapping")
        stage2 = dict(stage2)
        checkpoint_name = str(checkpoint_path)
    required_weights = ("key_encoder", "value_encoder", "key_projector", "value_projector")
    state_dicts = {name: _cpu_state_dict(stage2.get(name), name) for name in required_weights}
    checkpoint_config = stage2.get("config", stage2.get("configuration"))
    if not isinstance(checkpoint_config, Mapping):
        raise KeyError("Stage 2 checkpoint must contain its embedded config mapping")
    checkpoint_signature = _model_config_signature(checkpoint_config, "Stage 2 checkpoint config")
    _strict_validate_stage2_checkpoint(stage2, checkpoint_config)
    for field in ("keys", "values", "cluster_count", "cluster_variance"):
        if field not in bank:
            raise KeyError(f"Prototype bank is missing required field: {field}")
    keys = torch.as_tensor(bank["keys"]).detach().cpu().float()
    values = torch.as_tensor(bank["values"]).detach().cpu().float()
    counts = torch.as_tensor(bank["cluster_count"]).detach().cpu().long()
    variances = torch.as_tensor(bank["cluster_variance"]).detach().cpu().float()
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError("Prototype keys/values must be aligned rank-2 tensors")
    if counts.shape != (keys.shape[0],) or variances.shape != (keys.shape[0],):
        raise ValueError("cluster_count and cluster_variance must have one entry per prototype")

    source_config = configuration if configuration is not None else checkpoint_config
    _ensure_training_only(source_config, require_evidence=True)
    external_signature = _model_config_signature(source_config, "Bank build configuration")
    mismatches = {
        name: (checkpoint_signature[name], external_signature[name])
        for name in checkpoint_signature
        if checkpoint_signature[name] != external_signature[name]
    }
    if mismatches:
        details = ", ".join(
            f"{name}: checkpoint={values_[0]}, build={values_[1]}"
            for name, values_ in mismatches.items()
        )
        raise ValueError(f"Stage 2 checkpoint/build configuration mismatch ({details})")
    if checkpoint_signature["key_dim"] != keys.shape[1]:
        raise ValueError(
            f"Checkpoint key_dim={checkpoint_signature['key_dim']}, prototypes use {keys.shape[1]}"
        )
    if checkpoint_signature["value_dim"] != values.shape[1]:
        raise ValueError(
            f"Checkpoint value_dim={checkpoint_signature['value_dim']}, prototypes use {values.shape[1]}"
        )
    projection_dim = _nested_get(
        source_config, ("model", "projection_dim"), source_config.get("projection_dim")
    )
    if projection_dim is None:
        projection_dim = _infer_projector_dim(state_dicts["key_projector"])
    if projection_dim is None:
        raise ValueError("Unable to determine projection_dim from config or key_projector weights")
    patch_size = checkpoint_signature["patch_size"]
    top_k = _nested_get(source_config, ("retrieval", "top_k"), source_config.get("top_k", 4))
    temperature = _nested_get(
        source_config,
        ("retrieval", "temperature"),
        source_config.get("retrieval_temperature", 0.1),
    )
    if int(top_k) <= 0 or int(top_k) > keys.shape[0]:
        raise ValueError(f"Configured top_k={top_k} is invalid for {keys.shape[0]} prototypes")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("retrieval_temperature must be finite and positive")
    bank_configuration = {
        "patch_size": int(patch_size),
        "key_dim": int(keys.shape[1]),
        "value_dim": int(values.shape[1]),
        "projection_dim": int(projection_dim),
        "num_prototypes": int(keys.shape[0]),
        "top_k": int(top_k),
        "retrieval_temperature": float(temperature),
        "key_input": "[T, B, 1-T]",
        "value_input": "[J, J-I]",
        "stage2_checkpoint": checkpoint_name,
        "stage2_checkpoint_fingerprint": stage2_checkpoint_fingerprint(stage2),
        "training_split_only": True,
        "split": "train",
    }
    payload: Dict[str, Any] = {
        "keys": keys,
        "values": values,
        "cluster_count": counts,
        "cluster_variance": variances,
        "key_encoder": state_dicts["key_encoder"],
        "value_encoder": state_dicts["value_encoder"],
        "key_projector": state_dicts["key_projector"],
        "value_projector": state_dicts["value_projector"],
        "configuration": bank_configuration,
    }
    for optional in (
        "temporary_restorer",
        "prototype_metadata",
        "clustering_backend",
        "trim_fraction",
        "repaired_empty_clusters",
    ):
        if optional in stage2 and optional == "temporary_restorer":
            payload[optional] = _cpu_state_dict(stage2[optional], optional)
        elif optional in bank:
            payload[optional] = bank[optional]
    return payload


def save_bank_checkpoint(
    output_path: PathLike,
    bank: Mapping[str, Any],
    stage2_checkpoint: Union[PathLike, Mapping[str, Any]],
    configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = assemble_bank_checkpoint(bank, stage2_checkpoint, configuration)
    atomic_torch_save(payload, output_path)
    return payload


def build_and_save_bank(
    embeddings_dir: PathLike,
    stage2_checkpoint: Union[PathLike, Mapping[str, Any]],
    output_path: PathLike,
    configuration: Optional[Mapping[str, Any]] = None,
    *,
    allow_checkpoint_mismatch: bool = False,
    ram_warning_gib: float = 2.0,
    **build_options: Any,
) -> Dict[str, Any]:
    reader = EmbeddingReader(embeddings_dir)
    extraction_config = reader.load_config()
    _ensure_training_only(extraction_config, require_evidence=True)
    _verify_embedding_checkpoint(
        extraction_config,
        stage2_checkpoint,
        allow_mismatch=bool(allow_checkpoint_mismatch),
    )
    if not math.isfinite(float(ram_warning_gib)) or float(ram_warning_gib) <= 0:
        raise ValueError("ram_warning_gib must be finite and positive")
    # Current robust trimming needs assignments plus keys/values in memory. Keep
    # the shard iterator public for custom streaming builders and warn before a
    # standard build is likely to surprise the caller.
    estimated_bytes = reader.count * (
        4 * reader.value_dim + 12 * reader.key_dim + 17
    )
    estimated_gib = estimated_bytes / float(1024**3)
    if estimated_gib >= float(ram_warning_gib):
        warnings.warn(
            f"Bank construction is in-memory and is estimated to need at least {estimated_gib:.2f} GiB "
            "plus Python metadata/K-means overhead. Reduce extraction size or implement a multi-pass "
            "consumer over EmbeddingReader.iter_shards() if this exceeds available RAM.",
            RuntimeWarning,
            stacklevel=2,
        )
    keys, values, metadata = reader.load()
    config = configuration if configuration is not None else extraction_config
    _ensure_training_only(config, require_evidence=False)
    bank_config = config.get("bank", {}) if isinstance(config, Mapping) else {}
    if isinstance(bank_config, Mapping):
        option_names = {
            "num_prototypes": int,
            "trim_fraction": float,
            "batch_size": int,
            "max_iterations": int,
            "seed": int,
        }
        for name, converter in option_names.items():
            if name not in build_options and bank_config.get(name) is not None:
                build_options[name] = converter(bank_config[name])
    bank = build_bank(keys, values, metadata=metadata, **build_options)
    trusted_config = dict(config)
    trusted_config["training_split_only"] = True
    trusted_config["split"] = "train"
    trusted_config["dataset_split"] = "train"
    trusted_config["stage2_checkpoint_fingerprint"] = extraction_config[
        "stage2_checkpoint_fingerprint"
    ]
    return save_bank_checkpoint(output_path, bank, stage2_checkpoint, trusted_config)


def _tensor_stats(tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
    values = tensor.detach().cpu().double()
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
    }


def _load_bank(source: Union[PathLike, Mapping[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    if isinstance(source, Mapping):
        return dict(source), False
    payload = load_checkpoint(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Bank checkpoint root must be a mapping")
    return dict(payload), True


def _physical_plot_color(bank: Mapping[str, Any], count: int) -> Tuple[np.ndarray, str]:
    metadata = bank.get("prototype_metadata")
    if not isinstance(metadata, Mapping):
        return np.arange(count), "prototype index"
    for name in ("mean_T", "mean_t", "mean_B", "mean_b"):
        if name in metadata:
            array = torch.as_tensor(metadata[name]).detach().cpu().numpy()
            if array.shape[0] == count:
                if array.ndim > 1:
                    array = array.reshape(count, -1).mean(axis=1)
                return np.asarray(array), name
    for name, channel_a, channel_b, label in (
        ("mean_T_channels", 0, 2, "mean(T_R - T_B)"),
        ("mean_B_channels", 1, 0, "mean(B_G - B_R)"),
    ):
        if name in metadata:
            array = torch.as_tensor(metadata[name]).detach().cpu().numpy()
            if array.ndim == 2 and array.shape[0] == count and array.shape[1] >= 3:
                return array[:, channel_a] - array[:, channel_b], label
    return np.arange(count), "prototype index"


def _write_validation_plots(bank: Mapping[str, Any], output_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is unavailable; bank diagnostic plots were skipped", RuntimeWarning)
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = torch.as_tensor(bank["cluster_count"]).detach().cpu().numpy()
    variances = torch.as_tensor(bank["cluster_variance"]).detach().cpu().numpy()
    keys = torch.as_tensor(bank["keys"]).detach().cpu().float().numpy()
    files: List[str] = []
    for data, filename, xlabel in (
        (counts, "cluster_size_histogram.png", "cluster size"),
        (variances, "cluster_variance_histogram.png", "cluster variance"),
    ):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(data, bins=min(20, max(1, len(data))))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("prototype count")
        figure.tight_layout()
        target = output_dir / filename
        figure.savefig(target, dpi=150)
        plt.close(figure)
        files.append(str(target))
    if keys.shape[0] >= 2:
        coordinates: Optional[np.ndarray] = None
        if keys.shape[0] >= 3:
            try:
                from sklearn.manifold import TSNE

                perplexity = min(30.0, float(max(2, (keys.shape[0] - 1) // 3)))
                coordinates = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    init="pca",
                    learning_rate="auto",
                    random_state=42,
                ).fit_transform(keys)
            except (ImportError, ValueError, TypeError):
                coordinates = None
        if coordinates is None:
            centered = keys - keys.mean(axis=0, keepdims=True)
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            coordinates = centered @ right[: min(2, right.shape[0])].T
            if coordinates.shape[1] == 1:
                coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
        colors, color_label = _physical_plot_color(bank, keys.shape[0])
        figure, axis = plt.subplots(figsize=(6, 5))
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=28, cmap="viridis")
        axis.set_xlabel("embedding component 1")
        axis.set_ylabel("embedding component 2")
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label(color_label)
        figure.tight_layout()
        target = output_dir / "key_tsne_or_umap.png"
        figure.savefig(target, dpi=150)
        plt.close(figure)
        files.append(str(target))
    return files


def validate_bank(
    source: Union[PathLike, Mapping[str, Any]],
    output_dir: Optional[PathLike] = None,
    *,
    strict_checkpoint: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validate a bank and return JSON-serializable coverage statistics.

    File inputs are treated as final checkpoints and therefore require encoder
    state dictionaries and the configuration block. In-memory inputs default
    to prototype-only validation, which is useful immediately after K-means.
    """
    bank, from_file = _load_bank(source)
    if strict_checkpoint is None:
        strict_checkpoint = from_file
    required = ("keys", "values", "cluster_count", "cluster_variance")
    missing = [field for field in required if field not in bank]
    if missing:
        raise KeyError(f"Bank is missing required fields: {missing}")
    if strict_checkpoint:
        checkpoint_fields = (
            "key_encoder",
            "value_encoder",
            "key_projector",
            "value_projector",
            "configuration",
        )
        missing_checkpoint = [field for field in checkpoint_fields if field not in bank]
        if missing_checkpoint:
            raise KeyError(f"Final bank checkpoint is missing fields: {missing_checkpoint}")
        for state_name in ("key_encoder", "value_encoder", "key_projector", "value_projector"):
            _cpu_state_dict(bank[state_name], state_name)
    keys = torch.as_tensor(bank["keys"]).detach().cpu()
    values = torch.as_tensor(bank["values"]).detach().cpu()
    counts = torch.as_tensor(bank["cluster_count"]).detach().cpu()
    variances = torch.as_tensor(bank["cluster_variance"]).detach().cpu()
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError(f"Bank keys/values must be rank 2, got {keys.shape} and {values.shape}")
    if keys.shape[0] == 0 or keys.shape[0] != values.shape[0]:
        raise ValueError("Bank keys and values must have the same non-zero prototype count")
    if counts.shape != (keys.shape[0],) or variances.shape != (keys.shape[0],):
        raise ValueError("cluster_count/cluster_variance shape does not match prototypes")
    if not keys.is_floating_point() or not values.is_floating_point():
        raise TypeError("Bank keys and values must be floating point")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Bank keys or values contain NaN/Inf")
    if not torch.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("cluster_variance must be finite and non-negative")
    if counts.is_floating_point() and not torch.equal(counts, counts.round()):
        raise ValueError("cluster_count must contain integers")
    counts = counts.long()
    if (counts < 0).any():
        raise ValueError("cluster_count cannot be negative")
    key_norms = keys.float().norm(dim=1)
    value_norms = values.float().norm(dim=1)
    empty = int((counts == 0).sum())
    singleton_clusters = int((counts == 1).sum())
    repaired_empty = int(bank.get("repaired_empty_clusters", 0))
    duplicate_pairs = 0
    minimum_cosine_distance: Optional[float] = None
    if keys.shape[0] > 1:
        normalized_keys = keys.float() / key_norms.clamp_min(1.0e-12).unsqueeze(1)
        pairwise_similarity = normalized_keys.matmul(normalized_keys.t()).clamp(-1.0, 1.0)
        upper_mask = torch.triu(
            torch.ones_like(pairwise_similarity, dtype=torch.bool), diagonal=1
        )
        upper_similarity = pairwise_similarity[upper_mask]
        duplicate_pairs = int((upper_similarity >= 1.0 - 1.0e-6).sum())
        minimum_cosine_distance = float((1.0 - upper_similarity.max()).clamp_min(0.0))
    stats: Dict[str, Any] = {
        "number_of_prototypes": int(keys.shape[0]),
        "key_dim": int(keys.shape[1]),
        "value_dim": int(values.shape[1]),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "mean_cluster_size": float(counts.float().mean()),
        "empty_clusters": empty,
        "singleton_clusters": singleton_clusters,
        "singleton_fraction": singleton_clusters / float(keys.shape[0]),
        "repaired_empty_clusters": repaired_empty,
        "duplicate_prototype_pairs": duplicate_pairs,
        "minimum_interprototype_cosine_distance": minimum_cosine_distance,
        **_tensor_stats(variances.float(), "cluster_variance"),
        **_tensor_stats(key_norms, "key_norm"),
        **_tensor_stats(value_norms, "value_norm"),
    }
    issues: List[str] = []
    if empty:
        issues.append(f"Bank contains {empty} empty clusters")
    if repaired_empty < 0:
        raise ValueError("repaired_empty_clusters cannot be negative")
    if duplicate_pairs:
        issues.append(f"Bank contains {duplicate_pairs} duplicate/near-duplicate prototype pairs")
    if keys.shape[0] >= 4 and singleton_clusters / float(keys.shape[0]) > 0.5:
        issues.append(
            f"Bank is dominated by singleton clusters ({singleton_clusters}/{keys.shape[0]})"
        )
    max_norm_error = float((key_norms - 1.0).abs().max())
    if max_norm_error > 1.0e-3:
        issues.append(f"Prototype key norms deviate from 1 by as much as {max_norm_error:.6g}")
    if strict_checkpoint:
        config = bank["configuration"]
        if not isinstance(config, Mapping):
            raise TypeError("Bank configuration must be a mapping")
        required_config = (
            "patch_size",
            "key_dim",
            "value_dim",
            "projection_dim",
            "num_prototypes",
            "top_k",
            "retrieval_temperature",
            "key_input",
            "value_input",
            "stage2_checkpoint",
            "stage2_checkpoint_fingerprint",
            "training_split_only",
            "split",
        )
        missing_config = [field for field in required_config if field not in config]
        if missing_config:
            raise KeyError(f"Bank configuration is missing fields: {missing_config}")
        _ensure_training_only(config, require_evidence=True)
        fingerprint = config["stage2_checkpoint_fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Bank configuration has an invalid Stage 2 checkpoint fingerprint")
        try:
            int(fingerprint, 16)
        except ValueError as error:
            raise ValueError("Stage 2 checkpoint fingerprint is not hexadecimal") from error
        actual_fingerprint = stage2_checkpoint_fingerprint(bank)
        if not hmac.compare_digest(fingerprint.lower(), actual_fingerprint.lower()):
            raise ValueError("Bank encoder/projector weights do not match their recorded fingerprint")
        expected = {
            "num_prototypes": keys.shape[0],
            "key_dim": keys.shape[1],
            "value_dim": values.shape[1],
        }
        for field, actual in expected.items():
            if int(config.get(field, -1)) != int(actual):
                raise ValueError(
                    f"Bank configuration {field}={config.get(field)!r} does not match tensor value {actual}"
                )
        for field in ("patch_size", "projection_dim"):
            if int(config[field]) <= 0:
                raise ValueError(f"Bank configuration {field} must be positive")
        configured_top_k = int(config["top_k"])
        if not 1 <= configured_top_k <= keys.shape[0]:
            raise ValueError(
                f"Bank configuration top_k={configured_top_k} is invalid for {keys.shape[0]} prototypes"
            )
        configured_temperature = float(config["retrieval_temperature"])
        if not math.isfinite(configured_temperature) or configured_temperature <= 0:
            raise ValueError("Bank configuration retrieval_temperature must be finite and positive")
        if config["key_input"] != "[T, B, 1-T]" or config["value_input"] != "[J, J-I]":
            raise ValueError("Bank configuration contains incompatible key_input/value_input definitions")
    stats["valid"] = not issues
    stats["issues"] = issues
    if output_dir is not None:
        destination = Path(output_dir)
        plot_files = _write_validation_plots(bank, destination)
        stats["plot_files"] = plot_files
        destination.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(stats, destination / "bank_validation_report.json")
    return stats


def validate_bank_checkpoint(
    source: Union[PathLike, Mapping[str, Any]], output_dir: Optional[PathLike] = None
) -> Dict[str, Any]:
    """Strict variant for callers that already know they have a final checkpoint."""
    return validate_bank(source, output_dir=output_dir, strict_checkpoint=True)


__all__ = [
    "EmbeddingWriter",
    "IncrementalEmbeddingWriter",
    "EmbeddingReader",
    "load_embeddings",
    "load_embedding_bundle",
    "stage2_checkpoint_fingerprint",
    "build_bank",
    "assemble_bank_checkpoint",
    "save_bank_checkpoint",
    "build_and_save_bank",
    "validate_bank",
    "validate_bank_checkpoint",
]
