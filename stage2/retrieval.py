from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .utils import load_checkpoint


PathLike = Union[str, os.PathLike]
RetrievalOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _validate_inputs(
    query: torch.Tensor,
    bank_keys: torch.Tensor,
    bank_values: torch.Tensor,
    top_k: int,
    temperature: float,
) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not isinstance(bank_keys, torch.Tensor) or not isinstance(bank_values, torch.Tensor):
        raise TypeError("bank_keys and bank_values must be torch.Tensor instances")
    if query.ndim not in (2, 3):
        raise ValueError(f"query must have shape [B,D] or [B,N,D], got {tuple(query.shape)}")
    if bank_keys.ndim != 2 or bank_values.ndim != 2:
        raise ValueError(
            f"bank_keys and bank_values must be rank 2, got {bank_keys.shape} and {bank_values.shape}"
        )
    if bank_keys.shape[0] == 0:
        raise ValueError("The bank must contain at least one prototype")
    if bank_keys.shape[0] != bank_values.shape[0]:
        raise ValueError(
            f"Bank key/value counts differ: {bank_keys.shape[0]} and {bank_values.shape[0]}"
        )
    if query.shape[-1] != bank_keys.shape[1]:
        raise ValueError(
            f"Query dimension {query.shape[-1]} does not match bank key dimension {bank_keys.shape[1]}"
        )
    if bank_values.shape[1] == 0 or query.shape[-1] == 0:
        raise ValueError("Key and value dimensions must be non-zero")
    for name, tensor in (("query", query), ("bank_keys", bank_keys), ("bank_values", bank_values)):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if query.device != bank_keys.device or query.device != bank_values.device:
        raise ValueError(
            f"query, bank_keys, and bank_values must be on the same device, got "
            f"{query.device}, {bank_keys.device}, {bank_values.device}"
        )
    if isinstance(top_k, bool) or int(top_k) != top_k or not 1 <= int(top_k) <= bank_keys.shape[0]:
        raise ValueError(f"top_k must be in [1,{bank_keys.shape[0]}], got {top_k!r}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")


def retrieve(
    query: torch.Tensor,
    bank_keys: torch.Tensor,
    bank_values: torch.Tensor,
    top_k: int = 4,
    temperature: Optional[float] = None,
    *,
    temp: Optional[float] = None,
    epsilon: float = 1.0e-12,
) -> RetrievalOutput:
    """Soft top-k cosine retrieval for global or token-wise query keys.

    Args:
        query: ``[B,D]`` global keys or ``[B,N,D]`` token keys.
        bank_keys: ``[M,D]`` key prototypes.
        bank_values: ``[M,V]`` value prototypes.

    Returns:
        ``(retrieved_value, top_indices, top_weights, top_similarities)``.
        The leading output dimensions match ``query`` (excluding key dim), and
        the last dimensions are respectively ``V``, ``top_k``, ``top_k``, and
        ``top_k``.
    """
    if temperature is not None and temp is not None:
        raise ValueError("Pass only one of temperature or temp")
    resolved_temperature = 0.1 if temperature is None and temp is None else (
        temperature if temperature is not None else temp
    )
    if resolved_temperature is None:  # Kept explicit for static/runtime defensive checks.
        raise RuntimeError("Failed to resolve retrieval temperature")
    _validate_inputs(query, bank_keys, bank_values, top_k, float(resolved_temperature))
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")

    # Promote mixed floating dtypes instead of silently reducing bank precision.
    compute_dtype = torch.promote_types(query.dtype, bank_keys.dtype)
    compute_dtype = torch.promote_types(compute_dtype, bank_values.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16) and query.device.type == "cpu":
        compute_dtype = torch.float32
    flat_query = query.reshape(-1, query.shape[-1]).to(dtype=compute_dtype)
    keys = bank_keys.to(dtype=compute_dtype)
    values = bank_values.to(dtype=compute_dtype)
    query_norms = flat_query.norm(dim=-1)
    key_norms = keys.norm(dim=-1)
    if (query_norms <= epsilon).any():
        count = int((query_norms <= epsilon).sum())
        raise ValueError(f"query contains {count} zero-length key vectors")
    if (key_norms <= epsilon).any():
        indices = torch.nonzero(key_norms <= epsilon, as_tuple=False).flatten().tolist()
        raise ValueError(f"bank_keys contains zero-length vectors at indices {indices[:8]}")

    normalized_query = F.normalize(flat_query, dim=-1, eps=float(epsilon))
    normalized_keys = F.normalize(keys, dim=-1, eps=float(epsilon))
    similarities = normalized_query.matmul(normalized_keys.t())
    top_similarities, top_indices = torch.topk(
        similarities, k=int(top_k), dim=-1, largest=True, sorted=True
    )
    top_weights = torch.softmax(top_similarities / float(resolved_temperature), dim=-1)
    selected_values = values[top_indices]
    retrieved = (selected_values * top_weights.unsqueeze(-1)).sum(dim=-2)

    leading_shape = tuple(query.shape[:-1])
    retrieved = retrieved.reshape(*leading_shape, bank_values.shape[1])
    top_indices = top_indices.reshape(*leading_shape, int(top_k))
    top_weights = top_weights.reshape(*leading_shape, int(top_k))
    top_similarities = top_similarities.reshape(*leading_shape, int(top_k))
    return retrieved, top_indices, top_weights, top_similarities


retrieve_from_bank = retrieve
soft_retrieve = retrieve


class BankRetriever(nn.Module):
    """Reusable retrieval module whose bank entries follow module devices."""

    def __init__(
        self,
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
        top_k: int = 4,
        temperature: float = 0.1,
        *,
        persistent: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(bank_keys, torch.Tensor) or not isinstance(bank_values, torch.Tensor):
            raise TypeError("bank_keys and bank_values must be torch.Tensor instances")
        # A one-query validation catches all bank-level structural errors.
        if bank_keys.ndim != 2 or bank_keys.shape[0] == 0 or bank_keys.shape[1] == 0:
            raise ValueError(f"bank_keys must have non-empty shape [M,D], got {bank_keys.shape}")
        dummy = bank_keys[:1]
        _validate_inputs(dummy, bank_keys, bank_values, top_k, temperature)
        if (bank_keys.norm(dim=-1) <= 1.0e-12).any():
            raise ValueError("bank_keys contains a zero-length prototype")
        self.register_buffer("bank_keys", bank_keys.detach().clone(), persistent=persistent)
        self.register_buffer("bank_values", bank_values.detach().clone(), persistent=persistent)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[PathLike, Mapping[str, Any]],
        *,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        map_location: Any = "cpu",
        persistent: bool = True,
    ) -> "BankRetriever":
        if isinstance(checkpoint, Mapping):
            payload = checkpoint
        else:
            payload = load_checkpoint(Path(checkpoint), map_location=map_location)
        if not isinstance(payload, Mapping):
            raise TypeError("Bank checkpoint root must be a mapping")
        missing = [name for name in ("keys", "values") if name not in payload]
        if missing:
            raise KeyError(f"Bank checkpoint is missing fields: {missing}")
        configuration = payload.get("configuration", {})
        if configuration is not None and not isinstance(configuration, Mapping):
            raise TypeError("Bank checkpoint configuration must be a mapping")
        configuration = configuration or {}
        resolved_top_k = int(
            top_k if top_k is not None else configuration.get("top_k", 4)
        )
        resolved_temperature = float(
            temperature
            if temperature is not None
            else configuration.get("retrieval_temperature", 0.1)
        )
        return cls(
            torch.as_tensor(payload["keys"]),
            torch.as_tensor(payload["values"]),
            top_k=resolved_top_k,
            temperature=resolved_temperature,
            persistent=persistent,
        )

    def forward(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> RetrievalOutput:
        return retrieve(
            query,
            self.bank_keys,
            self.bank_values,
            top_k=self.top_k if top_k is None else top_k,
            temperature=self.temperature if temperature is None else temperature,
        )

    def extra_repr(self) -> str:
        return (
            f"prototypes={self.bank_keys.shape[0]}, key_dim={self.bank_keys.shape[1]}, "
            f"value_dim={self.bank_values.shape[1]}, top_k={self.top_k}, "
            f"temperature={self.temperature:g}"
        )


NeuralBankRetriever = BankRetriever


__all__ = [
    "RetrievalOutput",
    "retrieve",
    "retrieve_from_bank",
    "soft_retrieve",
    "BankRetriever",
    "NeuralBankRetriever",
]
