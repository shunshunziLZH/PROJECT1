from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import nn


def load_frozen_stage1(config: Mapping[str, object], device: torch.device) -> Optional[nn.Module]:
    stage1 = config.get("stage1", {})
    if not isinstance(stage1, Mapping):
        raise TypeError("stage1 config must be a mapping")
    checkpoint = stage1.get("checkpoint")
    if checkpoint in (None, ""):
        warnings.warn(
            "No Stage 1 T/B checkpoint configured; predicted-query consistency loss is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    path = Path(str(checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"Configured Stage 1 checkpoint does not exist: {path}")
    from tb_prediction.infer import load_model

    base_channels = stage1.get("base_channels")
    model = load_model(path, device=device, base_channels=None if base_channels is None else int(base_channels))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def predict_tb(model: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(image)
    if not isinstance(outputs, Mapping) or "T" not in outputs or "B" not in outputs:
        raise RuntimeError("Stage 1 predictor must return a mapping containing T and B")
    transmission, backscatter = outputs["T"], outputs["B"]
    if transmission.shape != image.shape or backscatter.shape != image.shape:
        raise RuntimeError(
            f"Stage 1 output shape mismatch: I={tuple(image.shape)}, T={tuple(transmission.shape)}, "
            f"B={tuple(backscatter.shape)}"
        )
    if not torch.isfinite(transmission).all() or not torch.isfinite(backscatter).all():
        raise RuntimeError("Stage 1 predictor returned NaN/Inf")
    return transmission, backscatter

