from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .models import BankPretrainingModel


MODULE_NAMES = (
    "key_encoder",
    "key_decoder",
    "value_encoder",
    "temporary_restorer",
    "key_projector",
    "value_projector",
)


def build_stage2_model(config: Mapping[str, Any]) -> BankPretrainingModel:
    data = config["data"]
    model = config["model"]
    return BankPretrainingModel(
        patch_size=int(data["patch_size"]),
        key_dim=int(model["key_dim"]),
        value_dim=int(model["value_dim"]),
        projection_dim=int(model["projection_dim"]),
        encoder_base_channels=int(model["base_channels"]),
        restorer_base_channels=int(model["base_channels"]),
    )


def module_state_dicts(model: BankPretrainingModel) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: getattr(model, name).state_dict() for name in MODULE_NAMES}


def load_module_state_dicts(
    model: BankPretrainingModel, checkpoint: Mapping[str, Any], *, strict: bool = True
) -> None:
    missing = [name for name in MODULE_NAMES if name not in checkpoint]
    if missing:
        raise KeyError(f"Stage 2 checkpoint is missing module states: {missing}")
    try:
        for name in MODULE_NAMES:
            getattr(model, name).load_state_dict(checkpoint[name], strict=strict)
    except RuntimeError as error:
        raise RuntimeError(
            "Stage 2 checkpoint dimensions are incompatible with the current configuration. "
            "Check patch_size, key_dim, value_dim, projection_dim, and base_channels."
        ) from error

