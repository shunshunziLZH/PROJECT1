"""Neural key/value models used to pre-train the Stage 2 degradation bank.

The modules in this file deliberately avoid batch-dependent normalization.  In
particular, no :class:`torch.nn.BatchNorm2d` is used because batch statistics
can change the colour and brightness representation needed by this task.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Size2D = Union[int, Tuple[int, int]]


def _as_size(size: Size2D) -> Tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("spatial size must be positive")
        return size, size
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("spatial size must contain two positive integers")
    return int(size[0]), int(size[1])


def make_physical_key_input(T: Tensor, B: Tensor) -> Tensor:
    """Construct the nine-channel key input ``[T, B, 1-T]``."""

    if T.ndim != 4 or B.ndim != 4:
        raise ValueError("T and B must be BCHW tensors")
    if T.shape != B.shape:
        raise ValueError(f"T and B must have equal shapes, got {T.shape} and {B.shape}")
    if T.shape[1] != 3:
        raise ValueError(f"T and B must have three channels, got {T.shape[1]}")
    return torch.cat((T, B, 1.0 - T), dim=1)


def make_value_input(I: Tensor, J: Tensor) -> Tensor:
    """Construct the six-channel value input ``[J, J-I]``."""

    if I.ndim != 4 or J.ndim != 4:
        raise ValueError("I and J must be BCHW tensors")
    if I.shape != J.shape:
        raise ValueError(f"I and J must have equal shapes, got {I.shape} and {J.shape}")
    if I.shape[1] != 3:
        raise ValueError(f"I and J must have three channels, got {I.shape[1]}")
    return torch.cat((J, J - I), dim=1)


class ResidualBlock(nn.Module):
    """Small residual block with a learnable, initially conservative branch."""

    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_scale))
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x) * self.scale


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, downsample: bool):
        super().__init__()
        if blocks < 1:
            raise ValueError("each encoder stage needs at least one residual block")
        if downsample:
            self.input = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        elif in_channels != out_channels:
            self.input = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.input = nn.Identity()
        self.blocks = nn.Sequential(*(ResidualBlock(out_channels) for _ in range(blocks)))

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.input(x))


class KeyEncoder(nn.Module):
    """Encode ``[T, B, 1-T]`` into raw and L2-normalized key vectors.

    ``forward`` returns ``(q_raw, q)``.  Keeping both outputs is useful because
    the decoder can preserve embedding magnitude while retrieval always uses
    the normalized key ``q``.
    """

    def __init__(
        self,
        in_channels: int = 9,
        key_dim: int = 64,
        base_channels: int = 32,
        blocks_per_stage: Tuple[int, int, int, int] = (1, 1, 2, 2),
    ):
        super().__init__()
        if in_channels != 9:
            raise ValueError("the physical key encoder expects nine input channels")
        if key_dim <= 0 or base_channels <= 0:
            raise ValueError("key_dim and base_channels must be positive")
        if len(blocks_per_stage) != 4:
            raise ValueError("blocks_per_stage must contain four integers")

        c = base_channels
        self.in_channels = in_channels
        self.key_dim = key_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.GELU(),
        )
        self.stage1 = EncoderStage(c, c, blocks_per_stage[0], downsample=False)
        self.stage2 = EncoderStage(c, c * 2, blocks_per_stage[1], downsample=True)
        self.stage3 = EncoderStage(c * 2, c * 4, blocks_per_stage[2], downsample=True)
        self.stage4 = EncoderStage(c * 4, c * 8, blocks_per_stage[3], downsample=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c * 8, c * 4),
            nn.GELU(),
            nn.Linear(c * 4, key_dim),
        )

    def forward(self, physical_input: Tensor) -> Tuple[Tensor, Tensor]:
        if physical_input.ndim != 4 or physical_input.shape[1] != self.in_channels:
            raise ValueError(
                f"physical_input must have shape [B, {self.in_channels}, H, W], "
                f"got {tuple(physical_input.shape)}"
            )
        x = self.stem(physical_input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        q_raw = self.head(self.pool(x))
        q = F.normalize(q_raw, p=2, dim=-1, eps=1e-8)
        return q_raw, q


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int = 1):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(out_channels) for _ in range(blocks)))

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.blocks(self.projection(x))


class KeyDecoder(nn.Module):
    """Decode a key vector back to a nine-channel physical map."""

    def __init__(
        self,
        key_dim: int = 64,
        out_channels: int = 9,
        patch_size: Size2D = 64,
        base_channels: int = 32,
        seed_size: int = 4,
    ):
        super().__init__()
        if key_dim <= 0 or base_channels <= 0 or seed_size <= 0:
            raise ValueError("key_dim, base_channels and seed_size must be positive")
        if out_channels != 9:
            raise ValueError("the key decoder must reconstruct nine physical channels")
        self.key_dim = key_dim
        self.out_channels = out_channels
        self.output_size = _as_size(patch_size)
        self.seed_size = seed_size
        start_channels = base_channels * 8
        self.fc = nn.Sequential(
            nn.Linear(key_dim, start_channels * seed_size * seed_size),
            nn.GELU(),
        )
        self.up1 = DecoderStage(start_channels, base_channels * 4, blocks=2)
        self.up2 = DecoderStage(base_channels * 4, base_channels * 2, blocks=1)
        self.up3 = DecoderStage(base_channels * 2, base_channels, blocks=1)
        self.up4 = DecoderStage(base_channels, base_channels, blocks=1)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.start_channels = start_channels

    def forward(self, key: Tensor, output_size: Optional[Size2D] = None) -> Tensor:
        if key.ndim != 2 or key.shape[1] != self.key_dim:
            raise ValueError(f"key must have shape [B, {self.key_dim}], got {tuple(key.shape)}")
        x = self.fc(key).reshape(
            key.shape[0], self.start_channels, self.seed_size, self.seed_size
        )
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        target_size = self.output_size if output_size is None else _as_size(output_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


class ValueEncoder(nn.Module):
    """Encode ``[J, J-I]`` into an unnormalized restoration value vector."""

    def __init__(
        self,
        in_channels: int = 6,
        value_dim: int = 128,
        base_channels: int = 32,
        blocks_per_stage: Tuple[int, int, int, int] = (1, 1, 2, 2),
    ):
        super().__init__()
        if in_channels != 6:
            raise ValueError("the value encoder expects six input channels")
        if value_dim <= 0 or base_channels <= 0:
            raise ValueError("value_dim and base_channels must be positive")
        if len(blocks_per_stage) != 4:
            raise ValueError("blocks_per_stage must contain four integers")

        c = base_channels
        self.in_channels = in_channels
        self.value_dim = value_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.GELU(),
        )
        self.stage1 = EncoderStage(c, c, blocks_per_stage[0], downsample=False)
        self.stage2 = EncoderStage(c, c * 2, blocks_per_stage[1], downsample=True)
        self.stage3 = EncoderStage(c * 2, c * 4, blocks_per_stage[2], downsample=True)
        self.stage4 = EncoderStage(c * 4, c * 8, blocks_per_stage[3], downsample=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c * 8, c * 4),
            nn.GELU(),
            nn.Linear(c * 4, value_dim),
        )

    def forward(self, value_input: Tensor) -> Tensor:
        if value_input.ndim != 4 or value_input.shape[1] != self.in_channels:
            raise ValueError(
                f"value_input must have shape [B, {self.in_channels}, H, W], "
                f"got {tuple(value_input.shape)}"
            )
        x = self.stem(value_input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(self.pool(x))


class FiLM(nn.Module):
    """Feature-wise affine modulation driven by a value embedding."""

    def __init__(self, value_dim: int, channels: int):
        super().__init__()
        if value_dim <= 0 or channels <= 0:
            raise ValueError("value_dim and channels must be positive")
        hidden = max(channels, value_dim // 2, 16)
        self.affine = nn.Sequential(
            nn.Linear(value_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        # Identity modulation is a stable starting point; gradients still flow
        # into the final affine layer on the first optimization step.
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, x: Tensor, value: Tensor) -> Tensor:
        if value.ndim != 2 or value.shape[0] != x.shape[0]:
            raise ValueError("value must be [B, value_dim] and match the feature batch")
        gamma, beta = self.affine(value).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class RestorationUpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up_projection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_projection(x)
        return self.fuse(torch.cat((x, skip), dim=1))


class TemporaryRestorer(nn.Module):
    """A deliberately small value-conditioned U-Net used only in Stage 2.

    FiLM is applied at the bottleneck and at both decoder resolutions, so the
    value embedding cannot be bypassed by the skip connections alone.
    """

    def __init__(
        self,
        value_dim: int = 128,
        base_channels: int = 32,
        residual_limit: float = 0.5,
    ):
        super().__init__()
        if value_dim <= 0 or base_channels <= 0 or residual_limit <= 0:
            raise ValueError("value_dim, base_channels and residual_limit must be positive")
        c = base_channels
        self.value_dim = value_dim
        self.residual_limit = float(residual_limit)

        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(ResidualBlock(c * 4), ResidualBlock(c * 4))

        self.film_bottleneck = FiLM(value_dim, c * 4)
        self.up2 = RestorationUpStage(c * 4, c * 2, c * 2)
        self.film_decoder2 = FiLM(value_dim, c * 2)
        self.up1 = RestorationUpStage(c * 2, c, c)
        self.film_decoder1 = FiLM(value_dim, c)
        self.head = nn.Sequential(
            ResidualBlock(c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, I: Tensor, value: Tensor) -> Tensor:
        if I.ndim != 4 or I.shape[1] != 3:
            raise ValueError(f"I must have shape [B, 3, H, W], got {tuple(I.shape)}")
        if value.ndim != 2 or value.shape != (I.shape[0], self.value_dim):
            raise ValueError(
                f"value must have shape [B, {self.value_dim}], got {tuple(value.shape)}"
            )

        skip1 = self.enc1(self.stem(I))
        skip2 = self.enc2(self.down1(skip1))
        x = self.bottleneck(self.down2(skip2))
        x = self.film_bottleneck(x, value)
        x = self.up2(x, skip2)
        x = self.film_decoder2(x, value)
        x = self.up1(x, skip1)
        x = self.film_decoder1(x, value)
        residual = torch.tanh(self.head(x)) * self.residual_limit
        return torch.clamp(I + residual, 0.0, 1.0)


class ProjectionHead(nn.Module):
    """Two-layer projection head used only for key/value relation alignment."""

    def __init__(self, input_dim: int, projection_dim: int = 64, hidden_dim: Optional[int] = None):
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0:
            raise ValueError("input_dim and projection_dim must be positive")
        hidden_dim = hidden_dim or max(input_dim, projection_dim)
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"embedding must have shape [B, {self.input_dim}], got {tuple(embedding.shape)}"
            )
        return self.net(embedding)


class KeyProjector(ProjectionHead):
    def __init__(self, key_dim: int = 64, projection_dim: int = 64, hidden_dim: Optional[int] = None):
        super().__init__(key_dim, projection_dim, hidden_dim)


class ValueProjector(ProjectionHead):
    def __init__(
        self, value_dim: int = 128, projection_dim: int = 64, hidden_dim: Optional[int] = None
    ):
        super().__init__(value_dim, projection_dim, hidden_dim)


class BankPretrainingModel(nn.Module):
    """Composition of all trainable Stage 2 bank pre-training modules."""

    def __init__(
        self,
        patch_size: Size2D = 64,
        key_dim: int = 64,
        value_dim: int = 128,
        projection_dim: int = 64,
        encoder_base_channels: int = 32,
        restorer_base_channels: int = 32,
    ):
        super().__init__()
        self.patch_size = _as_size(patch_size)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.projection_dim = projection_dim

        self.key_encoder = KeyEncoder(key_dim=key_dim, base_channels=encoder_base_channels)
        self.key_decoder = KeyDecoder(
            key_dim=key_dim,
            patch_size=self.patch_size,
            base_channels=encoder_base_channels,
        )
        self.value_encoder = ValueEncoder(
            value_dim=value_dim, base_channels=encoder_base_channels
        )
        self.temporary_restorer = TemporaryRestorer(
            value_dim=value_dim, base_channels=restorer_base_channels
        )
        self.key_projector = KeyProjector(key_dim, projection_dim)
        self.value_projector = ValueProjector(value_dim, projection_dim)

    def forward(
        self,
        I: Tensor,
        J: Tensor,
        T: Tensor,
        B: Tensor,
        T_pred: Optional[Tensor] = None,
        B_pred: Optional[Tensor] = None,
        J_aug: Optional[Tensor] = None,
        I_aug: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Run all available pre-training branches.

        Predicted-query and invariance outputs are ``None`` when their optional
        inputs are absent.  If ``J_aug`` is supplied, ``I_aug`` defaults to the
        exactly consistent image ``J_aug * T + B``.
        """

        P_gt = make_physical_key_input(T, B)
        q_raw, q = self.key_encoder(P_gt)
        P_reconstructed = self.key_decoder(q_raw, output_size=P_gt.shape[-2:])

        v = self.value_encoder(make_value_input(I, J))
        J_temp = self.temporary_restorer(I, v)
        z_key = self.key_projector(q)
        z_value = self.value_projector(v)

        if (T_pred is None) != (B_pred is None):
            raise ValueError("T_pred and B_pred must either both be supplied or both be omitted")
        q_pred_raw: Optional[Tensor] = None
        q_pred: Optional[Tensor] = None
        P_pred: Optional[Tensor] = None
        if T_pred is not None and B_pred is not None:
            P_pred = make_physical_key_input(T_pred, B_pred)
            q_pred_raw, q_pred = self.key_encoder(P_pred)

        if I_aug is not None and J_aug is None:
            raise ValueError("J_aug is required when I_aug is supplied")
        v_aug: Optional[Tensor] = None
        if J_aug is not None:
            if J_aug.shape != J.shape:
                raise ValueError("J_aug must have the same shape as J")
            if I_aug is None:
                I_aug = J_aug * T + B
            v_aug = self.value_encoder(make_value_input(I_aug, J_aug))

        return {
            "P_gt": P_gt,
            "P_pred": P_pred,
            "q_raw": q_raw,
            "q": q,
            "q_pred_raw": q_pred_raw,
            "q_pred": q_pred,
            "v": v,
            "v_aug": v_aug,
            "J_temp": J_temp,
            "P_reconstructed": P_reconstructed,
            "z_key": z_key,
            "z_value": z_value,
        }


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of parameters, optionally excluding frozen tensors."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "BankPretrainingModel",
    "FiLM",
    "KeyDecoder",
    "KeyEncoder",
    "KeyProjector",
    "ProjectionHead",
    "ResidualBlock",
    "TemporaryRestorer",
    "ValueEncoder",
    "ValueProjector",
    "count_parameters",
    "make_physical_key_input",
    "make_value_input",
]
