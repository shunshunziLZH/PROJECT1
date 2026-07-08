from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + eps)
        ctx.eps = eps
        ctx.save_for_backward(y, var, weight)
        return y * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)
        grad_x = torch.rsqrt(var + eps) * (grad - y * mean_grad_y - mean_grad)
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """NAFNet-style channel-wise LayerNorm for image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def _make_blocks(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[StableResBlock(channels) for _ in range(num_blocks)])


class StableResBlock(nn.Module):
    """NAF-style residual block with normalized, zero-initialized residual paths."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm1(x)
        residual = self.conv1(residual)
        residual = self.conv2(residual)
        residual = self.sg(residual)
        residual = residual * self.sca(residual)
        residual = self.conv3(residual)
        y = x + residual * self.beta

        residual = self.norm2(y)
        residual = self.conv4(residual)
        residual = self.sg(residual)
        residual = self.conv5(residual)
        return y + residual * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, skip_scale: float = 1.0):
        super().__init__()
        self.skip_scale = skip_scale
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.blocks = _make_blocks(out_channels, num_blocks)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None and self.skip_scale != 0:
            x = (x + skip * self.skip_scale) / math.sqrt(1.0 + self.skip_scale**2)
        return self.blocks(x)


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 16, out_channels: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@dataclass(frozen=True)
class TBResUNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    encoder_blocks: Tuple[int, int, int] = (2, 2, 4)
    bottleneck_blocks: int = 4
    t_decoder_blocks: Tuple[int, int, int] = (2, 2, 2)
    b_decoder_blocks: Tuple[int, int, int] = (2, 2, 1)
    b_deep_skip_scale: float = 0.25


class TBResUNet(nn.Module):
    """Shared-encoder dual-decoder network for estimating T and B from I.

    Input and outputs are expected to be in [0, 1]. The forward method returns
    a dict with keys "T" and "B", both cropped back to the original HxW.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        encoder_blocks: Sequence[int] = (2, 2, 4),
        bottleneck_blocks: int = 4,
        t_decoder_blocks: Sequence[int] = (2, 2, 2),
        b_decoder_blocks: Sequence[int] = (2, 2, 1),
        b_deep_skip_scale: float = 0.25,
    ):
        super().__init__()
        if len(encoder_blocks) != 3:
            raise ValueError("encoder_blocks must have three stages")
        if len(t_decoder_blocks) != 3:
            raise ValueError("t_decoder_blocks must have three stages")
        if len(b_decoder_blocks) != 3:
            raise ValueError("b_decoder_blocks must have three stages")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.config = TBResUNetConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            encoder_blocks=tuple(encoder_blocks),
            bottleneck_blocks=bottleneck_blocks,
            t_decoder_blocks=tuple(t_decoder_blocks),
            b_decoder_blocks=tuple(b_decoder_blocks),
            b_deep_skip_scale=b_deep_skip_scale,
        )

        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.enc1 = _make_blocks(c1, encoder_blocks[0])
        self.down1 = Downsample(c1, c2)
        self.enc2 = _make_blocks(c2, encoder_blocks[1])
        self.down2 = Downsample(c2, c3)
        self.enc3 = _make_blocks(c3, encoder_blocks[2])
        self.down3 = Downsample(c3, c4)

        self.bottleneck = _make_blocks(c4, bottleneck_blocks)

        self.t_up3 = UpBlock(c4, c3, t_decoder_blocks[0], skip_scale=1.0)
        self.t_up2 = UpBlock(c3, c2, t_decoder_blocks[1], skip_scale=1.0)
        self.t_up1 = UpBlock(c2, c1, t_decoder_blocks[2], skip_scale=1.0)
        self.t_head = PredictionHead(c1, hidden_channels=max(16, c1 // 2), out_channels=out_channels)

        self.b_up3 = UpBlock(c4, c3, b_decoder_blocks[0], skip_scale=b_deep_skip_scale)
        self.b_up2 = UpBlock(c3, c2, b_decoder_blocks[1], skip_scale=0.0)
        self.b_up1 = UpBlock(c2, c1, b_decoder_blocks[2], skip_scale=0.0)
        self.b_head = PredictionHead(c1, hidden_channels=max(16, c1 // 2), out_channels=out_channels)

    @property
    def padding_multiple(self) -> int:
        return 8

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, height, width = x.shape
        x = self._pad_to_multiple(x)

        x = self.stem(x)
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down1(skip1))
        skip3 = self.enc3(self.down2(skip2))
        latent = self.bottleneck(self.down3(skip3))

        t = self.t_up3(latent, skip3)
        t = self.t_up2(t, skip2)
        t = self.t_up1(t, skip1)
        t = self.t_head(t)

        b = self.b_up3(latent, skip3)
        b = self.b_up2(b)
        b = self.b_up1(b)
        b = self.b_head(b)

        return {
            "T": t[:, :, :height, :width],
            "B": b[:, :, :height, :width],
        }

    def _pad_to_multiple(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_h = (self.padding_multiple - height % self.padding_multiple) % self.padding_multiple
        pad_w = (self.padding_multiple - width % self.padding_multiple) % self.padding_multiple
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
