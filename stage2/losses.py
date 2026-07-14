"""Losses and image-quality metrics for Stage 2 bank pre-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Reduction = str


def _validate_image_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have equal shapes, got {prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError("image tensors must have shape [B, C, H, W]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("image tensors must be floating point")


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def charbonnier_loss(
    prediction: Tensor,
    target: Tensor,
    eps: float = 1e-3,
    reduction: Reduction = "mean",
) -> Tensor:
    """Robust differentiable L1 loss."""

    _validate_image_pair(prediction, target)
    if eps <= 0:
        raise ValueError("eps must be positive")
    error = torch.sqrt((prediction - target).square() + eps * eps)
    if reduction == "none":
        return error
    return _reduce(error, reduction)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: Reduction = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return charbonnier_loss(prediction, target, self.eps, self.reduction)


def gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """L1 distance between first-order horizontal and vertical gradients."""

    _validate_image_pair(prediction, target)
    zero = prediction.sum() * 0.0
    horizontal = zero
    vertical = zero
    if prediction.shape[-1] > 1:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        horizontal = F.l1_loss(pred_dx, target_dx)
    if prediction.shape[-2] > 1:
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        vertical = F.l1_loss(pred_dy, target_dy)
    return horizontal + vertical


def key_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    lambda_grad: float = 0.1,
) -> Tensor:
    """L1 physical-map reconstruction plus a spatial-gradient penalty."""

    if lambda_grad < 0:
        raise ValueError("lambda_grad must be non-negative")
    _validate_image_pair(prediction, target)
    return F.l1_loss(prediction, target) + lambda_grad * gradient_loss(prediction, target)


def key_reconstruction_components(
    prediction: Tensor,
    target: Tensor,
    lambda_grad: float = 0.1,
) -> Dict[str, Tensor]:
    l1 = F.l1_loss(prediction, target)
    grad = gradient_loss(prediction, target)
    return {"key_rec": l1 + lambda_grad * grad, "key_rec_l1": l1, "key_rec_grad": grad}


def _gaussian_kernel(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def _effective_window_size(height: int, width: int, requested: int) -> int:
    if requested <= 0:
        raise ValueError("window_size must be positive")
    size = min(requested, height, width)
    if size % 2 == 0:
        size -= 1
    return max(size, 1)


def ssim_map(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Return a differentiable SSIM map using a Gaussian local window."""

    _validate_image_pair(prediction, target)
    if data_range <= 0 or sigma <= 0:
        raise ValueError("data_range and sigma must be positive")
    channels = prediction.shape[1]
    size = _effective_window_size(prediction.shape[-2], prediction.shape[-1], window_size)
    kernel = _gaussian_kernel(channels, size, sigma, prediction.device, prediction.dtype)
    padding = size // 2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction.square(), kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target.square(), kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy

    # Roundoff can make a variance very slightly negative in half precision.
    sigma_x_sq = sigma_x_sq.clamp_min(0.0)
    sigma_y_sq = sigma_y_sq.clamp_min(0.0)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)


def ssim_metric(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    reduction: Reduction = "mean",
) -> Tensor:
    """Structural similarity, returned per image for ``reduction='none'``."""

    values = ssim_map(
        prediction, target, data_range=data_range, window_size=window_size
    ).mean(dim=(1, 2, 3))
    return _reduce(values, reduction)


def ssim_loss(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    reduction: Reduction = "mean",
) -> Tensor:
    values = 1.0 - ssim_metric(
        prediction,
        target,
        data_range=data_range,
        window_size=window_size,
        reduction="none",
    )
    return _reduce(values, reduction)


def fft_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """L1 distance between log-amplitude spectra (orthonormal FFT)."""

    _validate_image_pair(prediction, target)
    pred_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    pred_amplitude = torch.log1p(pred_spectrum.abs())
    target_amplitude = torch.log1p(target_spectrum.abs())
    loss = F.l1_loss(pred_amplitude, target_amplitude)
    return loss.to(dtype=prediction.dtype)


def restoration_loss(
    prediction: Tensor,
    target: Tensor,
    lambda_ssim: float = 0.2,
    lambda_fft: float = 0.05,
    charbonnier_eps: float = 1e-3,
) -> Tensor:
    """Charbonnier + SSIM + frequency loss for the temporary restorer."""

    if lambda_ssim < 0 or lambda_fft < 0:
        raise ValueError("restoration loss weights must be non-negative")
    return (
        charbonnier_loss(prediction, target, eps=charbonnier_eps)
        + lambda_ssim * ssim_loss(prediction, target)
        + lambda_fft * fft_loss(prediction, target)
    )


def restoration_loss_components(
    prediction: Tensor,
    target: Tensor,
    lambda_ssim: float = 0.2,
    lambda_fft: float = 0.05,
    charbonnier_eps: float = 1e-3,
) -> Dict[str, Tensor]:
    charb = charbonnier_loss(prediction, target, eps=charbonnier_eps)
    structural = ssim_loss(prediction, target)
    frequency = fft_loss(prediction, target)
    total = charb + lambda_ssim * structural + lambda_fft * frequency
    return {
        "restore": total,
        "restore_charbonnier": charb,
        "restore_ssim": structural,
        "restore_fft": frequency,
    }


def query_consistency_loss(q_pred: Tensor, q_gt: Tensor) -> Tensor:
    """Cosine consistency with a stop-gradient ground-truth query teacher."""

    if q_pred.shape != q_gt.shape or q_pred.ndim != 2:
        raise ValueError("q_pred and q_gt must be equal [B, key_dim] tensors")
    return (1.0 - F.cosine_similarity(q_pred, q_gt.detach(), dim=-1, eps=1e-8)).mean()


def physical_relation_target(
    physical_maps: Tensor,
    tau_physical: float = 0.1,
    downsample_size: Union[int, Tuple[int, int]] = 8,
) -> Tensor:
    """Create row-normalized soft relations from pairwise physical distances."""

    if physical_maps.ndim != 4:
        raise ValueError("physical_maps must be [B, C, H, W]")
    if tau_physical <= 0:
        raise ValueError("tau_physical must be positive")
    if isinstance(downsample_size, int):
        output_size = (downsample_size, downsample_size)
    else:
        output_size = downsample_size
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("downsample_size must be positive")

    compact = F.adaptive_avg_pool2d(physical_maps.float(), output_size).flatten(1)
    # Mean squared distance keeps the physical temperature independent of the
    # configured pooling resolution and number of channels.
    difference = compact[:, None, :] - compact[None, :, :]
    distance = difference.square().mean(dim=-1)
    return F.softmax(-distance / tau_physical, dim=1).detach()


def _target_to_prediction_kl(target: Tensor, logits: Tensor) -> Tensor:
    log_prediction = F.log_softmax(logits, dim=1)
    safe_target = target.clamp_min(1e-8)
    return (target * (safe_target.log() - log_prediction)).sum(dim=1).mean()


def soft_relation_alignment_loss(
    z_key: Tensor,
    z_value: Tensor,
    physical_maps: Tensor,
    tau_physical: float = 0.1,
    tau_embedding: float = 0.1,
    downsample_size: Union[int, Tuple[int, int]] = 8,
) -> Tensor:
    """Bidirectional KL alignment to soft, physical-similarity relations."""

    if z_key.ndim != 2 or z_value.ndim != 2:
        raise ValueError("z_key and z_value must be two-dimensional")
    if z_key.shape != z_value.shape:
        raise ValueError(
            "projected key and value tensors must have equal [B, projection_dim] shapes"
        )
    if z_key.shape[0] != physical_maps.shape[0]:
        raise ValueError("embedding and physical-map batch sizes must match")
    if tau_embedding <= 0:
        raise ValueError("tau_embedding must be positive")

    target_forward = physical_relation_target(
        physical_maps, tau_physical=tau_physical, downsample_size=downsample_size
    )
    # Re-normalizing the transposed relation matrix is important: a literal
    # transpose of a row-stochastic matrix need not remain row-stochastic.
    target_reverse = target_forward.transpose(0, 1)
    target_reverse = target_reverse / target_reverse.sum(dim=1, keepdim=True).clamp_min(1e-8)

    key = F.normalize(z_key, dim=-1, eps=1e-8)
    value = F.normalize(z_value, dim=-1, eps=1e-8)
    logits_forward = key @ value.transpose(0, 1) / tau_embedding
    logits_reverse = logits_forward.transpose(0, 1)
    return 0.5 * (
        _target_to_prediction_kl(target_forward, logits_forward)
        + _target_to_prediction_kl(target_reverse, logits_reverse)
    )


def augment_clean_content(
    J: Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
    gamma: float = 0.1,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.0,
    blur_probability: float = 0.15,
) -> Tensor:
    """Apply moderate per-image content augmentation to a clean batch.

    This helper intentionally receives only ``J``.  Physical maps are not
    transformed; callers must regenerate the corresponding degraded image.
    """

    if J.ndim != 4 or J.shape[1] != 3 or not J.is_floating_point():
        raise ValueError("J must be a floating-point [B, 3, H, W] tensor")
    for name, magnitude in (("brightness", brightness), ("contrast", contrast), ("gamma", gamma)):
        if not 0.0 <= magnitude < 1.0:
            raise ValueError(f"{name} magnitude must be in [0, 1)")
    for name, probability in (
        ("horizontal_flip_probability", horizontal_flip_probability),
        ("vertical_flip_probability", vertical_flip_probability),
        ("blur_probability", blur_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    batch = J.shape[0]
    shape = (batch, 1, 1, 1)
    dtype, device = J.dtype, J.device
    brightness_factor = 1.0 + (torch.rand(shape, dtype=dtype, device=device) * 2.0 - 1.0) * brightness
    contrast_factor = 1.0 + (torch.rand(shape, dtype=dtype, device=device) * 2.0 - 1.0) * contrast
    gamma_factor = 1.0 + (torch.rand(shape, dtype=dtype, device=device) * 2.0 - 1.0) * gamma

    augmented = J * brightness_factor
    mean = augmented.mean(dim=(1, 2, 3), keepdim=True)
    augmented = (augmented - mean) * contrast_factor + mean
    augmented = augmented.clamp(0.0, 1.0).pow(gamma_factor)

    if horizontal_flip_probability > 0:
        mask = torch.rand(shape, device=device) < horizontal_flip_probability
        augmented = torch.where(mask, augmented.flip(-1), augmented)
    if vertical_flip_probability > 0:
        mask = torch.rand(shape, device=device) < vertical_flip_probability
        augmented = torch.where(mask, augmented.flip(-2), augmented)
    if blur_probability > 0 and min(J.shape[-2:]) > 2:
        blurred = F.avg_pool2d(F.pad(augmented, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        mask = torch.rand(shape, device=device) < blur_probability
        augmented = torch.where(mask, blurred, augmented)
    return augmented.clamp(0.0, 1.0)


def build_value_invariance_pair(
    J: Tensor,
    T: Tensor,
    B: Tensor,
    augmentation: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Return ``(J_aug, I_aug)`` with exact ``I_aug = J_aug*T + B``."""

    if J.shape != T.shape or J.shape != B.shape:
        raise ValueError("J, T and B must have identical shapes")
    augmentation = augmentation or augment_clean_content
    J_aug = augmentation(J)
    if J_aug.shape != J.shape:
        raise ValueError("the content augmentation must preserve J's shape")
    I_aug = J_aug * T + B
    return J_aug, I_aug


def value_invariance_loss(value: Tensor, value_augmented: Tensor) -> Tensor:
    """Cosine loss that discourages scene-content leakage in bank values."""

    if value.shape != value_augmented.shape or value.ndim != 2:
        raise ValueError("value tensors must have equal [B, value_dim] shapes")
    return (
        1.0 - F.cosine_similarity(value, value_augmented, dim=-1, eps=1e-8)
    ).mean()


def shuffle_values(value: Tensor) -> Tensor:
    """Return a deterministic derangement for ranking diagnostics.

    A one-position roll guarantees a different sample index whenever the batch
    contains at least two examples.  For a singleton batch no valid wrong value
    exists, so the input is returned unchanged.
    """

    if value.ndim != 2:
        raise ValueError("value must be [B, value_dim]")
    return value.roll(1, dims=0) if value.shape[0] > 1 else value


def wrong_value_ranking_loss(
    correct_prediction: Tensor,
    wrong_prediction: Tensor,
    target: Tensor,
    margin: float = 0.05,
    charbonnier_eps: float = 1e-3,
) -> Tensor:
    """Require the correct bank value to restore better than a wrong value."""

    _validate_image_pair(correct_prediction, target)
    _validate_image_pair(wrong_prediction, target)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    correct_error = torch.sqrt(
        (correct_prediction - target).square() + charbonnier_eps**2
    ).mean(dim=(1, 2, 3))
    wrong_error = torch.sqrt(
        (wrong_prediction - target).square() + charbonnier_eps**2
    ).mean(dim=(1, 2, 3))
    return F.relu(margin + correct_error - wrong_error).mean()


def psnr(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Peak signal-to-noise ratio in dB, computed independently per image."""

    _validate_image_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
    peak = torch.as_tensor(data_range**2, device=mse.device, dtype=mse.dtype)
    values = 10.0 * torch.log10(peak / mse.clamp_min(torch.finfo(mse.dtype).tiny))
    return _reduce(values, reduction)


# Explicit aliases are convenient for downstream code that uses compute_* names.
compute_psnr = psnr
compute_ssim = ssim_metric


@dataclass(frozen=True)
class BankLossWeights:
    restore: float = 1.0
    key: float = 0.5
    align: float = 0.1
    query: float = 0.2
    invariance: float = 0.1
    ranking: float = 0.0


class BankPretrainingLoss(nn.Module):
    """Optional convenience wrapper returning total and all logging components."""

    def __init__(
        self,
        weights: BankLossWeights = BankLossWeights(),
        lambda_grad: float = 0.1,
        lambda_ssim: float = 0.2,
        lambda_fft: float = 0.05,
        tau_physical: float = 0.1,
        tau_embedding: float = 0.1,
        physical_relation_size: Union[int, Tuple[int, int]] = 8,
        ranking_margin: float = 0.05,
    ):
        super().__init__()
        self.weights = weights
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.lambda_fft = lambda_fft
        self.tau_physical = tau_physical
        self.tau_embedding = tau_embedding
        self.physical_relation_size = physical_relation_size
        self.ranking_margin = ranking_margin

    def forward(
        self,
        outputs: Mapping[str, Optional[Tensor]],
        target_J: Tensor,
        wrong_restoration: Optional[Tensor] = None,
        enable_alignment: bool = True,
    ) -> Dict[str, Tensor]:
        required = ("P_reconstructed", "P_gt", "J_temp", "q", "v", "z_key", "z_value")
        missing = [name for name in required if outputs.get(name) is None]
        if missing:
            raise KeyError(f"model outputs are missing required tensors: {missing}")

        # The checks above establish these values are tensors.
        P_reconstructed = outputs["P_reconstructed"]
        P_gt = outputs["P_gt"]
        J_temp = outputs["J_temp"]
        q = outputs["q"]
        value = outputs["v"]
        z_key = outputs["z_key"]
        z_value = outputs["z_value"]
        assert P_reconstructed is not None and P_gt is not None
        assert J_temp is not None and q is not None and value is not None
        assert z_key is not None and z_value is not None

        components: Dict[str, Tensor] = {}
        components.update(
            key_reconstruction_components(P_reconstructed, P_gt, self.lambda_grad)
        )
        components.update(
            restoration_loss_components(
                J_temp, target_J, self.lambda_ssim, self.lambda_fft
            )
        )
        zero = J_temp.sum() * 0.0

        q_pred = outputs.get("q_pred")
        components["query"] = (
            query_consistency_loss(q_pred, q) if q_pred is not None else zero
        )
        components["align"] = (
            soft_relation_alignment_loss(
                z_key,
                z_value,
                P_gt,
                tau_physical=self.tau_physical,
                tau_embedding=self.tau_embedding,
                downsample_size=self.physical_relation_size,
            )
            if enable_alignment
            else zero
        )
        value_aug = outputs.get("v_aug")
        components["invariance"] = (
            value_invariance_loss(value, value_aug) if value_aug is not None else zero
        )
        components["ranking"] = (
            wrong_value_ranking_loss(
                J_temp, wrong_restoration, target_J, margin=self.ranking_margin
            )
            if wrong_restoration is not None
            else zero
        )
        components["total"] = (
            self.weights.restore * components["restore"]
            + self.weights.key * components["key_rec"]
            + self.weights.align * components["align"]
            + self.weights.query * components["query"]
            + self.weights.invariance * components["invariance"]
            + self.weights.ranking * components["ranking"]
        )
        return components


__all__ = [
    "BankLossWeights",
    "BankPretrainingLoss",
    "CharbonnierLoss",
    "augment_clean_content",
    "build_value_invariance_pair",
    "charbonnier_loss",
    "compute_psnr",
    "compute_ssim",
    "fft_loss",
    "gradient_loss",
    "key_reconstruction_components",
    "key_reconstruction_loss",
    "physical_relation_target",
    "psnr",
    "query_consistency_loss",
    "restoration_loss",
    "restoration_loss_components",
    "shuffle_values",
    "soft_relation_alignment_loss",
    "ssim_loss",
    "ssim_map",
    "ssim_metric",
    "value_invariance_loss",
    "wrong_value_ranking_loss",
]
