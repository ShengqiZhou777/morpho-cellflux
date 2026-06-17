from __future__ import annotations

import torch
from torch.nn import functional as F


def maybe_drop_conditions(
    condition: torch.Tensor,
    drop_prob: float,
    null_condition_id: int = 0,
) -> torch.Tensor:
    """Apply classifier-free condition dropout during training."""

    if drop_prob <= 0:
        return condition
    keep = torch.rand(condition.shape, device=condition.device) >= float(drop_prob)
    nulls = torch.full_like(condition, int(null_condition_id))
    return torch.where(keep, condition, nulls)


def maybe_noise_source(
    source: torch.Tensor,
    noise_prob: float,
    noise_scale: float,
) -> torch.Tensor:
    """Add optional Gaussian source noise for smoother velocity fields."""

    if noise_prob <= 0 or noise_scale <= 0:
        return source
    mask = torch.rand(source.shape[0], device=source.device) < float(noise_prob)
    noise = torch.randn_like(source) * float(noise_scale)
    return torch.where(mask[:, None, None, None], source + noise, source)


def image_mask(image: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """Estimate a cell mask from non-zero signal across microscopy channels."""

    return (image.amax(dim=1, keepdim=True) > float(threshold)).to(image.dtype)


def build_flow_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    condition_drop_prob: float = 0.0,
    source_noise_prob: float = 0.0,
    source_noise_scale: float = 0.0,
    null_condition_id: int = 0,
    target_scaffold: bool = False,
    mask_threshold: float = 1e-4,
    start_mode: str = "source",
    start_noise_scale: float = 0.2,
    start_noise_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct rectified-flow inputs and velocity targets."""

    original_source = source
    source = maybe_noise_source(source, source_noise_prob, source_noise_scale)
    if target_scaffold:
        mask = image_mask(target, mask_threshold)
        source = source * mask
        target = target * mask

    if start_mode == "source":
        start = source
    elif start_mode == "noise":
        start = torch.randn_like(source) * float(start_noise_scale)
        if target_scaffold:
            start = start * mask
    elif start_mode == "source_mean":
        source_mask = image_mask(original_source, mask_threshold)
        denom = source_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        channel_mean = (original_source * source_mask).sum(dim=(2, 3), keepdim=True) / denom
        start = channel_mean.expand_as(source)
        if target_scaffold:
            start = start * mask
    elif start_mode == "source_noise_mix":
        noise = torch.randn_like(source) * float(start_noise_scale)
        if target_scaffold:
            noise = noise * mask
        choose_noise = (
            torch.rand(source.shape[0], device=source.device) < float(start_noise_prob)
        )[:, None, None, None]
        start = torch.where(choose_noise, noise, source)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    condition = maybe_drop_conditions(condition, condition_drop_prob, null_condition_id)
    t = torch.rand(source.shape[0], device=source.device, dtype=source.dtype)
    t_view = t[:, None, None, None]
    x_t = (1.0 - t_view) * start + t_view * target
    velocity = target - start
    if target_scaffold:
        x_t = torch.cat([x_t, mask], dim=1)
    return x_t, t, condition, velocity


def _channel_weight_view(
    channel_weights: list[float] | tuple[float, ...] | torch.Tensor | None,
    image: torch.Tensor,
) -> torch.Tensor | None:
    if channel_weights is None:
        return None
    weights = torch.as_tensor(channel_weights, device=image.device, dtype=image.dtype)
    if weights.numel() != image.shape[1]:
        raise ValueError(
            f"channel_weights has {weights.numel()} values, expected {image.shape[1]}"
        )
    return weights[None, :, None, None]


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    channel_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE with optional foreground and channel weighting."""

    diff = (pred - target).square()
    weight = torch.ones_like(diff)
    channel_weight = _channel_weight_view(channel_weights, pred)
    if channel_weight is not None:
        weight = weight * channel_weight
    if mask is not None:
        weight = weight * mask
    denom = weight.sum().clamp_min(1.0)
    return (diff * weight).sum() / denom


def _validate_channels(
    channels: list[int] | tuple[int, ...] | None,
    image: torch.Tensor,
    name: str,
) -> tuple[int, ...]:
    if not channels:
        raise ValueError(f"{name} is required")
    parsed = tuple(int(channel) for channel in channels)
    invalid = [channel for channel in parsed if channel < 0 or channel >= image.shape[1]]
    if invalid:
        raise ValueError(
            f"{name} contains invalid channels {invalid}; image has {image.shape[1]} channels"
        )
    return parsed


def _select_channels(
    image: torch.Tensor,
    channels: list[int] | tuple[int, ...],
) -> torch.Tensor:
    index = torch.as_tensor(channels, device=image.device, dtype=torch.long)
    return image.index_select(dim=1, index=index)


def gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Depthwise Gaussian blur used to isolate puncta-scale high frequency signal."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    coords = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weight = kernel_2d[None, None].expand(image.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(image, weight, padding=radius, groups=image.shape[1])


def dog_response(
    image: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Local contrast response; positive values emphasize puncta-like signal."""

    return image - gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def combine_velocity_heads(
    model_output: torch.Tensor,
    image_channels: int,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return final velocity and optional high-frequency residual velocity head."""

    image_channels = int(image_channels)
    if model_output.shape[1] == image_channels:
        return model_output, None
    if model_output.shape[1] == image_channels * 2:
        base_velocity, residual_velocity = model_output.chunk(2, dim=1)
        return base_velocity + float(residual_scale) * residual_velocity, residual_velocity
    raise ValueError(
        f"model output has {model_output.shape[1]} channels; expected "
        f"{image_channels} or {image_channels * 2}"
    )


def highpass_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    channels: list[int] | tuple[int, ...],
    kernel_size: int = 9,
    sigma: float = 1.5,
) -> torch.Tensor:
    selected = _validate_channels(channels, pred, "highpass_channels")
    pred_response = dog_response(
        _select_channels(pred, selected),
        kernel_size=kernel_size,
        sigma=sigma,
    )
    target_response = dog_response(
        _select_channels(target, selected),
        kernel_size=kernel_size,
        sigma=sigma,
    )
    return masked_mse(pred_response, target_response, mask=mask)


def puncta_spatial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channels: list[int] | tuple[int, ...],
    fraction: float = 0.03,
    kernel_size: int = 9,
    sigma: float = 1.5,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Match puncta location, intensity, and coarse spatial moments in selected channels."""

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    selected = _validate_channels(channels, pred, "puncta_channels")
    pred_response = F.relu(
        dog_response(_select_channels(pred, selected), kernel_size=kernel_size, sigma=sigma)
    )
    target_response = F.relu(
        dog_response(_select_channels(target, selected), kernel_size=kernel_size, sigma=sigma)
    )

    batch_size, channel_count, height, width = pred_response.shape
    pixel_count = height * width
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=pred.device, dtype=pred.dtype),
        torch.linspace(-1.0, 1.0, width, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    mask_flat = mask[:, 0].reshape(batch_size, pixel_count).bool()
    active_counts = mask_flat.sum(dim=1)
    min_active = int(active_counts.min().item())
    if min_active <= 0:
        return pred.new_tensor(0.0)

    topk_count = min(max(1, int(round(pixel_count * float(fraction)))), min_active)
    mask_bc = (
        mask_flat[:, None, :]
        .expand(batch_size, channel_count, pixel_count)
        .reshape(batch_size * channel_count, pixel_count)
    )
    pred_flat = pred_response.reshape(batch_size * channel_count, pixel_count)
    target_flat = target_response.reshape(batch_size * channel_count, pixel_count)

    neg_inf = torch.finfo(pred.dtype).min
    pred_masked = pred_flat.masked_fill(~mask_bc, neg_inf)
    target_masked = target_flat.masked_fill(~mask_bc, neg_inf)

    target_top = torch.topk(target_masked, k=topk_count, dim=1)
    pred_at_target_peaks = pred_flat.gather(dim=1, index=target_top.indices)
    pred_top_values = torch.topk(pred_masked, k=topk_count, dim=1).values
    peak_location = F.mse_loss(pred_at_target_peaks, target_top.values)
    peak_mass = (pred_top_values.mean(dim=1) - target_top.values.mean(dim=1)).square().mean()

    pred_weights = torch.softmax(
        pred_flat.masked_fill(~mask_bc, neg_inf) / float(temperature),
        dim=1,
    )
    target_weights = torch.softmax(
        target_flat.masked_fill(~mask_bc, neg_inf) / float(temperature),
        dim=1,
    )
    pred_center = pred_weights @ coords
    target_center = target_weights @ coords
    coord_sq = coords.square()
    pred_second = pred_weights @ coord_sq
    target_second = target_weights @ coord_sq
    pred_spread = (pred_second - pred_center.square()).clamp_min(0.0)
    target_spread = (target_second - target_center.square()).clamp_min(0.0)
    moments = F.mse_loss(pred_center, target_center) + 0.25 * F.mse_loss(
        pred_spread,
        target_spread,
    )
    return peak_location + peak_mass + moments


def flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    pred_residual_velocity: torch.Tensor | None = None,
    target_image: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    channel_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    foreground_weight: float = 0.0,
    image_weight: float = 0.0,
    highpass_weight: float = 0.0,
    highpass_channels: list[int] | tuple[int, ...] | None = None,
    highpass_kernel: int = 9,
    highpass_sigma: float = 1.5,
    puncta_weight: float = 0.0,
    puncta_channels: list[int] | tuple[int, ...] | None = None,
    puncta_fraction: float = 0.03,
    puncta_kernel: int = 9,
    puncta_sigma: float = 1.5,
    puncta_temperature: float = 0.05,
    residual_weight: float = 0.0,
    residual_channels: list[int] | tuple[int, ...] | None = None,
    residual_kernel: int = 9,
    residual_sigma: float = 1.5,
) -> torch.Tensor:
    """Flow matching objective with optional image-space structure terms.

    `target_image` is required for image-space and texture terms. The predicted
    endpoint is reconstructed as `x1_pred = start + pred_velocity`, where
    `start = target_image - target_velocity`.
    """

    loss = F.mse_loss(pred_velocity, target_velocity)
    if foreground_weight > 0:
        if mask is None:
            raise ValueError("mask is required when foreground_weight > 0")
        loss = loss + float(foreground_weight) * masked_mse(
            pred_velocity,
            target_velocity,
            mask=mask,
            channel_weights=channel_weights,
        )

    if (
        image_weight <= 0
        and highpass_weight <= 0
        and puncta_weight <= 0
        and residual_weight <= 0
    ):
        return loss

    if target_image is None:
        raise ValueError("target_image is required for image-space loss terms")

    start = target_image - target_velocity
    pred_image = start + pred_velocity
    if residual_weight > 0:
        if pred_residual_velocity is None:
            raise ValueError("pred_residual_velocity is required when residual_weight > 0")
        selected = _validate_channels(
            residual_channels,
            pred_residual_velocity,
            "residual_channels",
        )
        target_residual_velocity = dog_response(
            _select_channels(target_image, selected),
            kernel_size=residual_kernel,
            sigma=residual_sigma,
        ) - dog_response(
            _select_channels(start, selected),
            kernel_size=residual_kernel,
            sigma=residual_sigma,
        )
        loss = loss + float(residual_weight) * masked_mse(
            _select_channels(pred_residual_velocity, selected),
            target_residual_velocity,
            mask=mask,
        )
    if image_weight > 0:
        loss = loss + float(image_weight) * masked_mse(
            pred_image,
            target_image,
            mask=mask,
            channel_weights=channel_weights,
        )
    if highpass_weight > 0:
        loss = loss + float(highpass_weight) * highpass_mse(
            pred_image,
            target_image,
            mask=mask,
            channels=highpass_channels,
            kernel_size=highpass_kernel,
            sigma=highpass_sigma,
        )
    if puncta_weight > 0:
        if mask is None:
            raise ValueError("mask is required when puncta_weight > 0")
        loss = loss + float(puncta_weight) * puncta_spatial_loss(
            pred_image,
            target_image,
            mask=mask,
            channels=puncta_channels,
            fraction=puncta_fraction,
            kernel_size=puncta_kernel,
            sigma=puncta_sigma,
            temperature=puncta_temperature,
        )
    return loss


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    source: torch.Tensor,
    condition: torch.Tensor,
    steps: int = 16,
    guidance_scale: float = 1.0,
    null_condition_id: int = 0,
    scaffold_mask: torch.Tensor | None = None,
    start_mode: str = "source",
    start_noise_scale: float = 0.2,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Generate perturbed images by integrating the learned velocity field."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if start_mode == "source":
        x = source
    elif start_mode == "noise":
        x = torch.randn_like(source) * float(start_noise_scale)
    elif start_mode == "source_mean":
        source_mask = image_mask(source)
        denom = source_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        channel_mean = (source * source_mask).sum(dim=(2, 3), keepdim=True) / denom
        x = channel_mean.expand_as(source)
    else:
        raise ValueError(f"Unknown sampling start_mode: {start_mode}")
    if scaffold_mask is not None:
        x = x * scaffold_mask
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t = torch.full(
            (source.shape[0],),
            idx / float(steps),
            device=source.device,
            dtype=source.dtype,
        )
        model_input = x if scaffold_mask is None else torch.cat([x, scaffold_mask], dim=1)
        v_cond, _ = combine_velocity_heads(
            model(model_input, t, condition),
            image_channels=source.shape[1],
            residual_scale=residual_scale,
        )
        if guidance_scale != 1.0:
            null_condition = torch.full_like(condition, int(null_condition_id))
            v_null, _ = combine_velocity_heads(
                model(model_input, t, null_condition),
                image_channels=source.shape[1],
                residual_scale=residual_scale,
            )
            velocity = guidance_scale * v_cond + (1.0 - guidance_scale) * v_null
        else:
            velocity = v_cond
        x = x + dt * velocity
        if scaffold_mask is not None:
            x = x * scaffold_mask
    return x
