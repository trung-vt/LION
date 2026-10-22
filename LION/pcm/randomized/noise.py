"""Randomized noise for image denoising and physical PCM acquisition.

Image noise is a raster-domain surrogate, NOT a PCM measurement simulator.
PCM acquisition delegates to the existing split-pattern LION implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.noise import PCMNoiseParameter, PCMNoiseSimulation
from LION.pcm.simulation import PCMAcquisition, simulate_pcm_acquisition
from LION.pcm_tools.noise_generate import generate_noise
from LION.pcm_tools.pcm_sampling import uniform_random_sample

AR = (-0.9221, -0.1709, -0.0769, -0.0354)
MA = (0.0377, -0.3669)


@dataclass(frozen=True)
class NoiseRange:
    sigma_g_max: float = 3e-7
    sigma_bias_max: float = 4e-5
    sigma_beam_max: float = 1e-3
    sigma_dark_max: float = 1.0
    signal_to_bias_ratio: float = 0.3
    gaussian_probability: float = 0.5

    def __post_init__(self) -> None:
        for value in (
            self.sigma_g_max,
            self.sigma_bias_max,
            self.sigma_beam_max,
            self.sigma_dark_max,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Noise standard-deviation bounds must be finite and nonnegative"
                )
        if self.signal_to_bias_ratio <= 0:
            raise ValueError("signal_to_bias_ratio must be positive")
        if not 0 <= self.gaussian_probability <= 1:
            raise ValueError("gaussian_probability must lie in [0, 1]")


def decimal_interval(value: float) -> tuple[float, float]:
    """Enclose a coefficient between its floor and ceiling at two decimals."""
    number = Decimal(str(value))
    return (
        float(number.quantize(Decimal("0.01"), rounding=ROUND_FLOOR)),
        float(number.quantize(Decimal("0.01"), rounding=ROUND_CEILING)),
    )


def sample_parameters(
    rng: np.random.Generator, bounds: NoiseRange
) -> PCMNoiseParameter:
    """Draw stationary AR and invertible MA increments; integration order stays d=1."""
    for _ in range(1000):
        ar = tuple(float(rng.uniform(*decimal_interval(x))) for x in AR)
        ma = tuple(float(rng.uniform(*decimal_interval(x))) for x in MA)
        process = ArmaProcess(np.r_[1.0, -np.asarray(ar)], np.r_[1.0, ma])
        if process.isstationary and process.isinvertible:
            break
    else:
        raise RuntimeError("No stable/invertible ARMA coefficients were sampled")
    return PCMNoiseParameter(
        beam_ar=ar,
        beam_ma=ma,
        beam_sigma=float(rng.uniform(0, bounds.sigma_beam_max)),
        bias_sigma=float(rng.uniform(0, bounds.sigma_bias_max)),
        bias_gradient_sigma=float(rng.uniform(0, bounds.sigma_g_max)),
        dark_sigma=float(rng.uniform(0, bounds.sigma_dark_max)),
    )


def parameter_dict(p: PCMNoiseParameter) -> dict:
    return {
        "sigma_g": p.bias_gradient_sigma,
        "sigma_bias": p.bias_sigma,
        "sigma_beam": p.beam_sigma,
        "sigma_dark": p.dark_sigma,
        "ar": list(p.beam_ar),
        "ma": list(p.beam_ma),
        "d": 1,
    }


def image_corruption(
    clean: torch.Tensor,
    rng: np.random.Generator,
    bounds: NoiseRange,
    *,
    mode: str = "mixture",
    parameters: PCMNoiseParameter | None = None,
) -> tuple[torch.Tensor, float, dict]:
    """No Hadamard transform or subsampling. Use C-order raster pixel indices.

    z_t = x_t (1+b_t) + I_bar*t*g + I_bar*eta_t + delta_t,
    with I_bar = mean(x)/signal_to_bias_ratio and the constant offset removed.
    Inputs are never clipped. Conditioning uses dark/Gaussian SD, not oracle RMS.
    """
    if clean.ndim != 4 or clean.shape[:2] != (1, 1):
        raise ValueError("Expected one grayscale image shaped (1,1,H,W)")
    if not torch.isfinite(clean).all():
        raise ValueError("Clean image contains non-finite values")
    if mode not in ("mixture", "gaussian", "pcm_like"):
        raise ValueError(f"Unknown corruption mode {mode}")
    p = parameters or sample_parameters(rng, bounds)
    gaussian = mode == "gaussian" or (
        mode == "mixture" and rng.random() < bounds.gaussian_probability
    )
    if gaussian:
        residual = rng.standard_normal(tuple(clean.shape)) * float(p.dark_sigma)
        noisy = clean + torch.as_tensor(
            residual, device=clean.device, dtype=clean.dtype
        )
        kind = "gaussian"
    else:
        i_bar = float(clean.mean()) / bounds.signal_to_bias_ratio
        dark, bias, beam = generate_noise(clean.numel(), i_bar, p, rng)
        beam_t = torch.as_tensor(
            beam, device=clean.device, dtype=clean.dtype
        ).reshape_as(clean)
        offset = torch.as_tensor(
            dark + bias - i_bar, device=clean.device, dtype=clean.dtype
        ).reshape_as(clean)
        noisy = clean * (1.0 + beam_t) + offset
        kind = "pcm_like_raster_surrogate"
    if not torch.isfinite(noisy).all():
        raise FloatingPointError("Non-finite randomized image corruption")
    return (
        noisy,
        float(p.dark_sigma),
        parameter_dict(p)
        | {
            "kind": kind,
            "residual_rms": float((noisy - clean).square().mean().sqrt()),
            "image_mean": float(clean.mean()),
        },
    )


def valid_coarse_levels(j: int, measurements: int) -> list[int]:
    """Include truncated coarse levels only when their preceding level fits."""
    if j < 1 or not 1 <= measurements <= 4**j:
        raise ValueError("Invalid grid order or measurement count")
    return [0] + [c for c in range(1, j + 1) if 4 ** (c - 1) < measurements]


def sample_pattern_indices(
    side: int,
    rng: np.random.Generator,
    *,
    ratio: float | None = None,
    coarse_j: int | None = None,
) -> tuple[np.ndarray, dict]:
    if side < 16 or side & (side - 1):
        raise ValueError("Image side must be a power of two and at least 16")
    j = side.bit_length() - 1
    ratio = float(rng.uniform(0.01, 1.0)) if ratio is None else float(ratio)
    if not 0.01 <= ratio <= 1:
        raise ValueError("Sampling ratio must lie in [0.01, 1]")
    m = max(2, min(side * side, int(np.ceil(ratio * side * side))))
    choices = valid_coarse_levels(j, m)
    coarse = int(rng.choice(choices)) if coarse_j is None else int(coarse_j)
    if coarse not in choices:
        raise ValueError(
            f"coarseJ={coarse} invalid for {m} measurements; valid={choices}"
        )
    indices = uniform_random_sample(J=j, num_samples=m, coarse_J=coarse, rng=rng)
    indices = np.sort(indices).astype(np.int64)
    if len(indices) != m or len(np.unique(indices)) != m or indices[0] != 0:
        raise RuntimeError(
            "Sampling must return the requested unique row count and retain DC"
        )
    return indices, {
        "requested_ratio": ratio,
        "realized_ratio": m / (side * side),
        "coarse_j": coarse,
        "measurements": m,
        "side": side,
        "sampling_scheme": "coarse_block_uniform_tail",
    }


def randomized_acquisition(
    clean: torch.Tensor,
    rng: np.random.Generator,
    bounds: NoiseRange,
    *,
    ratio: float | None = None,
    coarse_j: int | None = None,
    parameters: PCMNoiseParameter | None = None,
) -> tuple[PCMAcquisition, dict]:
    if (
        clean.ndim != 4
        or clean.shape[:2] != (1, 1)
        or clean.shape[-2] != clean.shape[-1]
    ):
        raise ValueError("Expected one square grayscale image")
    indices, record = sample_pattern_indices(
        clean.shape[-1], rng, ratio=ratio, coarse_j=coarse_j
    )
    p = parameters or sample_parameters(rng, bounds)
    noise = PCMNoiseSimulation(
        name="randomized_physical_pcm",
        noise_params=p,
        signal_to_bias_ratio=bounds.signal_to_bias_ratio,
        noisy_measurement_filename="",
        ground_truth_filename="",
        rng_seed=0,
    )
    acquisition = simulate_pcm_acquisition(
        clean,
        j_order=clean.shape[-1].bit_length() - 1,
        sampled_indices=indices,
        noise=noise,
        rng=rng,
        device=clean.device,
    )
    residual = acquisition.measurement - acquisition.op(clean)
    if not torch.isfinite(acquisition.measurement).all():
        raise FloatingPointError("Non-finite PCM acquisition")
    record.update(parameter_dict(p))
    record.update(
        kind="physical_pcm", residual_rms=float(residual.square().mean().sqrt())
    )
    return acquisition, record


def split_ssdu(
    acquisition: PCMAcquisition,
    rng: np.random.Generator,
    heldout_fraction: float = 0.2,
) -> tuple[PhotocurrentMapOp, torch.Tensor, PhotocurrentMapOp, torch.Tensor, dict]:
    """Split already-generated rows; keep DC on the input side; do not regenerate noise.

    Correlated noise across the partition is NOT asserted independent or unbiased.
    """
    rows = acquisition.sampled_indices
    if len(rows) < 2 or not 0 < heldout_fraction < 1:
        raise ValueError(
            "SSDU needs at least two measurements and a proper holdout fraction"
        )
    eligible = np.flatnonzero(rows != 0)
    n_loss = min(len(eligible), max(1, int(round(len(rows) * heldout_fraction))))
    loss_pos = np.sort(rng.choice(eligible, n_loss, replace=False))
    train_pos = np.setdiff1d(np.arange(len(rows)), loss_pos)
    y = acquisition.measurement
    j = acquisition.op.domain_shape[-1].bit_length() - 1
    train_op = PhotocurrentMapOp(
        j_order=j, s_indices=rows[train_pos], normalized=True, device=y.device
    )
    loss_op = PhotocurrentMapOp(
        j_order=j, s_indices=rows[loss_pos], normalized=True, device=y.device
    )
    train_y = y.index_select(-1, torch.as_tensor(train_pos, device=y.device))
    loss_y = y.index_select(-1, torch.as_tensor(loss_pos, device=y.device))
    return (
        train_op,
        train_y,
        loss_op,
        loss_y,
        {
            "input_rows": rows[train_pos].tolist(),
            "loss_rows": rows[loss_pos].tolist(),
        },
    )
