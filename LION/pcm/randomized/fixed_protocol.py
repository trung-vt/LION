"""Deterministic PCM validation/test acquisitions using frozen NPL-inspired settings.

All experiments share mask and noise realization for a case; nothing depends
on the reconstruction model, validation metric or unseen image performance.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from LION.pcm.noise import PCMNoiseParameter, PCMNoiseSimulation
from LION.pcm.randomized.noise import AR, MA, sample_pattern_indices
from LION.pcm.simulation import PCMAcquisition, simulate_pcm_acquisition

SAMPLING_RATIOS = (0.10, 0.20, 0.50)
NOISE_SCENARIOS = ("noiseless", "gaussian_dark", "pcm_default_10ms", "pcm_short_100us")
CURRENT_TO_MODEL_SCALE = 1.0 / 3.5e-8  # Original LION NPL PCM preset
SEED_BY_SPLIT = {"validation": 1_300_000, "test": 2_300_000, "unseen": 3_300_000}


def fixed_noise_config(scenario: str, j_order: int) -> PCMNoiseSimulation | None:
    if scenario not in NOISE_SCENARIOS:
        raise ValueError(f"Unknown noise scenario: {scenario}")
    if scenario == "noiseless":
        return None
    default = PCMNoiseSimulation.npl_default(
        current_to_model_scale=CURRENT_TO_MODEL_SCALE, j_order=j_order
    )
    if scenario == "pcm_default_10ms":
        return default
    if scenario == "pcm_short_100us":
        return PCMNoiseSimulation.npl_extrapolate(
            current_to_model_scale=CURRENT_TO_MODEL_SCALE, j_order=j_order
        )
    dark_only = PCMNoiseParameter(
        beam_ar=AR,
        beam_ma=MA,
        beam_sigma=0.0,
        bias_sigma=0.0,
        bias_gradient_sigma=0.0,
        dark_sigma=default.noise_params.dark_sigma,
    )
    return PCMNoiseSimulation(
        name="Gaussian-only matched dark noise",
        noise_params=dark_only,
        signal_to_bias_ratio=default.signal_to_bias_ratio,
        noisy_measurement_filename="",
        ground_truth_filename="",
        rng_seed=0,
    )


def fixed_coarse_j(j_order: int) -> int:
    """Existing presets use coarseJ=J-2 (six at image side 256)."""
    if j_order < 5:
        raise ValueError("Expected J>=5")
    return j_order - 2


def fixed_acquisition(
    clean: torch.Tensor,
    *,
    split: str,
    case_index: int,
    ratio: float,
    scenario: str,
) -> tuple[PCMAcquisition, dict]:
    if split not in SEED_BY_SPLIT:
        raise ValueError(split)
    if ratio not in SAMPLING_RATIOS:
        raise ValueError(f"Requested rate {ratio} is not in frozen protocol")
    if (
        clean.ndim != 4
        or clean.shape[:2] != (1, 1)
        or clean.shape[-1] != clean.shape[-2]
    ):
        raise ValueError(f"Expected one square grayscale image, got {clean.shape}")
    j = clean.shape[-1].bit_length() - 1
    if 2**j != clean.shape[-1]:
        raise ValueError("Image side must be a power of two")
    if not 0 <= case_index < 10000:
        raise ValueError("Invalid case index")
    seed = SEED_BY_SPLIT[split] + 10_000 * case_index + int(round(ratio * 1000))
    mask_rng = np.random.default_rng(seed)
    indices, pattern = sample_pattern_indices(
        clean.shape[-1], mask_rng, ratio=ratio, coarse_j=fixed_coarse_j(j)
    )
    # Independent RNG stream ensures *identical masks* across noise scenarios.
    noise_rng = np.random.default_rng(seed + 8_000_000)
    noise = fixed_noise_config(scenario, j)
    acquisition = simulate_pcm_acquisition(
        clean,
        j_order=j,
        sampled_indices=indices,
        noise=noise,
        rng=noise_rng,
        device=clean.device,
        seed=seed,
    )
    params = asdict(noise.noise_params) if noise is not None else None
    mask_sha = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
    residual = acquisition.measurement - acquisition.op(clean)
    return acquisition, {
        "split": split,
        "case_index": case_index,
        "seed_mask": seed,
        "seed_noise": seed + 8_000_000,
        "scenario": scenario,
        "noise_params": params,
        "signal_to_bias_ratio": noise.signal_to_bias_ratio
        if noise is not None
        else None,
        "sampling": pattern,
        "mask_sha256": mask_sha,
        "realized_measurement_residual_rms": float(torch.sqrt((residual**2).mean())),
    }


def protocol_summary(j_order: int) -> dict:
    return {
        "j_order": j_order,
        "sampling_ratios": SAMPLING_RATIOS,
        "coarse_j": fixed_coarse_j(j_order),
        "noise_scenarios": {
            s: asdict(n.noise_params) if (n := fixed_noise_config(s, j_order)) else None
            for s in NOISE_SCENARIOS
        },
        "signal_to_bias_ratio": 0.3,
        "current_to_model_scale": CURRENT_TO_MODEL_SCALE,
        "rng_seed_bases": SEED_BY_SPLIT,
        "mask_policy": "independent deterministic mask RNG per case/rate, shared across noise conditions",
        "metric_policy": "No training or tuning on unseen or test cases",
        "limits": "Normalized image display scales not independently ampere-calibrated; scenarios are NPL-inspired simulated acquisitions",
    }
