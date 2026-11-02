"""Independent Step 6 numeric cross-checks.

This module does not import or call the experiment's metrics_for_image,
summarize_results, or _summarize_delta. MSE and PSNR use NumPy formulas,
SSIM is compared with the independent scikit-image reference as well as the
underlying torchmetrics functional, and HaarPSI calls piq directly.

The numerical parity checks and paired-statistic regeneration are intended
as independent guards against an error shared by the experiment and verifier.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from piq import haarpsi as piq_haarpsi
from skimage.metrics import structural_similarity as skimage_ssim
from torchmetrics.functional.image import structural_similarity_index_measure

CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
MODES = ("ols", "wls")
METRICS = ("mse", "psnr", "ssim", "haarpsi")
COMPARISONS = METRICS + ("runtime_seconds", "iterations", "seconds_per_iteration")
BASE_SEED = 20261102


def numerical_metrics(image: np.ndarray, truth: np.ndarray):
    """Recompute the four stored metrics without the experiment's wrapper.

    MSE/PSNR are fully independent NumPy calculations. For SSIM, reproduce
    the project's underlying torchmetrics call, plus an independent
    scikit-image sanity check (not expected to be numerically identical).
    HaarPSI is independently invoked through piq with the original clipping.
    """
    im = np.asarray(image)
    gt = np.asarray(truth)
    if im.shape != (256, 256) or gt.shape != (256, 256):
        raise ValueError("Independent metrics require a matched 256x256 image")
    if im.dtype != np.float32 or not np.isfinite(im).all() or not np.isfinite(gt).all():
        raise ValueError("Invalid metric input or non-float32 reconstruction")
    im64 = im.astype(np.float64)
    gt32 = gt.astype(np.float32)
    gt64 = gt32.astype(np.float64)
    mse = float(np.mean((im64 - gt64) ** 2))
    if mse <= 0:
        psnr = math.inf
    else:
        psnr = float(10 * np.log10(1.0 / mse))

    # Match the existing published LION metric implementation, without
    # importing its wrapper or using the experiment's metrics function.
    tx = torch.from_numpy(im.copy()).reshape(1, 1, 256, 256)
    ty = torch.from_numpy(gt32.copy()).reshape(1, 1, 256, 256)
    ssim = float(
        structural_similarity_index_measure(
            tx, ty, data_range=1.0, gaussian_kernel=False, kernel_size=7
        )
    )
    reference_ssim = float(
        skimage_ssim(im64, gt64, data_range=1.0, win_size=7, gaussian_weights=False)
    )
    # The scikit-image SSIM is an independent contextual check, not a
    # numerical identity: boundary treatment can differ substantially for
    # sharp PCM edges. Do not reject a valid archive on this alone.
    haar = float(piq_haarpsi(tx.clamp(0.0, 1.0), ty, data_range=1.0))
    result = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "haarpsi": haar,
        "ssim_skimage_reference": reference_ssim,
        "ssim_skimage_difference": ssim - reference_ssim,
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Independent metrics contain non-finite values")
    return result


def _independent_delta(values, metric):
    data = np.array(values, dtype=np.float64)
    if not len(data):
        return None
    rng = np.random.default_rng(BASE_SEED + COMPARISONS.index(metric))
    means = np.mean(data[rng.integers(0, len(data), size=(10000, len(data)))], axis=1)
    better_up = metric not in (
        "mse",
        "runtime_seconds",
        "iterations",
        "seconds_per_iteration",
    )
    favourable = data > 0 if better_up else data < 0
    return {
        "n": int(len(data)),
        "mean_delta_wls_minus_ols": float(np.mean(data)),
        "std_delta_sample": float(np.std(data, ddof=1)) if len(data) > 1 else None,
        "median_delta": float(np.median(data)),
        "win_count_wls": int(np.count_nonzero(favourable)),
        "tie_count": int(np.count_nonzero(data == 0)),
        "win_rate_wls": float(np.mean(favourable)),
        "mean_delta_bootstrap95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def independent_summaries(records):
    """Rebuild the Step 6 nested summary directly from original raw records."""
    output = {}
    for category in ("ALL", *CATEGORIES):
        cases = [
            record
            for record in records
            if category == "ALL" or record["category"] == category
        ]
        method_means = {}
        for mode in MODES:
            method_means[mode] = {}
            for metric in COMPARISONS:
                values = []
                for case in cases:
                    item = case["methods"][mode]
                    value = (
                        item.get("metrics", {}).get(metric)
                        if metric in METRICS
                        else item.get(metric)
                    )
                    if value is not None:
                        values.append(float(value))
                method_means[mode][metric] = float(np.mean(values)) if values else None
        differences = {}
        for metric in COMPARISONS:
            arr = []
            for case in cases:
                m0 = case["methods"]["ols"]
                m1 = case["methods"]["wls"]
                v0 = (
                    m0.get("metrics", {}).get(metric)
                    if metric in METRICS
                    else m0.get(metric)
                )
                v1 = (
                    m1.get("metrics", {}).get(metric)
                    if metric in METRICS
                    else m1.get(metric)
                )
                if v0 is not None and v1 is not None:
                    arr.append(float(v1 - v0))
            differences[metric] = _independent_delta(arr, metric)
        output[category] = {
            "total_cases": len(cases),
            "fully_successful": sum(
                r["methods"]["ols"]["status"] == "PASS"
                and r["methods"]["wls"]["status"] == "PASS"
                for r in cases
            ),
            "ols_solver_issues": sum(
                r["methods"]["ols"]["status"] != "PASS" for r in cases
            ),
            "wls_solver_issues": sum(
                r["methods"]["wls"]["status"] != "PASS" for r in cases
            ),
            "metric_differences": differences,
            "method_means": method_means,
        }
    return output
