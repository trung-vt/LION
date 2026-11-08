"""Measurement-only PCM SSDU split and risk utilities.

The public entry points deliberately do not receive or load ground-truth.
No candidate ranking may use PSNR, SSIM, HaarPSI or test-set measurements.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    COARSE_COUNT,
    MEASUREMENT_COUNT,
    CATEGORIES,
)


def read_blind_acquisition(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only archived y and sampled Hadamard indices, never an image."""
    with np.load(path, allow_pickle=False) as archive:
        if "ground_truth" in archive.files or "target" in archive.files:
            raise ValueError("SSDU tuning input must not contain a reference image")
        y = np.array(archive["measurement"], copy=True)
        ids = np.array(archive["sampled_indices"], copy=True)
    if y.shape != (1, 1, MEASUREMENT_COUNT) or y.dtype != np.float64:
        raise ValueError("Expected frozen float64 NPL PCM acquisition")
    if ids.shape != (MEASUREMENT_COUNT,) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("Expected 13,107 ordered Hadamard row indices")
    if not np.array_equal(ids[:COARSE_COUNT], np.arange(COARSE_COUNT)):
        raise ValueError("The fully acquired coarse block is missing")
    return y, ids.astype(np.int64)


def make_fold_inputs(
    measurement: np.ndarray,
    sampled_indices: np.ndarray,
    heldout_positions: np.ndarray,
    device: torch.device | str,
) -> tuple[PhotocurrentMapOp, torch.Tensor, PhotocurrentMapOp, torch.Tensor]:
    """Create disjoint training/validation PCM operators with unchanged normalization."""
    idx = np.asarray(sampled_indices, dtype=np.int64)
    held = np.asarray(heldout_positions, dtype=np.int64)
    n = len(idx)
    if measurement.shape != (1, 1, n):
        raise ValueError("Measurement and Hadamard rows are not paired")
    if held.ndim != 1 or len(held) == 0 or np.any(np.diff(held) <= 0):
        raise ValueError("Heldout positions must be nonempty, sorted and unique")
    if held[0] < COARSE_COUNT or held[-1] >= n:
        raise ValueError("Coarse training patterns or out-of-bounds positions held out")
    fitted = np.setdiff1d(np.arange(n), held, assume_unique=True)
    if len(fitted) + len(held) != n or np.intersect1d(fitted, held).size:
        raise AssertionError("Fit and validation positions overlap")
    if not np.array_equal(fitted[:COARSE_COUNT], np.arange(COARSE_COUNT)):
        raise AssertionError("Coarse patterns must remain in fit subset")
    op_fit = PhotocurrentMapOp(
        j_order=8, s_indices=idx[fitted], normalized=True, device=device
    )
    op_val = PhotocurrentMapOp(
        j_order=8, s_indices=idx[held], normalized=True, device=device
    )
    fit_y = torch.as_tensor(
        np.ascontiguousarray(measurement[..., fitted]),
        device=device,
        dtype=torch.float32,
    )
    val_y = torch.as_tensor(
        np.ascontiguousarray(measurement[..., held]), device=device, dtype=torch.float64
    )
    return op_fit, fit_y, op_val, val_y


@torch.no_grad()
def heldout_mse(
    reconstructed_image: torch.Tensor,
    validation_operator: PhotocurrentMapOp,
    hidden_measurements: torch.Tensor,
) -> float:
    """Only held-out measurement prediction risk; no image-space metrics."""
    predicted = validation_operator(reconstructed_image)
    if predicted.shape != hidden_measurements.shape:
        raise ValueError("Prediction and held-out measurement shapes differ")
    if (
        not torch.isfinite(predicted).all()
        or not torch.isfinite(hidden_measurements).all()
    ):
        raise FloatingPointError("Nonfinite hidden measurement or prediction")
    loss = (
        (predicted.to(torch.float64) - hidden_measurements.to(torch.float64))
        .square()
        .mean()
    )
    return float(loss.item())


def aggregate_case_losses(
    cases: list[dict],
    num_folds: int = 4,
) -> dict:
    """Equal-fold, equal-image, equal-category SSDU measurement-only objective."""
    per_cat = defaultdict(list)
    for item in cases:
        cat = item["category"]
        if cat not in CATEGORIES:
            raise ValueError("Unexpected category")
        fold_scores = np.asarray(item["fold_losses"], dtype=np.float64)
        if fold_scores.shape != (num_folds,) or not np.isfinite(fold_scores).all():
            raise ValueError("Need finite SSDU loss from each fold")
        if np.any(fold_scores < 0):
            raise ValueError("Squared residual cannot be negative")
        per_cat[cat].append(float(fold_scores.mean()))
    if set(per_cat) != set(CATEGORIES):
        raise ValueError("All five categories must be included")
    counts = {c: len(per_cat[c]) for c in CATEGORIES}
    if len(set(counts.values())) != 1:
        raise ValueError("Dataset must be balanced for 5-vs-50 comparison")
    category_means = {c: float(np.mean(per_cat[c])) for c in CATEGORIES}
    return {
        "loss": float(np.mean(list(category_means.values()))),
        "category_means": category_means,
        "cases_per_category": counts,
        "num_cases": len(cases),
        "folds": num_folds,
    }
