"""Step 5: calibrate separate 95% OLS and oracle WLS residual bounds.

Each frozen validation image uses fixed x,A and independent fresh NPL noise
realizations. Both residual norms use the SAME sampled noise in each draw.
Calibration and held-out coverage checks have distinct deterministic RNG
streams, independent of Steps 2, 3 and 4.

The source PNG is used ONLY for statistical calibration, never for solving.
All 50 original acquisition and weight files remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.oracle_wls.build_oracle_weights import verify_weights
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    _write_atomic_new,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import (
    _locked_noise,
    frozen_case,
    one_realization,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import CALIBRATION_SCHEMA

TARGET_COVERAGE = 0.95
SEED_NAMESPACE = "pcm-step5-q95-independent-20261102-v1"
# Engineering check on independent coverage. At 1024 validation draws, the
# binomial reference SE at 95% coverage is about 0.0068 (not a 95% CI).
MIN_HOLDOUT_COVERAGE = 0.91
MAX_HOLDOUT_COVERAGE = 0.99


def calibration_seed(category: str, sample_id: int, stage: str) -> int:
    """Separate from Step 2 acquisition, Step 4 MC and between stages."""
    if stage not in ("fit", "holdout"):
        raise ValueError("stage must be fit or holdout")
    text = f"{SEED_NAMESPACE}|{category}|{sample_id:04d}|{stage}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def sample_norm_pairs(
    clean_plus: torch.Tensor,
    clean_minus: torch.Tensor,
    i0: float,
    inverse_std: np.ndarray,
    params,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure L2 and standardized L2 errors using identical noise draws."""
    if draws < 2:
        raise ValueError("At least two noise realizations are required")
    d = np.asarray(inverse_std, dtype=np.float64)
    if d.ndim != 1 or d.size != clean_plus.numel():
        raise ValueError("Inverse STD must match the original sampling order")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("Invalid inverse noise standard deviations")
    rng = np.random.default_rng(seed)
    ols = np.empty(draws, dtype=np.float64)
    wls = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        e = one_realization(clean_plus, clean_minus, i0, params, rng)
        ols[i] = np.linalg.norm(e)
        wls[i] = np.linalg.norm(e * d)
    if not (np.isfinite(ols).all() and np.isfinite(wls).all()):
        raise ValueError("Non-finite residual norm samples")
    return ols, wls


def empirical_upper_quantile(values: np.ndarray, probability: float = TARGET_COVERAGE):
    """Upper empirical quantile; deterministic rank selection (method higher)."""
    if not 0 < probability < 1 or values.ndim != 1 or values.size < 2:
        raise ValueError("Invalid noise quantile request")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite empirical noise norms")
    return float(np.quantile(values, probability, method="higher"))


def calibrate_case(
    case: dict,
    weight: dict,
    source,
    weights,
    protocol,
    params,
    *,
    fit_draws: int,
    holdout_draws: int,
):
    """Calibrate two bounds with matched draws and test independent coverage."""
    acq, x, plus, minus, clean, components = frozen_case(
        source,
        weights,
        case,
        weight,
        protocol,
    )
    v = components["variance"]
    inverse_std64 = 1.0 / np.sqrt(v)
    i0 = float(acq.i0.reshape(-1)[0])
    category, sample_id = case["category"], int(case["sample_id"])
    fit_seed = calibration_seed(category, sample_id, "fit")
    holdout_seed = calibration_seed(category, sample_id, "holdout")
    ols_fit, wls_fit = sample_norm_pairs(
        plus,
        minus,
        i0,
        inverse_std64,
        params,
        draws=fit_draws,
        seed=fit_seed,
    )
    eps_ols = empirical_upper_quantile(ols_fit)
    eps_wls = empirical_upper_quantile(wls_fit)
    ols_hold, wls_hold = sample_norm_pairs(
        plus,
        minus,
        i0,
        inverse_std64,
        params,
        draws=holdout_draws,
        seed=holdout_seed,
    )
    coverage_ols = float(np.mean(ols_hold <= eps_ols))
    coverage_wls = float(np.mean(wls_hold <= eps_wls))
    if not (eps_ols > 0 and eps_wls > 0):
        raise ValueError("Calibrated positive residual bounds are required")
    return {
        "category": category,
        "sample_id": sample_id,
        "source_acquisition_sha256": case["sha256"],
        "source_weight_sha256": weight["sha256"],
        "fit_seed_decimal": str(fit_seed),
        "holdout_seed_decimal": str(holdout_seed),
        "fit_draws": fit_draws,
        "holdout_draws": holdout_draws,
        "epsilon_ols_95": eps_ols,
        "epsilon_wls_95": eps_wls,
        "epsilon_ols_reference": float(np.sqrt(v.sum())),
        "epsilon_wls_reference": float(np.sqrt(v.size)),
        "holdout_coverage_ols": coverage_ols,
        "holdout_coverage_wls": coverage_wls,
        "holdout_ols_within_gate": (
            MIN_HOLDOUT_COVERAGE <= coverage_ols <= MAX_HOLDOUT_COVERAGE
        ),
        "holdout_wls_within_gate": (
            MIN_HOLDOUT_COVERAGE <= coverage_wls <= MAX_HOLDOUT_COVERAGE
        ),
        "quantile_method": "numpy.quantile(method='higher')",
        "factor": 1.0,
        "residual_definition": "L2 of (y-Ax) or D(y-Ax) in float64",
    }


def build_calibration(
    acquisition_root,
    weight_root,
    output,
    *,
    expected_acquisition_manifest_sha256,
    expected_weight_manifest_sha256,
    fit_draws=2048,
    holdout_draws=1024,
    limit_cases=None,
):
    """New independent archive for all 50, with explicit source hashes."""
    source, weights, dest = map(Path, (acquisition_root, weight_root, output))
    if dest.exists():
        raise FileExistsError("Refusing to overwrite residual calibration")
    if fit_draws < 512 or holdout_draws < 256:
        raise ValueError("Too few samples for a 95% residual calibration")
    verify_weights(
        source,
        weights,
        expected_acquisition_manifest_sha256=expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=expected_weight_manifest_sha256,
    )
    acquisition_manifest = json.loads((source / "manifest.json").read_text())
    weight_manifest = json.loads((weights / "manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    cases = acquisition_manifest["acquisitions"]
    if len(cases) != 50 or acquisition_manifest["test_count"] != 0:
        raise ValueError("Step 5 calibration expects the frozen validation split")
    if limit_cases is not None:
        if not (1 <= limit_cases <= 50):
            raise ValueError("limit_cases must be within 1..50")
        cases = cases[:limit_cases]
    by_weight = {
        (r["category"], int(r["sample_id"])): r for r in weight_manifest["files"]
    }
    params = _locked_noise(protocol["noise"]).noise_params
    dest.mkdir(parents=True, exist_ok=False)
    records = []
    for count, case in enumerate(cases, start=1):
        key = (case["category"], int(case["sample_id"]))
        rec = calibrate_case(
            case,
            by_weight[key],
            source,
            weights,
            protocol,
            params,
            fit_draws=fit_draws,
            holdout_draws=holdout_draws,
        )
        records.append(rec)
        print(
            f"{count}/{len(cases)} {key}: eps_OLS={rec['epsilon_ols_95']:.5f}, "
            f"eps_WLS={rec['epsilon_wls_95']:.5f}, "
            f"coverage_OLS={rec['holdout_coverage_ols']:.3f}, "
            f"coverage_WLS={rec['holdout_coverage_wls']:.3f}",
            flush=True,
        )
    method_coverage = {
        mode: float(np.mean([r[f"holdout_coverage_{mode}"] for r in records]))
        for mode in ("ols", "wls")
    }
    gate_pass = all(
        r["holdout_ols_within_gate"] and r["holdout_wls_within_gate"] for r in records
    )
    manifest = {
        "schema": CALIBRATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target_quantile": TARGET_COVERAGE,
        "calibration_draws_per_case": fit_draws,
        "independent_holdout_draws_per_case": holdout_draws,
        "number_of_cases": len(records),
        "acquisition_manifest_sha256": expected_acquisition_manifest_sha256,
        "weight_manifest_sha256": expected_weight_manifest_sha256,
        "source_acquisition_protocol_sha256": _sha256(source / "protocol.json"),
        "source_weight_protocol_sha256": _sha256(weights / "protocol.json"),
        "fit_and_holdout_independent": True,
        "ols_and_wls_use_same_noise_draws": True,
        "original_measurements_reused_for_both_reconstructions": True,
        "wavelet": "db4",
        "solver_adapter_dtype": "float32",
        "noise_calibration_dtype": "float64",
        "mean_holdout_coverage": method_coverage,
        "all_per_case_coverage_in_predeclared_gate": bool(gate_pass),
        "predeclared_coverage_gate": [MIN_HOLDOUT_COVERAGE, MAX_HOLDOUT_COVERAGE],
        "threshold_caveat": (
            "Finite empirical 95% quantile for each image under known ground truth "
            "and simulated noise. Not an exact probability guarantee; Step 6 "
            "will perform the matched reconstruction benchmark."
        ),
        "code_sha256": _sha256(Path(__file__)),
        "cases": records,
    }
    write_json_new(dest / "manifest.json", manifest)
    columns = list(records[0].keys())
    with (dest / "summary.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=columns)
        w.writeheader()
        w.writerows(records)
    contents = "".join(
        f"{_sha256(dest / filename)}  {filename}\n"
        for filename in ("manifest.json", "summary.csv")
    )
    _write_atomic_new(dest / "SHA256SUMS", lambda f: f.write(contents.encode()))
    print(f"EXTERNAL_STEP5_CALIBRATION_SHA256={_sha256(dest / 'manifest.json')}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-acquisition-manifest-sha256", required=True)
    parser.add_argument("--expected-weight-manifest-sha256", required=True)
    parser.add_argument("--fit-draws", type=int, default=2048)
    parser.add_argument("--holdout-draws", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = build_calibration(
        args.acquisitions,
        args.weights,
        args.output,
        expected_acquisition_manifest_sha256=args.expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=args.expected_weight_manifest_sha256,
        fit_draws=args.fit_draws,
        holdout_draws=args.holdout_draws,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
