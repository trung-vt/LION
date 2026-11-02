"""Independent Step 5 reviewer audit: frozen Step 2 noise versus new NPL draws.

For each of the 50 validation cases, compare the actual stored error y-Ax
with the Step 5 95% residual bounds and with fresh simulated norm samples.
Read-only source inputs. Never use ground truth inside the SPGL1 solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.oracle_wls.calibrate_residual_bounds import (
    sample_norm_pairs,
)
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import (
    _locked_noise,
    frozen_case,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import (
    ACQUISITION_MANIFEST_SHA256,
    WEIGHT_MANIFEST_SHA256,
    validate_frozen_archives,
)

CALIBRATION_SHA256 = "fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a"
SCHEMA = "pcm-step5-frozen-noise-review-v1"


def seed_for_case(category: str, sample_id: int):
    text = f"{SCHEMA}|independent|{category}|{sample_id}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def review(
    acquisition_root, weights_root, calibration_root, output, *, reference_draws=256
):
    acquisition_root, weights_root, calibration_root, output = map(
        Path, (acquisition_root, weights_root, calibration_root, output)
    )
    if output.exists():
        raise FileExistsError("Refusing to overwrite Step 5 reviewer audit")
    if reference_draws < 128:
        raise ValueError("Require at least 128 independent norm realizations")
    validate_frozen_archives(acquisition_root, weights_root)
    if _sha256(calibration_root / "manifest.json") != CALIBRATION_SHA256:
        raise ValueError("Frozen Step 5 calibration manifest SHA mismatch")
    acq_manifest = json.loads((acquisition_root / "manifest.json").read_text())
    weight_manifest = json.loads((weights_root / "manifest.json").read_text())
    calibration = json.loads((calibration_root / "manifest.json").read_text())
    protocol = json.loads((acquisition_root / "protocol.json").read_text())
    if protocol["noise"]["repeat_acquisitions"] != 1:
        raise ValueError("This noise audit requires exactly one acquisition repeat")
    if len(calibration["cases"]) != 50:
        raise ValueError("All 50 original 95% bounds are required")
    by_weight = {
        (r["category"], int(r["sample_id"])): r for r in weight_manifest["files"]
    }
    by_cal = {(r["category"], int(r["sample_id"])): r for r in calibration["cases"]}
    params = _locked_noise(protocol["noise"]).noise_params
    rows = []
    for index, case in enumerate(acq_manifest["acquisitions"]):
        key = case["category"], int(case["sample_id"])
        acq, _, plus, minus, clean, parts = frozen_case(
            acquisition_root,
            weights_root,
            case,
            by_weight[key],
            protocol,
        )
        if int(acq.measurement.numel()) != 13107:
            raise ValueError("Incorrect number of Hadamard measurements")
        d = 1 / np.sqrt(parts["variance"])
        e = (acq.measurement.reshape(-1) - clean).detach().numpy()
        saved_ols = float(np.linalg.norm(e))
        saved_wls = float(np.linalg.norm(d * e))
        seed = seed_for_case(*key)
        fit = sample_norm_pairs(
            plus,
            minus,
            float(acq.i0.reshape(-1)[0]),
            d,
            params,
            draws=reference_draws,
            seed=seed,
        )
        reference = by_cal[key]
        row = {
            "category": key[0],
            "sample_id": key[1],
            "source_acquisition_sha256": case["sha256"],
            "source_weight_sha256": by_weight[key]["sha256"],
            "noise_review_seed_decimal": str(seed),
        }
        for mode, actual, samples in zip(("ols", "wls"), (saved_ols, saved_wls), fit):
            sample_mean = float(samples.mean())
            sample_sd = float(samples.std(ddof=1))
            if sample_sd <= 0:
                raise ValueError("Degenerate simulated norm samples")
            row[f"saved_{mode}_norm"] = actual
            row[f"calibration_{mode}_epsilon95"] = reference[f"epsilon_{mode}_95"]
            row[f"saved_{mode}_within_calibrated_95pct"] = bool(
                actual <= reference[f"epsilon_{mode}_95"]
            )
            row[f"norm_z_{mode}"] = (actual - sample_mean) / sample_sd
            row[f"fresh_{mode}_mean"] = sample_mean
            row[f"fresh_{mode}_std"] = sample_sd
        rows.append(row)
        print(
            f"{index+1:02d}/50 {key[0]}:{key[1]:04d} "
            f"OLS_z={row['norm_z_ols']:+.3f} "
            f"WLS_z={row['norm_z_wls']:+.3f}",
            flush=True,
        )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "cases": 50,
        "source_acquisition_manifest_sha256": ACQUISITION_MANIFEST_SHA256,
        "source_weight_manifest_sha256": WEIGHT_MANIFEST_SHA256,
        "source_calibration_manifest_sha256": CALIBRATION_SHA256,
        "reference_draws_per_case": reference_draws,
        "stochastic_model": "original LION generate_noise and add_noise",
        "saved_acquisitions_regenerated": False,
        "heldout_test_used": False,
        "source_hash_sha256": _sha256(Path(__file__)),
        "python_version": platform.python_version(),
        "summary": {},
        "records": rows,
    }
    for mode in ("ols", "wls"):
        z = np.array([r[f"norm_z_{mode}"] for r in rows], dtype=float)
        counts = sum(r[f"saved_{mode}_within_calibrated_95pct"] for r in rows)
        result["summary"][mode] = {
            "count_within_95pct": counts,
            "count_total": len(rows),
            "expected_count_from_nominal_coverage": 0.95 * len(rows),
            "mean_z": float(z.mean()),
            "std_z": float(z.std(ddof=1)),
            "max_abs_z": float(np.abs(z).max()),
            "z_outside_four_sigma": int(np.sum(np.abs(z) > 4)),
        }
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "review.json", result)
    print(json.dumps(result["summary"], indent=2))
    print("REVIEW_JSON_SHA256=" + _sha256(output / "review.json"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", default=256, type=int)
    a = parser.parse_args()
    torch.set_num_threads(2)
    review(a.acquisitions, a.weights, a.calibration, a.output, reference_draws=a.draws)


if __name__ == "__main__":
    main()
