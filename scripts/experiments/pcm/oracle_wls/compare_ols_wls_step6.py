"""Step 6 paired validation: OLS vs oracle WLS SPGL1-BPDN on frozen PCM y/A.

10-image pilot (two per category), followed by remaining 40 validation
images. Exactly the same saved y, mask, unchanged db4 wavelet and SPGL1
settings are used for both methods; only the diagonal whitener and its
pre-calibrated 95% residual bound differ.

No new noise acquisitions, 450-case held-out images, training or shared
numerical-code modifications are involved. Record failures instead of
dropping cases. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from LION.utils.metrics import compute_metrics_2d
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    _write_atomic_new,
    load_png_float64,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import (
    ACQUISITION_MANIFEST_SHA256,
    WEIGHT_MANIFEST_SHA256,
    load_calibration_record,
    load_step5_case,
    solve_step5_case,
    validate_frozen_archives,
)

SCHEMA = "pcm-step6-paired-ols-wls-validation-v1"
CALIBRATION_MANIFEST_SHA256 = (
    "fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a"
)
ORDERED_CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
METHODS = ("ols", "wls")
METRICS = ("mse", "psnr", "ssim", "haarpsi")
COMPARISONS = METRICS + ("runtime_seconds", "iterations", "seconds_per_iteration")
BOOTSTRAP_SEED = 20261102


def metrics_for_image(reconstructed: torch.Tensor, ground_truth: torch.Tensor):
    """Use the project's original compute_metrics_2d convention.

    MSE/PSNR/SSIM operate on raw float32 reconstructions and data_range=1.0.
    HaarPSI clips the reconstruction to [0,1] internally (as in the
    original LION/utils/metrics.py), not the image used for other metrics.
    """
    if reconstructed.ndim != 2 or ground_truth.shape != (1, 1, *reconstructed.shape):
        raise ValueError(
            "Expected matched one-image 2D ground truth and reconstruction"
        )
    target = ground_truth.to(dtype=torch.float32).cpu()
    reconstructed = reconstructed.detach().to(dtype=torch.float32).cpu()
    values = compute_metrics_2d(
        reconstructed.unsqueeze(0).unsqueeze(0),
        target=target,
        data_range=1.0,
    )
    out = dict(zip(METRICS, (float(v) for v in values), strict=True))
    if not all(math.isfinite(x) for x in out.values()):
        raise ValueError("Nonfinite accuracy metric")
    return out


def paired_case(
    acq_dir: Path,
    weight_dir: Path,
    calibration_dir: Path,
    calibration_sha: str,
    acquisition_record: dict,
    acquisition_protocol: dict,
    destination: Path,
    *,
    iters: int = 120,
    opt_tol: float = 1e-3,
    stage: str = "pilot",
):
    """Run both methods once, store raw images, diagnostics and full trace."""
    category, sample_id = (
        acquisition_record["category"],
        int(acquisition_record["sample_id"]),
    )
    case = load_step5_case(acq_dir, weight_dir, category, sample_id, verify_all=False)
    cal = load_calibration_record(
        calibration_dir,
        category,
        sample_id,
        expected_manifest_sha256=calibration_sha,
    )
    if (
        cal["source_acquisition_sha256"] != acquisition_record["sha256"]
        or cal["source_weight_sha256"] != case.weight_sha256
    ):
        raise ValueError("Calibrated epsilons do not belong to frozen A,y")
    with np.load(acq_dir / acquisition_record["file"], allow_pickle=False) as f:
        source_meta = json.loads(str(f["metadata_json"]))
    if source_meta["source_sha256"] != acquisition_record["source_sha256"]:
        raise ValueError("Frozen image source identity differs")
    image_path = (
        Path(acquisition_protocol["data"]["root"]) / source_meta["source_relpath"]
    )
    if _sha256(image_path) != acquisition_record["source_sha256"]:
        raise ValueError("Ground-truth PNG has changed since frozen Step 2")
    target64 = load_png_float64(image_path)
    if target64.dtype != torch.float64 or target64.shape != (1, 1, 256, 256):
        raise ValueError("Unexpected validation ground-truth pixel convention")
    original_measurements = case.acquisition.measurement.clone()
    original_indices = np.asarray(case.acquisition.sampled_indices).copy()
    digest = hashlib.sha256(
        original_measurements.detach().cpu().numpy().tobytes()
    ).hexdigest()
    result = {
        "stage": stage,
        "category": category,
        "sample_id": sample_id,
        "permutation_position_1based": acquisition_record[
            "permutation_position_1based"
        ],
        "source_image_sha256": acquisition_record["source_sha256"],
        "source_acquisition_sha256": case.acquisition_sha256,
        "source_weight_sha256": case.weight_sha256,
        "measurement_tensor_sha256": digest,
        "wavelet_name": "db4",
        "wavelet_mode": "periodization",
        "sampling_ratio": 0.2,
        "measurement_count": len(original_indices),
        "reconstruction_dtype": "float32",
        "calibration_manifest_sha256": calibration_sha,
        "solve_order": list(
            METHODS
            if int(acquisition_record["permutation_position_1based"]) % 2
            else tuple(reversed(METHODS))
        ),
        "methods": {},
    }
    for method in result["solve_order"]:
        epsilon = cal[f"epsilon_{method}_95"]
        timestamp_start = time.perf_counter()
        try:
            reconstruction, solver_diag = solve_step5_case(
                case,
                method=method,
                epsilon=epsilon,
                iters=iters,
                opt_tol=opt_tol,
                factor=1.0,
            )
            runtime = time.perf_counter() - timestamp_start
            metrics = metrics_for_image(reconstruction, target64)
            status = int(solver_diag["solver_status"])
            iterations = int(solver_diag["iterations"])
            stable = (
                bool(solver_diag["image_finite"])
                and bool(solver_diag["feasible_with_tolerance"])
                and bool(solver_diag["solver_converged"])
            )
            image_file = (
                f"reconstructions/{category}/" f"sample_{sample_id:04d}_{method}.npz"
            )
            image_path_out = destination / image_file
            image_path_out.parent.mkdir(parents=True, exist_ok=True)
            image_array = (
                reconstruction.detach().cpu().numpy().astype(np.float32, copy=True)
            )
            if image_array.shape != (256, 256):
                raise ValueError("Reconstructed image has unexpected shape")
            _write_atomic_new(
                image_path_out,
                lambda f: np.savez_compressed(
                    f,
                    image=image_array,
                    category=np.array(category),
                    sample_id=np.array(sample_id, dtype=np.int64),
                    method=np.array(method),
                ),
            )
            image_path_out.chmod(0o444)
            result["methods"][method] = {
                "status": "PASS" if stable else "SOLVER_WARNING",
                "solver": solver_diag,
                "metrics": metrics,
                "runtime_seconds": runtime,
                "iterations": iterations,
                "seconds_per_iteration": (
                    runtime / iterations if iterations > 0 else None
                ),
                "reconstruction_file": image_file,
                "reconstruction_sha256": _sha256(image_path_out),
                "calibrated_epsilon": epsilon,
            }
        except Exception as exc:
            result["methods"][method] = {
                "status": "ERROR",
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "runtime_seconds": time.perf_counter() - timestamp_start,
                "calibrated_epsilon": epsilon,
            }
        print(
            f"{stage:5} {category:10} {sample_id:04d} {method.upper():3} "
            f"status={result['methods'][method]['status']:14} "
            f"iters={result['methods'][method].get('iterations','?'):>3} "
            f"sec={result['methods'][method]['runtime_seconds']:.4f}",
            flush=True,
        )
    if not torch.equal(case.acquisition.measurement, original_measurements):
        raise RuntimeError(
            "Frozen noisy measurements were modified during reconstruction"
        )
    if not np.array_equal(case.acquisition.sampled_indices, original_indices):
        raise RuntimeError("Frozen sensing indices changed during reconstruction")
    return result


def _rows_for_csv(records: list[dict]):
    rows = []
    for item in records:
        row = {
            k: item[k]
            for k in (
                "stage",
                "category",
                "sample_id",
                "permutation_position_1based",
                "measurement_tensor_sha256",
                "source_acquisition_sha256",
                "source_weight_sha256",
            )
        }
        for method in METHODS:
            data = item["methods"].get(method, {})
            row[f"{method}_status"] = data.get("status")
            row[f"{method}_epsilon"] = data.get("calibrated_epsilon")
            for k in COMPARISONS:
                row[f"{method}_{k}"] = (
                    data.get("metrics", {}).get(k) if k in METRICS else data.get(k)
                )
            solver = data.get("solver", {})
            for field in (
                "solver_status",
                "solver_status_name",
                "solver_converged",
                "feasible_with_tolerance",
                "ols_residual_norm",
                "wls_residual_norm",
            ):
                row[f"{method}_{field}"] = solver.get(field)
        for k in COMPARISONS:
            ols, wls = row.get(f"ols_{k}"), row.get(f"wls_{k}")
            row[f"delta_{k}"] = (
                float(wls - ols) if ols is not None and wls is not None else None
            )
        rows.append(row)
    return rows


def _summarize_delta(values, *, metric):
    """Fixed-seed paired 95% percentile bootstrap CI for the mean difference."""
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED + COMPARISONS.index(metric))
    samples = x[rng.integers(0, len(x), size=(10_000, len(x)))]
    means = samples.mean(axis=1)
    positive_good = metric not in (
        "mse",
        "runtime_seconds",
        "iterations",
        "seconds_per_iteration",
    )
    return {
        "n": int(x.size),
        "mean_delta_wls_minus_ols": float(x.mean()),
        "std_delta_sample": float(x.std(ddof=1)) if len(x) > 1 else None,
        "median_delta": float(np.median(x)),
        "win_count_wls": int(np.sum(x > 0) if positive_good else np.sum(x < 0)),
        "tie_count": int(np.sum(x == 0)),
        "win_rate_wls": float(np.mean(x > 0) if positive_good else np.mean(x < 0)),
        "mean_delta_bootstrap95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def summarize_results(records):
    rows = _rows_for_csv(records)
    summary = {}
    for category in ("ALL", *ORDERED_CATEGORIES):
        part = [r for r in rows if category == "ALL" or r["category"] == category]
        summary[category] = {
            "total_cases": len(part),
            "fully_successful": sum(
                r["ols_status"] == r["wls_status"] == "PASS" for r in part
            ),
            "ols_solver_issues": sum(r["ols_status"] != "PASS" for r in part),
            "wls_solver_issues": sum(r["wls_status"] != "PASS" for r in part),
            "metric_differences": {
                k: _summarize_delta(
                    [r[f"delta_{k}"] for r in part if r[f"delta_{k}"] is not None],
                    metric=k,
                )
                for k in COMPARISONS
            },
            "method_means": {
                m: {
                    k: (
                        float(
                            np.mean(
                                [
                                    r[f"{m}_{k}"]
                                    for r in part
                                    if r[f"{m}_{k}"] is not None
                                ]
                            )
                        )
                        if any(r[f"{m}_{k}"] is not None for r in part)
                        else None
                    )
                    for k in COMPARISONS
                }
                for m in METHODS
            },
        }
    return rows, summary


def _write_manifest(
    root, records, summary, *, iters, opt_tol, calibration_sha, stage_status
):
    rows, recomputed = summarize_results(records)
    if summary != recomputed:
        raise RuntimeError("Summary not consistent with case results")
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_acquisitions_sha256": ACQUISITION_MANIFEST_SHA256,
        "source_weights_sha256": WEIGHT_MANIFEST_SHA256,
        "source_calibration_sha256": calibration_sha,
        "measurement_repeats": 1,
        "validation_cases": len(records),
        "test_cases": 0,
        "stage_status": stage_status,
        "shared_wavelet": "db4",
        "wavelet_mode": "periodization",
        "solver": {
            "backend": "original LION/pcm/spgl1_bpdn.py with existing spgl1_torch",
            "device": "cpu",
            "max_iterations": iters,
            "opt_tol": opt_tol,
            "factor": 1,
            "debias": False,
            "master_data_dtype": "float64",
            "solver_dtype": "float32",
        },
        "metrics": {
            "implementation": "LION.utils.metrics.compute_metrics_2d",
            "data_range": 1.0,
            "MSE_PSNR_SSIM": "raw reconstruction; no clipping",
            "HaarPSI": "reconstruction clipped to [0,1] internally",
            "paired_delta": "WLS minus OLS",
            "bootstrap": "10000 paired bootstrap replicates with fixed seed",
        },
        "source_sha256": _sha256(Path(__file__)),
        "categories": list(ORDERED_CATEGORIES),
        "pilot_per_category": 2,
        "cases": records,
        "summary": summary,
    }
    write_json_new(root / "manifest.json", manifest)
    output_csv = root / "per_image_metrics.csv"
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_new(root / "summary.json", summary)
    filepaths = [
        "manifest.json",
        "summary.json",
        "pilot_summary.json",
        "per_image_metrics.csv",
        *[
            r["methods"][method]["reconstruction_file"]
            for r in records
            for method in METHODS
            if r["methods"][method].get("reconstruction_file")
        ],
    ]
    _write_atomic_new(
        root / "SHA256SUMS",
        lambda f: f.write(
            "".join(f"{_sha256(root / p)}  {p}\n" for p in filepaths).encode()
        ),
    )
    print("EXTERNAL_STEP6_MANIFEST_SHA256=" + _sha256(root / "manifest.json"))
    return manifest


def run_validation(
    acquisition_root,
    weight_root,
    calibration_root,
    output_dir,
    *,
    calibration_sha=CALIBRATION_MANIFEST_SHA256,
    iters=120,
    opt_tol=1e-3,
    pilot_per_category=2,
):
    """Run 10-image pilot then full set, saving one immutable archive."""
    acq_root, w_root, cal_root, root = map(
        Path, (acquisition_root, weight_root, calibration_root, output_dir)
    )
    if root.exists():
        raise FileExistsError("Step 6 output already exists; cannot overwrite")
    if pilot_per_category != 2:
        raise ValueError("The approved Step 6 pilot is fixed at 2 images/category")
    if iters != 120 or opt_tol != 1e-3:
        raise ValueError("Step 6 uses the unchanged Step 5 solver settings")
    validate_frozen_archives(acq_root, w_root)
    cal_manifest_path = cal_root / "manifest.json"
    if _sha256(cal_manifest_path) != calibration_sha:
        raise ValueError("Calibration hash does not match external anchor")
    cal_manifest = json.loads(cal_manifest_path.read_text())
    if (
        len(cal_manifest["cases"]) != 50
        or not cal_manifest["all_per_case_coverage_in_predeclared_gate"]
    ):
        raise ValueError("All 50 validated 95% calibration bounds are required")
    source_protocol = json.loads((acq_root / "protocol.json").read_text())
    source_manifest = json.loads((acq_root / "manifest.json").read_text())
    orig = source_manifest["acquisitions"]
    if source_manifest["test_count"] != 0 or len(orig) != 50:
        raise ValueError("Step 6 uses only the frozen validation split")
    for cat in ORDERED_CATEGORIES:
        selected = [r for r in orig if r["category"] == cat]
        if len(selected) != 10:
            raise ValueError(f"Missing validation cases from {cat}")
    pilot = []
    rest = []
    for cat in ORDERED_CATEGORIES:
        cases = sorted(
            [r for r in orig if r["category"] == cat],
            key=lambda r: int(r["permutation_position_1based"]),
        )
        pilot.extend(cases[:pilot_per_category])
        rest.extend(cases[pilot_per_category:])
    if len(pilot) != 10 or len(rest) != 40:
        raise ValueError("Incorrect predeclared pilot/full validation sizes")
    root.mkdir(parents=True, exist_ok=False)
    records = []
    torch.set_num_threads(2)
    for stage, items in (("pilot", pilot), ("full", rest)):
        for rec in items:
            records.append(
                paired_case(
                    acq_root,
                    w_root,
                    cal_root,
                    calibration_sha,
                    rec,
                    source_protocol,
                    root,
                    iters=iters,
                    opt_tol=opt_tol,
                    stage=stage,
                )
            )
        if stage == "pilot":
            _, current_summary = summarize_results(records)
            write_json_new(root / "pilot_summary.json", current_summary)
            print(
                "PILOT_CHECK:",
                current_summary["ALL"]["fully_successful"],
                "/10 both methods successful",
                flush=True,
            )
            # Preserve failures, but still complete the predeclared 50.
    _, summary = summarize_results(records)
    manifest = _write_manifest(
        root,
        records,
        summary,
        iters=iters,
        opt_tol=opt_tol,
        calibration_sha=calibration_sha,
        stage_status="complete",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-calibration-manifest-sha256", default=CALIBRATION_MANIFEST_SHA256
    )
    args = parser.parse_args()
    result = run_validation(
        args.acquisitions,
        args.weights,
        args.calibration,
        args.output,
        calibration_sha=args.expected_calibration_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "pilot": result["summary"]["ALL"]["fully_successful"],
                "all_methods": result["summary"]["ALL"]["method_means"],
                "status": result["stage_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
