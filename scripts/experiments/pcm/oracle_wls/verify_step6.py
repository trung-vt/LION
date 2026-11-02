"""Integrity and independent metric checks for the frozen Step 6 PCM comparison.

Read-only: verifies SHA256SUMS, all 100 reconstructed arrays, the original
frozen A,y identity, matched source/calibration hashes, exact summary values,
and re-evaluates PSNR, SSIM, MSE and HaarPSI from the ground-truth PNGs.

A separately preserved Step 6 manifest SHA256 is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.experiments.pcm.oracle_wls.compare_ols_wls_step6 import (
    CALIBRATION_MANIFEST_SHA256,
    COMPARISONS,
    ORDERED_CATEGORIES,
    SCHEMA,
    metrics_for_image,
    summarize_results,
)
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    load_master_npz,
    load_png_float64,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import (
    ACQUISITION_MANIFEST_SHA256,
    WEIGHT_MANIFEST_SHA256,
    validate_frozen_archives,
)
import torch


def check_checksum_sidecar(directory: Path, manifest: dict):
    """Check every expected file and reject unlisted reconstruction NPZs."""
    files = {
        "manifest.json",
        "pilot_summary.json",
        "summary.json",
        "per_image_metrics.csv",
    }
    files.update(
        r["methods"][method]["reconstruction_file"]
        for r in manifest["cases"]
        for method in ("ols", "wls")
        if "reconstruction_file" in r["methods"][method]
    )
    entries = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        sha, relpath = line.split("  ", 1)
        p = Path(relpath)
        if p.is_absolute() or ".." in p.parts or relpath in entries:
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        entries[relpath] = sha
        if _sha256(directory / p) != sha:
            raise ValueError(f"Step 6 SHA256SUMS mismatch: {relpath}")
    if set(entries) != files:
        raise ValueError("Step 6 checksum coverage incomplete or unexpected")
    found = {
        str(p.relative_to(directory))
        for p in (directory / "reconstructions").rglob("*.npz")
    }
    expected = {p for p in files if p.endswith(".npz")}
    if found != expected:
        raise ValueError("Unlisted or missing reconstructed image NPZ")


def validate_outputs(
    acquisition_root,
    weight_root,
    calibration_root,
    output_root,
    *,
    expected_manifest_sha256,
    recompute_metrics=True,
):
    """Check fixed 50-pair experiment and optionally recalculate all metrics."""
    acq, weights, calibration, root = map(
        Path, (acquisition_root, weight_root, calibration_root, output_root)
    )
    validate_frozen_archives(acq, weights)
    if _sha256(calibration / "manifest.json") != CALIBRATION_MANIFEST_SHA256:
        raise ValueError("Step 5 calibration source changed")
    if _sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Step 6 external manifest SHA256 mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["validation_cases"] != 50:
        raise ValueError("Incorrect Step 6 schema or case count")
    if manifest["test_cases"] != 0:
        raise ValueError("Step 6 unexpectedly includes held-out cases")
    if (
        manifest["source_acquisitions_sha256"] != ACQUISITION_MANIFEST_SHA256
        or manifest["source_weights_sha256"] != WEIGHT_MANIFEST_SHA256
        or manifest["source_calibration_sha256"] != CALIBRATION_MANIFEST_SHA256
    ):
        raise ValueError("Step 6 frozen input identities do not match")
    if manifest["shared_wavelet"] != "db4":
        raise ValueError("Unexpected wavelet in validation")
    if manifest["solver"] != {
        "backend": "original LION/pcm/spgl1_bpdn.py with existing spgl1_torch",
        "device": "cpu",
        "max_iterations": 120,
        "opt_tol": 1e-3,
        "factor": 1,
        "debias": False,
        "master_data_dtype": "float64",
        "solver_dtype": "float32",
    }:
        raise ValueError("Solver settings differ from frozen Step 5")
    if (
        _sha256(Path(__file__).with_name("compare_ols_wls_step6.py"))
        != manifest["source_sha256"]
    ):
        raise ValueError("Step 6 experiment source changed since archive creation")
    check_checksum_sidecar(root, manifest)
    cases = manifest["cases"]
    categories = Counter(r["category"] for r in cases)
    if any(categories[c] != 10 for c in ORDERED_CATEGORIES):
        raise ValueError("Incomplete validation category coverage")
    if len({(r["category"], r["sample_id"]) for r in cases}) != 50:
        raise ValueError("Duplicate or missing validation cases")
    if sum(r["stage"] == "pilot" for r in cases) != 10:
        raise ValueError("Incorrect pilot size")
    if sum(r["stage"] == "full" for r in cases) != 40:
        raise ValueError("Incorrect remaining validation size")

    a_manifest = json.loads((acq / "manifest.json").read_text())
    source_protocol = json.loads((acq / "protocol.json").read_text())
    indexed = {(r["category"], r["sample_id"]): r for r in a_manifest["acquisitions"]}
    checks = 0
    for record in cases:
        key = record["category"], record["sample_id"]
        src = indexed[key]
        if record["source_acquisition_sha256"] != src["sha256"]:
            raise ValueError("Reconstruction used incorrect noisy source")
        if record["source_image_sha256"] != src["source_sha256"]:
            raise ValueError("Reconstruction used incorrect source image")
        acquisition, md = load_master_npz(
            acq / src["file"], expected_sha256=src["sha256"]
        )
        y_sha = hashlib.sha256(
            acquisition.measurement.detach().cpu().numpy().tobytes()
        ).hexdigest()
        if y_sha != record["measurement_tensor_sha256"]:
            raise ValueError("OLS/WLS reconstruction did not use frozen y")
        clean = load_png_float64(
            Path(source_protocol["data"]["root"]) / md["source_relpath"]
        )
        if (
            _sha256(Path(source_protocol["data"]["root"]) / md["source_relpath"])
            != src["source_sha256"]
        ):
            raise ValueError("Ground-truth PNG changed")
        for mode in ("ols", "wls"):
            item = record["methods"][mode]
            if item["status"] not in ("PASS", "SOLVER_WARNING", "ERROR"):
                raise ValueError("Unrecognized solver status")
            if item["status"] == "ERROR":
                raise ValueError("Step 6 encountered a recorded solver failure")
            if (
                item["solver"]["source_acquisition_sha256"] != src["sha256"]
                or item["solver"]["debias"] is not False
                or item["solver"]["wavelet_name"] != "db4"
            ):
                raise ValueError("Incorrect solver A,y source or wavelet")
            file = root / item["reconstruction_file"]
            if _sha256(file) != item["reconstruction_sha256"]:
                raise ValueError("Unexpected reconstructed-image SHA")
            with np.load(file, allow_pickle=False) as f:
                im = f["image"].copy()
                if (
                    int(f["sample_id"]) != record["sample_id"]
                    or str(f["category"]) != record["category"]
                    or str(f["method"]) != mode
                ):
                    raise ValueError("Reconstruction file metadata disagrees with case")
            if im.dtype != np.float32 or im.shape != (256, 256):
                raise ValueError("Unexpected reconstruction array shape or dtype")
            if not np.isfinite(im).all():
                raise ValueError("Non-finite reconstruction")
            if recompute_metrics:
                fresh = metrics_for_image(torch.from_numpy(im), clean)
                for metric in ("mse", "psnr", "ssim", "haarpsi"):
                    if not np.isclose(
                        fresh[metric],
                        item["metrics"][metric],
                        rtol=0,
                        atol=5e-7,
                    ):
                        raise ValueError(
                            f"Recomputed {metric} differs for {key}/{mode}"
                        )
                checks += 1
    rows, summary = summarize_results(cases)
    # The reconstruction timestamps are excluded from reproducible summary.
    if summary != manifest["summary"]:
        raise ValueError("Paired summary / paired bootstrap does not regenerate")
    if json.loads((root / "summary.json").read_text()) != summary:
        raise ValueError("Summary sidecar inconsistent with manifest")
    # Single-source evidence from the pilot is checked independently.
    pilot_cases = cases[:10]
    _, pilot_summary = summarize_results(pilot_cases)
    if json.loads((root / "pilot_summary.json").read_text()) != pilot_summary:
        raise ValueError("Pilot summary does not match predeclared first ten images")
    return {
        "status": "PASS",
        "total_cases": 50,
        "pilot_cases": 10,
        "full_cases_after_pilot": 40,
        "reconstruction_files_checked": 100,
        "metrics_recomputed": checks,
        "ols_solver_issues": summary["ALL"]["ols_solver_issues"],
        "wls_solver_issues": summary["ALL"]["wls_solver_issues"],
        "saved_step6_manifest_sha256": expected_manifest_sha256,
        "saved_step5_calibration_sha256": CALIBRATION_MANIFEST_SHA256,
        "step4_rerun": False,
        "heldout_test_images_used": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquisitions", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-step6-manifest-sha256", required=True)
    p.add_argument("--skip-metric-recompute", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    result = validate_outputs(
        args.acquisitions,
        args.weights,
        args.calibration,
        args.output,
        expected_manifest_sha256=args.expected_step6_manifest_sha256,
        recompute_metrics=not args.skip_metric_recompute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
