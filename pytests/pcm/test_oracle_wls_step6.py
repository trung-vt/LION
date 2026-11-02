"""Step 6 paired OLS/oracle WLS regression, statistics and integrity tests."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared
from scripts.experiments.pcm.oracle_wls import compare_ols_wls_step6 as exp
from scripts.experiments.pcm.oracle_wls import weighted_bpdn as wb
from scripts.experiments.pcm.oracle_wls.calibrate_residual_bounds import (
    build_calibration,
)


def _fake(status="PASS", value=1.0):
    return {
        "methods": {
            method: {
                "status": status,
                "calibrated_epsilon": 3.0,
                "metrics": {
                    "mse": value + (0.01 if method == "wls" else 0.0),
                    "psnr": value + (0.05 if method == "wls" else 0.0),
                    "ssim": value,
                    "haarpsi": value,
                },
                "runtime_seconds": 0.12 if method == "ols" else 0.10,
                "iterations": 10 if method == "ols" else 8,
                "seconds_per_iteration": 0.012 if method == "ols" else 0.0125,
                "solver": {
                    "solver_status": 1,
                    "solver_converged": True,
                    "solver_status_name": "root_found",
                    "feasible_with_tolerance": True,
                    "ols_residual_norm": 2,
                    "wls_residual_norm": 33,
                },
            }
            for method in ("ols", "wls")
        },
        "stage": "pilot",
        "category": "CIGS",
        "sample_id": 733,
        "permutation_position_1based": 901,
        "measurement_tensor_sha256": "0" * 64,
        "source_acquisition_sha256": "1" * 64,
        "source_weight_sha256": "2" * 64,
    }


def test_bootstrap_paired_difference_not_incorrectly_unpaired():
    recs = []
    for i in range(50):
        rec = _fake(value=float(i))
        rec["sample_id"] = i + 1
        recs.append(rec)
    rows, result = exp.summarize_results(recs)
    assert len(rows) == 50
    assert result["ALL"]["fully_successful"] == 50
    d = result["ALL"]["metric_differences"]["psnr"]
    assert d["n"] == 50
    assert d["mean_delta_wls_minus_ols"] == pytest.approx(0.05)
    assert d["std_delta_sample"] < 1e-12
    assert d["win_count_wls"] == 50
    assert d["mean_delta_bootstrap95_ci"][0] == pytest.approx(0.05)
    assert result["ALL"]["metric_differences"]["mse"]["win_count_wls"] == 0
    assert result["ALL"]["metric_differences"]["runtime_seconds"]["win_count_wls"] == 50
    assert (
        result["ALL"]["metric_differences"]["seconds_per_iteration"]["win_count_wls"]
        == 0
    )


def test_failed_solver_is_counted_not_excluded():
    rec = _fake()
    rec["methods"]["wls"]["status"] = "SOLVER_WARNING"
    rows, summary = exp.summarize_results([rec])
    assert len(rows) == 1
    assert summary["ALL"]["fully_successful"] == 0
    assert summary["ALL"]["wls_solver_issues"] == 1


def test_metric_uses_raw_for_mse_and_clipped_for_haarpsi():
    x = torch.linspace(0, 1, 256 * 256).reshape(1, 1, 256, 256).double()
    y = x[0, 0].float() + 1e-5
    a = exp.metrics_for_image(y, x)
    assert a["mse"] == pytest.approx(1e-10, rel=0.02)
    assert a["psnr"] > 90
    assert a["ssim"] > 0.999
    assert a["haarpsi"] > 0.99
    # Clipping is reserved for HaarPSI.
    z = y.clone()
    z[:10, :] = -0.1
    b = exp.metrics_for_image(z, x)
    assert b["mse"] > a["mse"] and b["psnr"] < a["psnr"]
    assert all(np.isfinite(list(b.values())))


def test_fixed_source_anchors_are_not_changed_by_step6():
    assert exp.ACQUISITION_MANIFEST_SHA256 == wb.ACQUISITION_MANIFEST_SHA256
    assert exp.WEIGHT_MANIFEST_SHA256 == wb.WEIGHT_MANIFEST_SHA256
    assert exp.CALIBRATION_MANIFEST_SHA256 == (
        "fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a"
    )


def test_six_case_pairing_pilot_recovers_synthetic_one_case(
    prepared, tmp_path, monkeypatch
):
    base, a_hash, w_hash = prepared
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", a_hash)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", w_hash)
    monkeypatch.setattr(exp, "ACQUISITION_MANIFEST_SHA256", a_hash)
    monkeypatch.setattr(exp, "WEIGHT_MANIFEST_SHA256", w_hash)
    cal_dir = tmp_path / "bounds"
    calibration = build_calibration(
        base / "acquisitions",
        base / "weights",
        cal_dir,
        expected_acquisition_manifest_sha256=a_hash,
        expected_weight_manifest_sha256=w_hash,
        fit_draws=512,
        holdout_draws=256,
        limit_cases=1,
    )
    manifest_hash = hashlib.sha256((cal_dir / "manifest.json").read_bytes()).hexdigest()
    sample = json.loads((base / "acquisitions/manifest.json").read_text())[
        "acquisitions"
    ][0]
    protocol = json.loads((base / "acquisitions/protocol.json").read_text())
    output = tmp_path / "paired"
    output.mkdir()
    result = exp.paired_case(
        base / "acquisitions",
        base / "weights",
        cal_dir,
        manifest_hash,
        sample,
        protocol,
        output,
        stage="pilot",
    )
    assert set(result["methods"]) == {"ols", "wls"}
    assert result["measurement_count"] == 13107
    assert result["wavelet_name"] == "db4"
    for name in ("ols", "wls"):
        assert result["methods"][name]["status"] == "PASS"
        path = output / result["methods"][name]["reconstruction_file"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            result["methods"][name]["reconstruction_sha256"]
        )
        with np.load(path, allow_pickle=False) as z:
            assert z["image"].shape == (256, 256)
            assert np.isfinite(z["image"]).all()
    assert np.isfinite(result["methods"]["ols"]["metrics"]["haarpsi"])
    assert np.isfinite(result["methods"]["wls"]["metrics"]["psnr"])
