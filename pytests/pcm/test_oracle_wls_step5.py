"""Comprehensive Step 5 weighted SPGL1-BPDN and q95 calibration tests.

All inputs are synthetic or disposable test archives. The production Step 2/3
frozen files, shared Wavelet2D and existing SPGL1 adapters are never modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from LION.operators.CompositeOp import CompositeOp
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.spgl1_bpdn import spgl1_bpdn_torch
from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared
from scripts.experiments.pcm.oracle_wls import weighted_bpdn as wb
from scripts.experiments.pcm.oracle_wls.calibrate_residual_bounds import (
    build_calibration,
    calibration_seed,
    empirical_upper_quantile,
    sample_norm_pairs,
)


def base_operator(j, mask):
    pcm = PhotocurrentMapOp(j_order=j, s_indices=mask, device="cpu", normalized=True)
    wavelet = Wavelet2D(
        (2**j, 2**j),
        wavelet_name="db4",
        mode="periodization",
        device="cpu",
    )
    return pcm, wavelet, CompositeOp(wavelet, pcm, device="cpu")


@pytest.mark.parametrize("j,m", [(4, 91), (6, 512)])
def test_weighted_forward_and_adjoint_consistency_with_db4(j, m):
    pcm, wavelet, base = base_operator(j, np.arange(m))
    torch.manual_seed(4)
    d = torch.linspace(0.4, 18, m, dtype=torch.float64)
    op = wb.DiagonallyWeightedOperator(base, d)
    z = torch.randn(wavelet.size, dtype=torch.float32)
    r = torch.randn(m, dtype=torch.float32)
    assert op.domain_shape == base.domain_shape
    assert op.range_shape == base.range_shape
    torch.testing.assert_close(op(z), base(z) * d.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        op.adjoint(r), base.adjoint(r * d.float()), rtol=0, atol=0
    )
    lhs = torch.dot(op(z).float(), r)
    rhs = torch.dot(z, op.adjoint(r).float())
    assert abs(float(lhs - rhs)) <= 6e-4 + 3e-4 * max(abs(float(lhs)), abs(float(rhs)))
    assert wavelet.wavelet.name == "db4"
    assert wavelet.mode == "periodization"


def test_identity_weights_reproduce_existing_operator():
    _, wavelet, base = base_operator(4, np.arange(160))
    d = np.ones(160, dtype=np.float64)
    op = wb.DiagonallyWeightedOperator(base, d)
    z = torch.randn(wavelet.size)
    r = torch.randn(160)
    torch.testing.assert_close(op(z), base(z), rtol=0, atol=0)
    torch.testing.assert_close(op.adjoint(r), base.adjoint(r), rtol=0, atol=0)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 2.0]),
        np.array([float("nan"), 1.0]),
        np.array([float("inf"), 1.0]),
        np.array([1.0]),
    ],
)
def test_invalid_weight_vectors_are_rejected(weights):
    _, _, base = base_operator(4, np.arange(2))
    with pytest.raises(ValueError, match="inverse_std"):
        wb.DiagonallyWeightedOperator(base, weights)


def test_unit_float64_source_weights_remain_float64():
    _, _, base = base_operator(4, np.arange(12))
    d = torch.linspace(1.0, 5.0, 12, dtype=torch.float64)
    snapshot = d.numpy().copy()
    op = wb.DiagonallyWeightedOperator(base, d)
    assert d.dtype == torch.float64
    assert op.inverse_std.dtype == torch.float32
    np.testing.assert_array_equal(snapshot, d.numpy())


@pytest.fixture(scope="module")
def archives(prepared):
    base, acq_sha, weight_sha = prepared
    return base, acq_sha, weight_sha


def test_frozen_case_loader_binds_exact_operator_and_master_y(archives, monkeypatch):
    root, sha_a, sha_w = archives
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    first = json.loads((root / "acquisitions/manifest.json").read_text())[
        "acquisitions"
    ][0]
    case = wb.load_step5_case(
        root / "acquisitions",
        root / "weights",
        first["category"],
        int(first["sample_id"]),
    )
    assert case.acquisition.measurement.dtype == torch.float64
    assert case.inverse_std64.dtype == torch.float64
    assert case.variance64.dtype == torch.float64
    assert case.acquisition_sha256 == first["sha256"]
    np.testing.assert_array_equal(
        case.inverse_std64.numpy(), 1 / np.sqrt(case.variance64.numpy())
    )


def test_frozen_case_loader_rejects_uncatalogued_sample(archives, monkeypatch):
    root, sha_a, sha_w = archives
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    with pytest.raises(ValueError, match="not uniquely"):
        wb.load_step5_case(root / "acquisitions", root / "weights", "CIGS", 99999)


def test_calibration_seed_is_independent_of_other_experiments():
    samples = [
        calibration_seed("CIGS", 733, "fit"),
        calibration_seed("CIGS", 733, "holdout"),
        calibration_seed("CIGS", 917, "fit"),
        calibration_seed("Organic", 733, "fit"),
    ]
    assert len(set(samples)) == len(samples)
    assert all(0 <= x < 2**64 for x in samples)
    with pytest.raises(ValueError, match="fit or holdout"):
        calibration_seed("CIGS", 733, "test")


def test_95_percent_quantile_rank_and_unit_scaling():
    x = np.arange(100, dtype=np.float64) + 1
    assert empirical_upper_quantile(x) == 96.0
    assert empirical_upper_quantile(10 * x) == 960.0
    with pytest.raises(ValueError):
        empirical_upper_quantile(x, probability=1.0)


def test_calibration_norm_samples_have_matched_noise():
    from LION.pcm.noise import PCMNoiseSimulation

    m = 11
    p = torch.linspace(4, 6, m, dtype=torch.float64)
    q = torch.linspace(1, 2, m, dtype=torch.float64)
    params = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8, j_order=4
    ).noise_params
    ols, wls = sample_norm_pairs(
        p,
        q,
        22.0,
        np.full(m, 3.0),
        params,
        draws=120,
        seed=123,
    )
    np.testing.assert_allclose(wls, 3 * ols, atol=2e-14, rtol=1e-14)
    second = sample_norm_pairs(
        p,
        q,
        22.0,
        np.full(m, 3.0),
        params,
        draws=120,
        seed=123,
    )
    np.testing.assert_array_equal(ols, second[0])
    np.testing.assert_array_equal(wls, second[1])


def test_synthetic_full_calibration_rejects_reuse_and_checks_coverage(
    archives,
    tmp_path,
    monkeypatch,
):
    root, sha_a, sha_w = archives
    out = tmp_path / "calibration"
    # A disposable two-image pilot exercises the complete immutable workflow.
    result = build_calibration(
        root / "acquisitions",
        root / "weights",
        out,
        expected_acquisition_manifest_sha256=sha_a,
        expected_weight_manifest_sha256=sha_w,
        fit_draws=512,
        holdout_draws=256,
        limit_cases=2,
    )
    assert result["number_of_cases"] == 2
    assert len(result["cases"]) == 2
    assert set((r["category"] for r in result["cases"])) == {"CIGS"}
    for rec in result["cases"]:
        assert rec["epsilon_ols_95"] > 0
        assert rec["epsilon_wls_95"] > 0
        assert rec["fit_seed_decimal"] != rec["holdout_seed_decimal"]
        assert 0 <= rec["holdout_coverage_ols"] <= 1
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    item = result["cases"][0]
    checked = wb.load_calibration_record(
        out,
        item["category"],
        item["sample_id"],
        expected_manifest_sha256=hashlib.sha256(
            (out / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    assert checked["epsilon_ols_95"] == item["epsilon_ols_95"]
    with pytest.raises(ValueError, match="External calibration"):
        wb.load_calibration_record(
            out,
            item["category"],
            item["sample_id"],
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(FileExistsError):
        build_calibration(
            root / "acquisitions",
            root / "weights",
            out,
            expected_acquisition_manifest_sha256=sha_a,
            expected_weight_manifest_sha256=sha_w,
            fit_draws=512,
            holdout_draws=256,
            limit_cases=2,
        )


def test_solver_uses_current_db4_and_residuals_ols_wls():
    pcm, wavelet, base = base_operator(4, np.arange(178))
    torch.manual_seed(10)
    image = torch.rand((16, 16), dtype=torch.float32)
    y = pcm(image) + 0.02 * torch.randn(178)
    case = wb.Step5Case(
        category="CIGS",
        sample_id=1,
        acquisition=type("Acq", (), {"op": pcm, "measurement": y.double()})(),
        inverse_std64=torch.linspace(4, 9, 178, dtype=torch.float64),
        variance64=1 / torch.linspace(4, 9, 178, dtype=torch.float64) ** 2,
        acquisition_sha256="x",
        weight_sha256="y",
    )
    x_ols, d_ols = wb.solve_step5_case(
        case,
        method="ols",
        epsilon=0.45,
        iters=65,
    )
    x_wls, d_wls = wb.solve_step5_case(
        case,
        method="wls",
        epsilon=2.8,
        iters=65,
    )
    for im, d in [(x_ols, d_ols), (x_wls, d_wls)]:
        assert tuple(im.shape) == (16, 16)
        assert torch.isfinite(im).all()
        assert d["wavelet_name"] == "db4"
        assert d["wavelet_mode"] == "periodization"
        assert d["solver_dtype"] == "float32"
        assert d["debias"] is False
        assert d["ols_residual_norm"] >= 0
        assert d["wls_residual_norm"] >= 0
        assert d["solver_residual_consistency_abs_error"] < 3e-4
        assert d["iterations"] > 0
    with pytest.raises(ValueError, match="factor=1"):
        wb.solve_step5_case(case, method="ols", epsilon=0.5, factor=2)


def test_full_resolution_db4_weighted_adjoint():
    # Confirm the actual 256x256 db4 operator, not merely 16x16 db4 tests.
    pcm, wavelet, base = base_operator(8, np.arange(0, 65536, 5))
    torch.manual_seed(47)
    whiten = torch.linspace(15.0, 19.0, base.range_shape[-1], dtype=torch.float64)
    op = wb.DiagonallyWeightedOperator(base, whiten)
    z = torch.randn(wavelet.size, dtype=torch.float32)
    residual = torch.randn(base.range_shape[-1], dtype=torch.float32)
    lhs = torch.dot(op(z), residual)
    rhs = torch.dot(z, op.adjoint(residual))
    assert abs(float(lhs - rhs)) <= (
        2e-3 + 3e-4 * max(abs(float(lhs)), abs(float(rhs)))
    )


def test_bad_calibration_csv_detected_by_original_manifest_anchor(
    archives,
    tmp_path,
    monkeypatch,
):
    root, sha_a, sha_w = archives
    target = tmp_path / "bounds"
    result = build_calibration(
        root / "acquisitions",
        root / "weights",
        target,
        expected_acquisition_manifest_sha256=sha_a,
        expected_weight_manifest_sha256=sha_w,
        fit_draws=512,
        holdout_draws=256,
        limit_cases=1,
    )
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    sha = hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    one = result["cases"][0]
    wb.load_calibration_record(
        target, one["category"], one["sample_id"], expected_manifest_sha256=sha
    )
    (target / "summary.csv").write_text("tampered,contents\n")
    with pytest.raises(ValueError, match="Calibration SHA256SUMS"):
        wb.load_calibration_record(
            target, one["category"], one["sample_id"], expected_manifest_sha256=sha
        )
