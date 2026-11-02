"""Independent Step 5 reviewer tests, retained as real (not xfail) regressions.

Exercises constant-weight solver equivalence, 95% seed replay, frozen
acquisition noise, CLI epsilon, coherent archive tampering and SPGL1 rgap.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys

import numpy as np
import pytest
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared
from scripts.experiments.pcm.oracle_wls import calibrate_residual_bounds as calibration
from scripts.experiments.pcm.oracle_wls import weighted_bpdn as wb
from scripts.experiments.pcm.oracle_wls.calibrate_residual_bounds import (
    build_calibration,
    sample_norm_pairs,
)
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import (
    _locked_noise,
    frozen_case,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_case(inverse_std, m=1200, sigma=0.02):
    """64x64 map with thin lines/dark block and controlled Gaussian noise."""
    img = np.full((64, 64), 0.7)
    img[:, 8::16] = 0.2
    img[20:36, 24:44] = 0.35
    op = PhotocurrentMapOp(
        j_order=6,
        s_indices=np.arange(m),
        normalized=True,
        device="cpu",
    )
    noise = sigma * np.random.default_rng(3).standard_normal(m)
    y = op(torch.from_numpy(img)).reshape(-1) + torch.from_numpy(noise)
    d = torch.full((m,), float(inverse_std), dtype=torch.float64)
    acquisition = type("Acq", (), {"op": op, "measurement": y})()
    return (
        wb.Step5Case("Toy", 1, acquisition, d, 1 / d**2, "a", "w"),
        1.03 * sigma * np.sqrt(m),
    )


def test_constant_weights_reproduce_the_ols_solution():
    """D=cI and eps_WLS=c eps_OLS must be the same constrained problem."""
    c = 16.0
    case_ols, eps = _toy_case(1.0)
    case_wls, _ = _toy_case(c)
    x_ols, info_ols = wb.solve_step5_case(case_ols, method="ols", epsilon=eps)
    x_wls, info_wls = wb.solve_step5_case(
        case_wls,
        method="wls",
        epsilon=c * eps,
    )
    for info in (info_ols, info_wls):
        assert info["solver_status_name"] == "root_found"
        assert abs(info["selected_residual_norm"] / info["epsilon_used"] - 1) < 1.5e-3
    assert torch.linalg.vector_norm(x_wls - x_ols) < (
        1e-3 * torch.linalg.vector_norm(x_ols)
    )
    assert info_wls["ols_residual_norm"] == pytest.approx(
        info_ols["ols_residual_norm"],
        rel=1e-3,
    )


@pytest.fixture(scope="module")
def calibrated(prepared, tmp_path_factory):
    """One-case synthetic calibration records exact draws and seed per call."""
    root, sha_a, sha_w = prepared
    out = tmp_path_factory.mktemp("step5_review") / "calibration"
    calls = []

    def spy(*args, **kwargs):
        calls.append((kwargs["draws"], kwargs["seed"]))
        return sample_norm_pairs(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(calibration, "sample_norm_pairs", spy)
        result = build_calibration(
            root / "acquisitions",
            root / "weights",
            out,
            expected_acquisition_manifest_sha256=sha_a,
            expected_weight_manifest_sha256=sha_w,
            fit_draws=512,
            holdout_draws=256,
            limit_cases=1,
        )
    return root, sha_a, sha_w, out, result["cases"][0], calls


def _archive(root):
    source, weights = root / "acquisitions", root / "weights"
    protocol = json.loads((source / "protocol.json").read_text())
    cases = json.loads((source / "manifest.json").read_text())["acquisitions"]
    files = {
        (r["category"], int(r["sample_id"])): r
        for r in json.loads((weights / "manifest.json").read_text())["files"]
    }
    return source, weights, protocol, cases, files


def _frozen(root, category, sample_id):
    source, weights, protocol, cases, files = _archive(root)
    case = next(
        c
        for c in cases
        if (c["category"], int(c["sample_id"])) == (category, int(sample_id))
    )
    acquired, _, plus, minus, clean, parts = frozen_case(
        source,
        weights,
        case,
        files[(category, int(sample_id))],
        protocol,
    )
    return (
        acquired,
        plus,
        minus,
        clean,
        parts["variance"],
        _locked_noise(protocol["noise"]).noise_params,
    )


def test_recorded_seeds_replay_bounds_and_coverage_exactly(calibrated):
    root, _, _, _, rec, calls = calibrated
    assert calls == [
        (rec["fit_draws"], int(rec["fit_seed_decimal"])),
        (rec["holdout_draws"], int(rec["holdout_seed_decimal"])),
    ]
    acq, plus, minus, _, variance, params = _frozen(
        root,
        rec["category"],
        rec["sample_id"],
    )
    d, i0 = 1 / np.sqrt(variance), float(acq.i0.reshape(-1)[0])
    fit = sample_norm_pairs(
        plus,
        minus,
        i0,
        d,
        params,
        draws=rec["fit_draws"],
        seed=int(rec["fit_seed_decimal"]),
    )
    hold = sample_norm_pairs(
        plus,
        minus,
        i0,
        d,
        params,
        draws=rec["holdout_draws"],
        seed=int(rec["holdout_seed_decimal"]),
    )
    for k, name in enumerate(("ols", "wls")):
        eps = float(np.quantile(fit[k], 0.95, method="higher"))
        assert eps == rec[f"epsilon_{name}_95"]
        assert float(np.mean(hold[k] <= eps)) == rec[f"holdout_coverage_{name}"]


def test_bounds_just_above_reference_scales(calibrated):
    rec = calibrated[4]
    assert rec["epsilon_wls_reference"] == pytest.approx(np.sqrt(13107))
    for name in ("ols", "wls"):
        assert 1.0 < rec[f"epsilon_{name}_95"] / rec[f"epsilon_{name}_reference"] < 1.03


def test_frozen_step2_errors_match_new_noise_draws(prepared):
    root, *_ = prepared
    source, weights, protocol, cases, files = _archive(root)
    params = _locked_noise(protocol["noise"]).noise_params
    z = {"ols": [], "wls": []}
    for n, case in enumerate(cases[:10]):
        acq, _, plus, minus, clean, parts = frozen_case(
            source,
            weights,
            case,
            files[(case["category"], int(case["sample_id"]))],
            protocol,
        )
        d, i0 = 1 / np.sqrt(parts["variance"]), float(acq.i0.reshape(-1)[0])
        e = (acq.measurement.reshape(-1) - clean).numpy()
        draws = sample_norm_pairs(
            plus,
            minus,
            i0,
            d,
            params,
            draws=256,
            seed=7000 + n,
        )
        for k, (name, value) in enumerate(
            (
                ("ols", np.linalg.norm(e)),
                ("wls", np.linalg.norm(d * e)),
            )
        ):
            z[name].append((value - draws[k].mean()) / draws[k].std(ddof=1))
    for name, scores in z.items():
        assert np.all(np.abs(scores) < 4), name
        assert abs(np.mean(scores)) < 4 / np.sqrt(len(scores)), name


@pytest.mark.parametrize("method", ["ols", "wls"])
def test_cli_uses_the_selected_method_bound(
    calibrated,
    monkeypatch,
    capsys,
    method,
):
    root, sha_a, sha_w, out, rec, _ = calibrated
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weighted_bpdn",
            "--acquisitions",
            str(root / "acquisitions"),
            "--weights",
            str(root / "weights"),
            "--calibration",
            str(out),
            "--expected-calibration-manifest-sha256",
            _sha(out / "manifest.json"),
            "--category",
            rec["category"],
            "--sample-id",
            str(rec["sample_id"]),
            "--method",
            method,
        ],
    )
    capsys.readouterr()
    wb.main()
    info = json.loads(capsys.readouterr().out)
    assert info["method"] == method
    assert info["epsilon_used"] == rec[f"epsilon_{method}_95"]
    assert info["solver_status_name"] == "root_found"
    assert info["feasible_with_tolerance"]


def _coherently_tampered_weights(root, tmp_path):
    target = tmp_path / "weights"
    shutil.copytree(root / "weights", target)
    for path in target.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    manifest = json.loads((target / "manifest.json").read_text())
    record = manifest["files"][0]
    path = target / record["file"]
    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in f.files}
    for key in (
        "variance",
        "variance_dark",
        "variance_bias_white",
        "variance_bias_drift",
        "variance_beam",
    ):
        arrays[key] *= 2
    arrays["weight"] = 1 / arrays["variance"]
    arrays["inverse_std"] = 1 / np.sqrt(arrays["variance"])
    path.unlink()
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    record.update(sha256=_sha(path), bytes=path.stat().st_size)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = []
    for line in (target / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        lines.append(
            f"{_sha(target/rel) if rel in ('manifest.json',record['file']) else digest}  {rel}"
        )
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return target, record


def test_full_verification_rejects_coherent_weight_edit(
    prepared,
    tmp_path,
    monkeypatch,
):
    root, sha_a, sha_w = prepared
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    target, record = _coherently_tampered_weights(root, tmp_path)
    with pytest.raises(ValueError, match="Externally anchored"):
        wb.load_step5_case(
            root / "acquisitions",
            target,
            record["category"],
            int(record["sample_id"]),
        )


def test_loader_always_checks_anchors_with_verify_all_false(
    prepared,
    tmp_path,
    monkeypatch,
):
    root, sha_a, sha_w = prepared
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", sha_a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", sha_w)
    target, record = _coherently_tampered_weights(root, tmp_path)
    with pytest.raises(ValueError, match="Externally anchored"):
        wb.load_step5_case(
            root / "acquisitions",
            target,
            record["category"],
            int(record["sample_id"]),
            verify_all=False,
        )


def test_solver_records_relative_duality_gap():
    case, eps = _toy_case(1.0)
    _, info = wb.solve_step5_case(case, method="ols", epsilon=eps)
    assert "relative_duality_gap" in info
    assert isinstance(info["relative_duality_gap"], float)
    assert np.isfinite(info["relative_duality_gap"])
    assert info["relative_duality_gap"] >= 0
    assert info["spgl1_l1_ball_radius"] > 0
