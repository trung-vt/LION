"""Independent reviewer Step 6 tests: metrics, paired stats and verifier tampering.

Reproduces the 12 supplied reviewer cases. The previous strict xfail
(shared PSNR mistake) now must PASS because Step 6 verification uses a
separate NumPy metric implementation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared
from scripts.experiments.pcm.oracle_wls import compare_ols_wls_step6 as exp
from scripts.experiments.pcm.oracle_wls import verify_step6 as verify
from scripts.experiments.pcm.oracle_wls import weighted_bpdn as wb
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    load_master_npz,
    load_png_float64,
    write_json_new,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene(n=256):
    yy, xx = np.mgrid[0:n, 0:n] / n
    gt = np.clip(0.6 + 0.3 * np.sin(6 * xx) * np.cos(4 * yy), 0, 1)
    gt[n // 3 : n // 3 + 6, :] = 0.1
    return gt


def test_psnr_and_mse_follow_the_plain_formulas():
    gt = _scene()
    rec = (gt + 0.03 * np.random.default_rng(0).standard_normal(gt.shape)).astype(
        np.float32
    )
    rec[:5] = -0.05
    m = exp.metrics_for_image(
        torch.from_numpy(rec), torch.from_numpy(gt).reshape(1, 1, *gt.shape)
    )
    mse = np.mean((rec.astype(np.float64) - gt.astype(np.float32)) ** 2)
    assert m["mse"] == pytest.approx(mse, rel=1e-6)
    assert m["psnr"] == pytest.approx(10 * np.log10(1 / mse), abs=1e-4)


def test_ssim_stays_close_to_scikit_image():
    gt = _scene()
    rec = (gt + 0.03 * np.random.default_rng(0).standard_normal(gt.shape)).astype(
        np.float32
    )
    m = exp.metrics_for_image(
        torch.from_numpy(rec), torch.from_numpy(gt).reshape(1, 1, *gt.shape)
    )
    reference = structural_similarity(
        rec.astype(np.float64),
        gt.astype(np.float32).astype(np.float64),
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
    )
    assert abs(m["ssim"] - reference) < 0.01


def _record(category, sample_id, delta_psnr, position):
    methods = {}
    for method, psnr in (("ols", 30.0), ("wls", 30.0 + delta_psnr)):
        methods[method] = {
            "status": "PASS",
            "metrics": {
                "mse": 10 ** (-psnr / 10),
                "psnr": psnr,
                "ssim": 0.7,
                "haarpsi": 0.6,
            },
            "runtime_seconds": 0.11,
            "iterations": 17,
            "seconds_per_iteration": 0.11 / 17,
            "calibrated_epsilon": 7.0 if method == "ols" else 115.0,
            "solver": {},
        }
    return {
        "stage": "full",
        "category": category,
        "sample_id": sample_id,
        "permutation_position_1based": position,
        "measurement_tensor_sha256": "0" * 64,
        "source_acquisition_sha256": "1" * 64,
        "source_weight_sha256": "2" * 64,
        "methods": methods,
    }


def _records(deltas):
    return [
        _record(exp.ORDERED_CATEGORIES[i // 10], i + 1, float(d), 901 + i % 10)
        for i, d in enumerate(deltas)
    ]


def test_bootstrap_interval_has_expected_mean_width():
    deltas = np.random.default_rng(1).normal(-0.005, 0.02, 50)
    d = exp.summarize_results(_records(deltas))[1]["ALL"]["metric_differences"]["psnr"]
    assert d["mean_delta_wls_minus_ols"] == pytest.approx(deltas.mean(), abs=1e-12)
    lo, hi = d["mean_delta_bootstrap95_ci"]
    assert lo < deltas.mean() < hi
    half = 1.96 * deltas.std(ddof=1) / np.sqrt(50)
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.15)
    skewed = np.array([-0.01] * 45 + [0.1] * 5)
    lo, hi = exp.summarize_results(_records(skewed))[1]["ALL"]["metric_differences"][
        "psnr"
    ]["mean_delta_bootstrap95_ci"]
    assert lo < skewed.mean() < hi and hi - lo > 0.005


def test_category_medians_and_win_counts_match_numpy():
    deltas = np.random.default_rng(2).normal(-0.005, 0.02, 50)
    summary = exp.summarize_results(_records(deltas))[1]
    assert summary["ALL"]["metric_differences"]["mse"]["win_count_wls"] == int(
        np.sum(deltas > 0)
    )
    assert int(np.sum(deltas > 0)) != int(np.sum(deltas < 0))
    for k, category in enumerate(exp.ORDERED_CATEGORIES):
        part = deltas[10 * k : 10 * k + 10]
        d = summary[category]["metric_differences"]["psnr"]
        assert d["median_delta"] == pytest.approx(np.median(part), abs=1e-12)
        assert d["win_count_wls"] == int(np.sum(part > 0))


def test_resolving_a_case_gives_identical_bytes():
    img = np.full((64, 64), 0.7)
    img[:, 8::16] = 0.2
    img[20:36, 24:44] = 0.35
    op = PhotocurrentMapOp(
        j_order=6,
        s_indices=np.arange(1200),
        normalized=True,
        device="cpu",
    )
    y = op(torch.from_numpy(img)).reshape(-1) + torch.from_numpy(
        0.02 * np.random.default_rng(3).standard_normal(1200)
    )
    d = torch.linspace(10.0, 16.0, 1200, dtype=torch.float64)
    case = wb.Step5Case(
        "Toy",
        1,
        type("Acq", (), {"op": op, "measurement": y})(),
        d,
        1 / d**2,
        "a",
        "w",
    )
    for method, eps in (("ols", 0.72), ("wls", 0.72 * 13)):
        first, _ = wb.solve_step5_case(case, method=method, epsilon=eps)
        second, _ = wb.solve_step5_case(case, method=method, epsilon=eps)
        assert first.numpy().tobytes() == second.numpy().tobytes()


def _patch_anchors(patch, sha_a, sha_w, sha_cal):
    for module in (wb, exp, verify):
        patch.setattr(module, "ACQUISITION_MANIFEST_SHA256", sha_a)
        patch.setattr(module, "WEIGHT_MANIFEST_SHA256", sha_w)
    patch.setattr(verify, "CALIBRATION_MANIFEST_SHA256", sha_cal)


def _fabricate(base, out, calibration, metric):
    """Produce 50 checksummed fake-image records bound to real synthetic A,y."""
    source = base / "acquisitions"
    protocol = json.loads((source / "protocol.json").read_text())
    cases = json.loads((source / "manifest.json").read_text())["acquisitions"]
    ordered = []
    for stage, part in (("pilot", slice(0, 2)), ("full", slice(2, None))):
        for category in exp.ORDERED_CATEGORIES:
            group = sorted(
                (c for c in cases if c["category"] == category),
                key=lambda c: int(c["permutation_position_1based"]),
            )
            ordered += [(stage, c) for c in group[part]]
    out.mkdir()
    records = []
    for n, (stage, c) in enumerate(ordered):
        acquisition, meta = load_master_npz(
            source / c["file"],
            expected_sha256=c["sha256"],
        )
        truth = load_png_float64(
            Path(protocol["data"]["root"]) / meta["source_relpath"]
        )
        record = {
            "stage": stage,
            "category": c["category"],
            "sample_id": int(c["sample_id"]),
            "permutation_position_1based": c["permutation_position_1based"],
            "source_image_sha256": c["source_sha256"],
            "source_acquisition_sha256": c["sha256"],
            "source_weight_sha256": "w" * 64,
            "measurement_tensor_sha256": hashlib.sha256(
                acquisition.measurement.numpy().tobytes()
            ).hexdigest(),
            "methods": {},
        }
        for k, method in enumerate(exp.METHODS):
            noise = (0.02 + 0.002 * k) * np.random.default_rng(
                2 * n + k
            ).standard_normal((256, 256))
            image = (truth[0, 0].numpy() + noise).astype(np.float32)
            relative = f"reconstructions/{c['category']}/sample_{int(c['sample_id']):04d}_{method}.npz"
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            with (out / relative).open("wb") as f:
                np.savez_compressed(
                    f,
                    image=image,
                    category=np.array(c["category"]),
                    sample_id=np.array(int(c["sample_id"]), dtype=np.int64),
                    method=np.array(method),
                )
            record["methods"][method] = {
                "status": "PASS",
                "solver": {
                    "source_acquisition_sha256": c["sha256"],
                    "debias": False,
                    "wavelet_name": "db4",
                    "solver_status": 1,
                    "solver_status_name": "root_found",
                    "solver_converged": True,
                    "feasible_with_tolerance": True,
                    "ols_residual_norm": 7.0,
                    "wls_residual_norm": 115.0,
                },
                "metrics": metric(torch.from_numpy(image), truth),
                "runtime_seconds": 0.11,
                "iterations": 17 + k,
                "seconds_per_iteration": 0.11 / (17 + k),
                "reconstruction_file": relative,
                "reconstruction_sha256": _sha(out / relative),
                "calibrated_epsilon": 7.0 if method == "ols" else 115.0,
            }
        records.append(record)
    write_json_new(out / "pilot_summary.json", exp.summarize_results(records[:10])[1])
    exp._write_manifest(
        out,
        records,
        exp.summarize_results(records)[1],
        iters=120,
        opt_tol=1e-3,
        calibration_sha=_sha(calibration / "manifest.json"),
        stage_status="complete",
    )


@pytest.fixture(scope="module")
def step6(prepared, tmp_path_factory):
    base, sha_a, sha_w = prepared
    root = tmp_path_factory.mktemp("step6_external_review")
    (root / "calibration").mkdir()
    (root / "calibration" / "manifest.json").write_text('{"stand_in":true}\n')
    sha_cal = _sha(root / "calibration" / "manifest.json")
    with pytest.MonkeyPatch.context() as patch:
        _patch_anchors(patch, sha_a, sha_w, sha_cal)
        _fabricate(base, root / "run", root / "calibration", exp.metrics_for_image)
    return base, sha_a, sha_w, sha_cal, root


def _verify(step6, run, monkeypatch, **kwargs):
    base, sha_a, sha_w, sha_cal, root = step6
    _patch_anchors(monkeypatch, sha_a, sha_w, sha_cal)
    return verify.validate_outputs(
        base / "acquisitions",
        base / "weights",
        root / "calibration",
        run,
        expected_manifest_sha256=_sha(run / "manifest.json"),
        **kwargs,
    )


def _copy(step6, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(step6[4] / "run", target)
    for file in target.rglob("*"):
        if file.is_file():
            file.chmod(0o644)
    return target


def _refresh_sums(run, changed):
    lines = []
    for line in (run / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        lines.append(f"{_sha(run/rel) if rel in changed else digest}  {rel}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_verifier_passes_a_consistent_archive(step6, monkeypatch):
    r = _verify(step6, step6[4] / "run", monkeypatch)
    assert r["status"] == "PASS" and r["metrics_recomputed"] == 100


def test_verifier_rejects_swapped_ols_and_wls_files(step6, tmp_path, monkeypatch):
    run = _copy(step6, tmp_path)
    doc = json.loads((run / "manifest.json").read_text())
    item = doc["cases"][0]["methods"]
    a, b = (
        run / item["ols"]["reconstruction_file"],
        run / item["wls"]["reconstruction_file"],
    )
    data_a, data_b = a.read_bytes(), b.read_bytes()
    a.write_bytes(data_b)
    b.write_bytes(data_a)
    item["ols"]["reconstruction_sha256"], item["wls"]["reconstruction_sha256"] = _sha(
        a
    ), _sha(b)
    (run / "manifest.json").write_text(json.dumps(doc))
    _refresh_sums(
        run,
        {
            "manifest.json",
            item["ols"]["reconstruction_file"],
            item["wls"]["reconstruction_file"],
        },
    )
    with pytest.raises(ValueError, match="metadata disagrees"):
        _verify(step6, run, monkeypatch)


def test_verifier_requires_external_anchor(step6, monkeypatch):
    base, sha_a, sha_w, sha_cal, root = step6
    _patch_anchors(monkeypatch, sha_a, sha_w, sha_cal)
    with pytest.raises(ValueError, match="external manifest"):
        verify.validate_outputs(
            base / "acquisitions",
            base / "weights",
            root / "calibration",
            root / "run",
            expected_manifest_sha256="0" * 64,
            recompute_metrics=False,
        )


def test_verifier_rejects_an_edited_summary(step6, tmp_path, monkeypatch):
    run = _copy(step6, tmp_path)
    doc = json.loads((run / "manifest.json").read_text())
    doc["summary"]["ALL"]["metric_differences"]["psnr"][
        "mean_delta_wls_minus_ols"
    ] += 0.01
    (run / "manifest.json").write_text(json.dumps(doc))
    _refresh_sums(run, {"manifest.json"})
    with pytest.raises(ValueError, match="does not regenerate"):
        _verify(step6, run, monkeypatch, recompute_metrics=False)


def test_verifier_rejects_an_edited_metric(step6, tmp_path, monkeypatch):
    run = _copy(step6, tmp_path)
    doc = json.loads((run / "manifest.json").read_text())
    doc["cases"][0]["methods"]["wls"]["metrics"]["psnr"] += 0.01
    (run / "manifest.json").write_text(json.dumps(doc))
    _refresh_sums(run, {"manifest.json"})
    with pytest.raises(ValueError, match="Recomputed psnr differs"):
        _verify(step6, run, monkeypatch)


def test_verifier_rejects_unlisted_reconstruction(step6, tmp_path, monkeypatch):
    run = _copy(step6, tmp_path)
    source = next((run / "reconstructions" / "CIGS").glob("*.npz"))
    shutil.copy(source, run / "reconstructions" / "CIGS" / "extra.npz")
    with pytest.raises(ValueError, match="Unlisted or missing"):
        _verify(step6, run, monkeypatch, recompute_metrics=False)


def test_verifier_catches_wrong_shared_psnr_convention(step6, tmp_path, monkeypatch):
    """Previously strict xfail: +6.02dB shared bug must be caught now."""
    original_metric = exp.metrics_for_image

    def wrong(image, truth):
        actual = original_metric(image, truth)
        return {**actual, "psnr": actual["psnr"] + 20 * np.log10(2)}

    base, sha_a, sha_w, sha_cal, root = step6
    run = tmp_path / "run"
    _patch_anchors(monkeypatch, sha_a, sha_w, sha_cal)
    _fabricate(base, run, root / "calibration", wrong)
    monkeypatch.setattr(verify, "metrics_for_image", wrong)
    with pytest.raises(ValueError, match="Recomputed psnr differs"):
        _verify(step6, run, monkeypatch)
