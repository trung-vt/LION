"""Extra Step 5 reviewer regression cases for archive and solver diagnostics.

Uses disposable one-case calibration files. Covers independently anchored
incomplete calibration, bad coverage gates, and native SPGL1 duality gap.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch

from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.pcm.spgl1_bpdn import spgl1_bpdn_torch
from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared
from pytests.pcm.test_oracle_wls_step5_review import calibrated as calibrated
from scripts.experiments.pcm.oracle_wls import weighted_bpdn as wb


class Identity:
    """Minimal linear identity operator for SPGL1 diagnostic parity."""

    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


def _changed_calibration(calibrated, tmp_path, key, value):
    root, sha_a, sha_w, calibration_dir, rec, calls = calibrated
    target = tmp_path / "calibration"
    shutil.copytree(calibration_dir, target)
    # Step 5 archives intentionally publish read-only files. Copies are made
    # writable only inside this disposable mutation test directory.
    for file in target.iterdir():
        if file.is_file():
            file.chmod(0o644)
    doc = json.loads((target / "manifest.json").read_text())
    doc[key] = value
    (target / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    (target / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((target/name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("manifest.json", "summary.csv")
        )
    )
    return target, doc, rec, sha_a, sha_w


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("status", "partial", "incomplete"),
        ("all_per_case_coverage_in_predeclared_gate", False, "coverage gate"),
        ("number_of_cases", 50, "case count"),
    ],
)
def test_calibration_manifest_rejects_incomplete_or_inconsistent_gates(
    calibrated,
    tmp_path,
    monkeypatch,
    key,
    value,
    error,
):
    target, doc, rec, a, w = _changed_calibration(
        calibrated,
        tmp_path,
        key,
        value,
    )
    monkeypatch.setattr(wb, "ACQUISITION_MANIFEST_SHA256", a)
    monkeypatch.setattr(wb, "WEIGHT_MANIFEST_SHA256", w)
    digest = hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=error):
        wb.load_calibration_record(
            target,
            rec["category"],
            rec["sample_id"],
            expected_manifest_sha256=digest,
        )


def test_native_spgl1_rgap_and_tau_survive_wrapper_without_reinterpreting():
    op = Identity()
    y = torch.tensor([1.0, -0.4, 0.11, 0.02], dtype=torch.float32)
    eps = 0.1
    settings = dict(iter_lim=250, opt_tol=1e-4, verbosity=0)
    sol, diagnostic = spgl1_bpdn_torch(
        op,
        y,
        epsilon=eps,
        iters=settings["iter_lim"],
        opt_tol=settings["opt_tol"],
        verbosity=settings["verbosity"],
    )
    direct, info = spgl1_torch(op, y, sigma=eps, **settings)
    np.testing.assert_allclose(sol.numpy(), direct.numpy(), rtol=0, atol=1e-7)
    assert diagnostic.relative_duality_gap == pytest.approx(
        info["rgap"],
        abs=0,
        rel=0,
    )
    assert diagnostic.spgl1_l1_ball_radius == pytest.approx(
        info["tau"],
        abs=0,
        rel=0,
    )
