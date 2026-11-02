"""Step 4 archive safety tests on 50 disposable synthetic validation PNGs.

Builds Step 2/3 only inside pytest temp folders. Produces a 5-image Step 4
Monte Carlo subset (one image per category, 1024 independent draws each).
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from pytests.pcm.test_oracle_wls_step3_e2e import prepared as prepared  # noqa: F401
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import run_validation
from scripts.experiments.pcm.oracle_wls.verify_monte_carlo import (
    verify_saved_monte_carlo,
)
from scripts.experiments.pcm.oracle_wls.float64_acquisition import _sha256


@pytest.fixture(scope="module")
def small_mc(prepared):
    root, acq_sha, weight_sha = prepared
    run_validation(
        root / "acquisitions",
        root / "weights",
        root / "step4",
        expected_acquisition_manifest_sha256=acq_sha,
        expected_weight_manifest_sha256=weight_sha,
        repetitions=1024,
        cases_per_category=1,
        enable_controls=False,
    )
    return root, acq_sha, weight_sha, _sha256(root / "step4/manifest.json")


def test_five_case_mc_archive_verifies_and_replays_byte_exact(small_mc):
    root, a, w, m = small_mc
    result = verify_saved_monte_carlo(
        root / "acquisitions",
        root / "weights",
        root / "step4",
        expected_acquisition_manifest_sha256=a,
        expected_weight_manifest_sha256=w,
        expected_monte_carlo_manifest_sha256=m,
        replay=True,
    )
    assert result["status"] == "PASS"
    assert result["verified_files"] == result["replayed_files"] == 5
    assert result["full_checks_pass"] == result["full_checks_total"] == 5
    assert result["control_checks_total"] == 0


def test_no_heldout_inputs_or_modified_source_files(small_mc):
    root, _, _, _ = small_mc
    evidence = root / "step4"
    assert not (evidence / "test").exists()
    assert len(list((evidence / "validation").rglob("*_mc.npz"))) == 5
    assert len(list((root / "acquisitions/test").iterdir())) == 0
    assert len(list((root / "weights/validation").rglob("*weights.npz"))) == 50


def _copy_evidence(small_mc, tmp_path):
    root, a, w, m = small_mc
    dest = tmp_path / "step4"
    shutil.copytree(root / "step4", dest)
    for file in dest.rglob("*"):
        if file.is_file():
            file.chmod(0o644)
    return root, dest, a, w, m


def test_modified_npz_rejected(small_mc, tmp_path):
    root, dest, a, w, m = _copy_evidence(small_mc, tmp_path)
    record = json.loads((dest / "manifest.json").read_text())["records"][0]
    target = dest / record["file"]
    raw = bytearray(target.read_bytes())
    raw[120] ^= 1
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256SUMS|checksum"):
        verify_saved_monte_carlo(
            root / "acquisitions",
            root / "weights",
            dest,
            expected_acquisition_manifest_sha256=a,
            expected_weight_manifest_sha256=w,
            expected_monte_carlo_manifest_sha256=m,
        )


def test_manifest_external_hash_detects_coordinated_edit(small_mc, tmp_path):
    root, dest, a, w, m = _copy_evidence(small_mc, tmp_path)
    path = dest / "manifest.json"
    doc = json.loads(path.read_text())
    doc["unchecked_change"] = "tampered"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest SHA256"):
        verify_saved_monte_carlo(
            root / "acquisitions",
            root / "weights",
            dest,
            expected_acquisition_manifest_sha256=a,
            expected_weight_manifest_sha256=w,
            expected_monte_carlo_manifest_sha256=m,
        )


def test_reject_existing_output_directory(small_mc):
    root, a, w, m = small_mc
    with pytest.raises(FileExistsError):
        run_validation(
            root / "acquisitions",
            root / "weights",
            root / "step4",
            expected_acquisition_manifest_sha256=a,
            expected_weight_manifest_sha256=w,
            repetitions=1024,
            cases_per_category=1,
            enable_controls=False,
        )


def test_reject_invalid_small_sample_and_category_ranges(small_mc, tmp_path):
    root, a, w, m = small_mc
    with pytest.raises(ValueError, match="1024"):
        run_validation(
            root / "acquisitions",
            root / "weights",
            tmp_path / "bad1",
            expected_acquisition_manifest_sha256=a,
            expected_weight_manifest_sha256=w,
            repetitions=128,
            enable_controls=False,
        )
    with pytest.raises(ValueError, match="between 1 and 10"):
        run_validation(
            root / "acquisitions",
            root / "weights",
            tmp_path / "bad2",
            expected_acquisition_manifest_sha256=a,
            expected_weight_manifest_sha256=w,
            repetitions=1024,
            cases_per_category=12,
            enable_controls=False,
        )
