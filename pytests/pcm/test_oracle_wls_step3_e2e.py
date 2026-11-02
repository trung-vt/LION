"""End-to-end Step 3 tests on a disposable 50-image synthetic validation set.

All PNGs and acquisition archives are created in pytest temporary directories.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pytests.pcm.test_oracle_wls_step2_review_followup import inputs as make_inputs
from scripts.experiments.pcm.oracle_wls import acquisition_archive
from scripts.experiments.pcm.oracle_wls import build_oracle_weights as builder
from scripts.experiments.pcm.oracle_wls.float64_acquisition import _sha256


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("step3_oracle_e2e")
    protocol, images = make_inputs(base)
    acquisition_archive.build_archive(
        protocol, images, base / "acquisitions", device="cpu"
    )
    anchor = _sha256(base / "acquisitions/manifest.json")
    builder.build_weights(
        base / "acquisitions",
        base / "weights",
        expected_acquisition_manifest_sha256=anchor,
    )
    return base, anchor, _sha256(base / "weights/manifest.json")


def test_builds_fifty_validation_weights_and_recomputes_exactly(prepared):
    base, acquired, weighted = prepared
    result = builder.verify_weights(
        base / "acquisitions",
        base / "weights",
        expected_acquisition_manifest_sha256=acquired,
        expected_weight_manifest_sha256=weighted,
        recompute=True,
    )
    assert result["status"] == "PASS"
    assert result["verified_cases"] == result["recomputed_cases"] == 50
    record = json.loads((base / "weights/manifest.json").read_text())
    assert record["category_counts"] == {
        "CIGS": 10,
        "Organic": 10,
        "Perovskite": 10,
        "Reference": 10,
        "Silicone": 10,
    }
    assert not (base / "weights/test").exists()
    assert len(list((base / "weights/validation").rglob("*.npz"))) == 50


def test_weight_file_has_float64_raw_inverse_weights(prepared):
    base, _, _ = prepared
    root = base / "weights"
    record = json.loads((root / "manifest.json").read_text())["files"][0]
    with np.load(root / record["file"], allow_pickle=False) as data:
        variance = data["variance"].copy()
        weight = data["weight"].copy()
        invstd = data["inverse_std"].copy()
        note = json.loads(str(data["metadata_json"]))
        sampled = data["sampled_indices"].copy()
    assert variance.shape == (13107,) and sampled.dtype == np.uint16
    assert weight.dtype == invstd.dtype == variance.dtype == np.float64
    assert np.array_equal(weight, 1 / variance)
    assert np.array_equal(invstd, 1 / np.sqrt(variance))
    assert note["realized_swap_flags_used"] is False
    assert note["realized_noisy_measurements_used"] is False
    assert np.all(np.isfinite(weight)) and np.all(weight > 0)


def test_realized_y_and_swaps_cannot_change_oracle_weights(prepared, monkeypatch):
    """Changing the noise realization in memory leaves the oracle unchanged."""
    base, _, _ = prepared
    cases = json.loads((base / "acquisitions/manifest.json").read_text())[
        "acquisitions"
    ]
    case = cases[0]
    protocol = json.loads((base / "acquisitions/protocol.json").read_text())
    original, _, _ = builder._oracle_for_case(case, base / "acquisitions", protocol)
    genuine_loader = builder.load_master_npz

    def changed_noise_realization(*args, **kwargs):
        result, metadata = genuine_loader(*args, **kwargs)
        altered = replace(
            result,
            measurement=torch.full_like(result.measurement, 1e9),
            swap_flags=~result.swap_flags,
        )
        return altered, metadata

    monkeypatch.setattr(builder, "load_master_npz", changed_noise_realization)
    modified, _, _ = builder._oracle_for_case(case, base / "acquisitions", protocol)
    assert np.array_equal(original.variance, modified.variance)
    assert np.array_equal(original.weight, modified.weight)


def test_tampered_oracle_npz_rejected(prepared, tmp_path):
    base, acquired, weighted = prepared
    copied = tmp_path / "weights"
    shutil.copytree(base / "weights", copied)
    record = json.loads((copied / "manifest.json").read_text())["files"][0]
    item = copied / record["file"]
    item.chmod(0o644)
    raw = bytearray(item.read_bytes())
    raw[len(raw) // 2] ^= 1
    item.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        builder.verify_weights(
            base / "acquisitions",
            copied,
            expected_acquisition_manifest_sha256=acquired,
            expected_weight_manifest_sha256=weighted,
        )


def test_external_manifest_anchor_rejects_modified_manifest(prepared, tmp_path):
    base, acquired, weighted = prepared
    copied = tmp_path / "weights"
    shutil.copytree(base / "weights", copied)
    path = copied / "manifest.json"
    path.chmod(0o644)
    doc = json.loads(path.read_text())
    doc["tampered"] = True
    path.write_text(json.dumps(doc) + "\n")
    with pytest.raises(ValueError, match="externally|Externally"):
        builder.verify_weights(
            base / "acquisitions",
            copied,
            expected_acquisition_manifest_sha256=acquired,
            expected_weight_manifest_sha256=weighted,
        )


def test_refuses_to_overwrite_existing_oracle_output(prepared):
    base, acquired, _ = prepared
    with pytest.raises(FileExistsError):
        builder.build_weights(
            base / "acquisitions",
            base / "weights",
            expected_acquisition_manifest_sha256=acquired,
        )
