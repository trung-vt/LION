"""External Step 3 review: archive mutation and external trust-anchor tests.

Tests reproduce the reviewer's coherent mutation probes on disposable
synthetic 50-image archives, never on the saved PCM data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pytests.pcm.test_oracle_wls_step2_review_followup import inputs
from scripts.experiments.pcm.oracle_wls import acquisition_archive
from scripts.experiments.pcm.oracle_wls import build_oracle_weights as builder

REPO = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _in_repo_root():
    previous = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("step3_external_review")
    parent, manifest = inputs(base)
    with _in_repo_root():
        acquisition_archive.build_archive(
            parent, manifest, base / "acquisitions", device="cpu"
        )
    anchor = _sha(base / "acquisitions" / "manifest.json")
    builder.build_weights(
        base / "acquisitions",
        base / "weights",
        expected_acquisition_manifest_sha256=anchor,
    )
    return base, anchor, _sha(base / "weights" / "manifest.json")


def _copy(base, tmp_path):
    target = tmp_path / "weights"
    shutil.copytree(base / "weights", target)
    for path in target.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return target


def _refresh_sums(root, relatives):
    lines = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        lines.append(
            f"{_sha(root / relative) if relative in relatives else digest}  {relative}"
        )
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def _update_record(root, index, path):
    doc = json.loads((root / "manifest.json").read_text())
    doc["files"][index].update(sha256=_sha(path), bytes=path.stat().st_size)
    (root / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")


def _double_first_case(root):
    """Tamper with one case but update every internal consistency checksum."""
    record = json.loads((root / "manifest.json").read_text())["files"][0]
    path = root / record["file"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    for key in (
        "variance",
        "variance_dark",
        "variance_bias_white",
        "variance_bias_drift",
        "variance_beam",
    ):
        arrays[key] = 2 * arrays[key]
    arrays["weight"] = 1 / arrays["variance"]
    arrays["inverse_std"] = 1 / np.sqrt(arrays["variance"])
    path.unlink()
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _update_record(root, 0, path)
    _refresh_sums(root, {"manifest.json", record["file"]})


def test_swapped_weight_files_rejected_without_weight_manifest_anchor(built, tmp_path):
    base, anchor, _ = built
    root = _copy(base, tmp_path)
    files = json.loads((root / "manifest.json").read_text())["files"]
    first, second = root / files[0]["file"], root / files[1]["file"]
    a, b = first.read_bytes(), second.read_bytes()
    first.write_bytes(b)
    second.write_bytes(a)
    _update_record(root, 0, first)
    _update_record(root, 1, second)
    _refresh_sums(root, {"manifest.json", files[0]["file"], files[1]["file"]})
    with pytest.raises(ValueError, match="source measurement hash"):
        builder.verify_weights(
            base / "acquisitions",
            root,
            expected_acquisition_manifest_sha256=anchor,
        )


def test_anchor_or_recompute_catches_consistent_edit(built, tmp_path):
    base, anchor, weight_anchor = built
    root = _copy(base, tmp_path)
    _double_first_case(root)
    with pytest.raises(ValueError, match="Externally anchored"):
        builder.verify_weights(
            base / "acquisitions",
            root,
            expected_acquisition_manifest_sha256=anchor,
            expected_weight_manifest_sha256=weight_anchor,
        )
    with pytest.raises(AssertionError, match="regeneration differs"):
        builder.verify_weights(
            base / "acquisitions",
            root,
            expected_acquisition_manifest_sha256=anchor,
            recompute=True,
        )


def test_verify_requires_anchor_or_recompute(built, tmp_path):
    base, anchor, _ = built
    root = _copy(base, tmp_path)
    _double_first_case(root)
    with pytest.raises(ValueError):
        builder.verify_weights(
            base / "acquisitions",
            root,
            expected_acquisition_manifest_sha256=anchor,
        )


def test_verify_checks_weight_protocol_against_manifest(built, tmp_path):
    base, anchor, weight_anchor = built
    root = _copy(base, tmp_path)
    protocol = json.loads((root / "protocol.json").read_text())
    protocol.update(
        diagonal_only=False,
        oracle_variance_source_sha256="0" * 64,
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    _refresh_sums(root, {"protocol.json"})
    with pytest.raises(ValueError):
        builder.verify_weights(
            base / "acquisitions",
            root,
            expected_acquisition_manifest_sha256=anchor,
            expected_weight_manifest_sha256=weight_anchor,
        )


def test_valid_weights_require_external_anchor_or_recomputation(built):
    """Even a clean-looking archive needs an independent integrity gate."""
    base, anchor, weight_anchor = built
    with pytest.raises(ValueError, match="requires an external manifest"):
        builder.verify_weights(
            base / "acquisitions",
            base / "weights",
            expected_acquisition_manifest_sha256=anchor,
        )
    assert (
        builder.verify_weights(
            base / "acquisitions",
            base / "weights",
            expected_acquisition_manifest_sha256=anchor,
            expected_weight_manifest_sha256=weight_anchor,
        )["status"]
        == "PASS"
    )
    assert (
        builder.verify_weights(
            base / "acquisitions",
            base / "weights",
            expected_acquisition_manifest_sha256=anchor,
            recompute=True,
        )["recomputed_cases"]
        == 50
    )
