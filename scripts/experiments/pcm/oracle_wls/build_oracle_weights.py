"""Build or verify Step 3 oracle variance files for the frozen validation archive.

No existing acquisition or image file is modified. The 450 held-out cases are
not touched. Use the independently pinned Step 2 manifest SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from LION.pcm.simulation import _sample_split_measurements
from scripts.experiments.pcm.oracle_wls.acquisition_archive import verify_archive
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    _write_atomic_new,
    load_master_npz,
    load_png_float64,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.oracle_variance import (
    oracle_marginal_variance,
)

STEP3_SCHEMA = "pcm-oracle-marginal-variance-v1"
RECORD_KEYS = (
    "variance",
    "weight",
    "inverse_std",
    "variance_dark",
    "variance_bias_white",
    "variance_bias_drift",
    "variance_beam",
    "sampled_indices",
)


def _check_sidecar(folder: Path, expected: set[str]) -> None:
    """Verify the checksum sidecar covers the complete weight archive."""
    sidecar = folder / "SHA256SUMS"
    seen = {}
    for entry in sidecar.read_text().splitlines():
        sha, path = entry.split("  ", 1)
        if path.startswith("/") or ".." in Path(path).parts or path in seen:
            raise ValueError("Invalid Step 3 checksum entry")
        seen[path] = sha
        if _sha256(folder / path) != sha:
            raise ValueError(f"Step 3 checksum mismatch: {path}")
    if set(seen) != expected:
        raise ValueError("Incomplete or unexpected Step 3 checksum entries")


def _oracle_for_case(case, acquisition_root: Path, acquisition_protocol: dict):
    """Calculate from *ground truth and A*, never from noisy y or swap flags."""
    path = acquisition_root / case["file"]
    acquisition, metadata = load_master_npz(path, expected_sha256=case["sha256"])
    image_path = Path(acquisition_protocol["data"]["root"]) / metadata["source_relpath"]
    if _sha256(image_path) != metadata["source_sha256"]:
        raise ValueError("Source PNG has changed since the frozen acquisition")
    x = load_png_float64(image_path)
    p, q = _sample_split_measurements(x, acquisition.op)
    clean = p - q
    torch.testing.assert_close(clean.cpu(), acquisition.op(x).cpu(), rtol=0, atol=2e-12)
    i0 = float(acquisition.i0.reshape(-1)[0])
    expected_i0 = float(
        (x.sum() / 256 / acquisition_protocol["noise"]["signal_to_bias_ratio"]).item()
    )
    if abs(i0 - expected_i0) > 2e-12:
        raise ValueError("Stored I0 differs from the ground-truth model")
    parameters = acquisition_protocol["noise"]
    # Beam/noise variances depend on ground-truth split currents, I0 and the
    # locked noise model. The saved realised swaps and noisy y are NOT read.
    terms = oracle_marginal_variance(
        p.reshape(-1).cpu().numpy(),
        q.reshape(-1).cpu().numpy(),
        i0,
        beam_ar=parameters["beam_ar"],
        beam_ma=parameters["beam_ma"],
        beam_sigma=parameters["beam_sigma"],
        bias_sigma=parameters["bias_sigma"],
        bias_gradient_sigma=parameters["bias_gradient_sigma"],
        dark_sigma=parameters["dark_sigma"],
        repeat_acquisitions=parameters["repeat_acquisitions"],
    )
    return terms, metadata, np.asarray(acquisition.sampled_indices, dtype=np.uint16)


def _arrays(terms, metadata, indices, case, acquisition_manifest_sha256):
    arrays = {
        "variance": terms.variance,
        "weight": terms.weight,
        "inverse_std": terms.inverse_std,
        "variance_dark": terms.dark,
        "variance_bias_white": terms.bias_white,
        "variance_bias_drift": terms.bias_drift,
        "variance_beam": terms.beam_integrated_arma,
        "sampled_indices": indices,
        "metadata_json": json.dumps(
            {
                "schema": STEP3_SCHEMA,
                "split": "validation",
                "category": case["category"],
                "sample_id": case["sample_id"],
                "source_png_sha256": metadata["source_sha256"],
                "source_acquisition_file": case["file"],
                "source_acquisition_sha256": case["sha256"],
                "source_acquisition_manifest_sha256": acquisition_manifest_sha256,
                "source_protocol_id": metadata["protocol_id"],
                "oracle_conditioning": "ground-truth x and A; marginal over noise and random swaps",
                "realized_swap_flags_used": False,
                "realized_noisy_measurements_used": False,
                "units": "normalized differential measurement units",
            },
            sort_keys=True,
        ),
    }
    assert all(
        arrays[k].dtype == np.float64 for k in RECORD_KEYS if k != "sampled_indices"
    )
    return arrays


def build_weights(
    acquisition_root, destination, *, expected_acquisition_manifest_sha256
):
    """Calculate 50 independent float64 diagonal oracle-weight files."""
    source = Path(acquisition_root)
    out = Path(destination)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing Step 3 result: {out}")
    archive_validation = verify_archive(
        source,
        expected_manifest_sha256=expected_acquisition_manifest_sha256,
    )
    if archive_validation["verified_files"] != 50:
        raise ValueError("Step 3 expects 50 checked validation acquisitions")
    protocol = json.loads((source / "protocol.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if protocol["noise"]["repeat_acquisitions"] != 1:
        raise ValueError("The locked Step 3 test uses one acquisition repeat")
    if manifest["test_count"] or len(manifest["acquisitions"]) != 50:
        raise ValueError(
            "No held-out acquisitions or changed validation counts allowed"
        )
    out.mkdir(parents=True, exist_ok=False)

    records = []
    for case in manifest["acquisitions"]:
        terms, metadata, indices = _oracle_for_case(case, source, protocol)
        arrays = _arrays(
            terms, metadata, indices, case, expected_acquisition_manifest_sha256
        )
        filename = (
            f"validation/{case['category']}/"
            f"sample_{case['sample_id']:04d}_r0_oracle_weights.npz"
        )
        file_path = out / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic_new(
            file_path,
            lambda f: np.savez_compressed(f, **arrays),
        )
        file_path.chmod(0o444)
        records.append(
            {
                "file": filename,
                "sha256": _sha256(file_path),
                "bytes": file_path.stat().st_size,
                "category": case["category"],
                "sample_id": case["sample_id"],
                "source_acquisition_sha256": case["sha256"],
                "variance_min": float(terms.variance.min()),
                "variance_median": float(np.median(terms.variance)),
                "variance_max": float(terms.variance.max()),
                "weight_min": float(terms.weight.min()),
                "weight_median": float(np.median(terms.weight)),
                "weight_max": float(terms.weight.max()),
                "mean_beam_variance": float(terms.beam_integrated_arma.mean()),
                "mean_dark_variance": float(terms.dark.mean()),
                "mean_bias_white_variance": float(terms.bias_white.mean()),
                "mean_bias_drift_variance": float(terms.bias_drift.mean()),
            }
        )
    provenance = {
        "schema": STEP3_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step": 3,
        "status": "analytical_variances_calculated_monte_carlo_not_yet_validated",
        "source_acquisitions_folder": str(source),
        "source_manifest_sha256": expected_acquisition_manifest_sha256,
        "source_protocol_sha256": _sha256(source / "protocol.json"),
        "oracle_variance_source_sha256": _sha256(
            Path(__file__).with_name("oracle_variance.py")
        ),
        "builder_source_sha256": _sha256(Path(__file__)),
        "condition": "x,A only; marginal over Gaussian sources and random swaps",
        "beam_model": "integrated ARMA, finite zero initial state, exact impulse response",
        "diagonal_only": True,
        "full_covariance_included": False,
        "white_residual": "inverse_std = 1 / sqrt(variance)",
        "raw_weight": "weight = 1 / variance",
        "solver_integration": "not implemented in Step 3",
        "monte_carlo_validation": "reserved for Step 4",
        "test_acquisitions_generated": 0,
    }
    write_json_new(out / "protocol.json", provenance)
    result = {
        "schema": STEP3_SCHEMA,
        "count": len(records),
        "status": "complete",
        "protocol_sha256": _sha256(out / "protocol.json"),
        "source_manifest_sha256": expected_acquisition_manifest_sha256,
        "category_counts": {
            cat: sum(record["category"] == cat for record in records)
            for cat in sorted({record["category"] for record in records})
        },
        "files": records,
    }
    write_json_new(out / "manifest.json", result)
    all_files = ["protocol.json", "manifest.json"] + [r["file"] for r in records]
    sha_text = "".join(f"{_sha256(out / rel)}  {rel}\n" for rel in all_files)
    _write_atomic_new(out / "SHA256SUMS", lambda f: f.write(sha_text.encode()))
    print("EXTERNAL_STEP3_MANIFEST_SHA256=" + _sha256(out / "manifest.json"))
    return result


def verify_weights(
    acquisition_root,
    weight_root,
    *,
    expected_acquisition_manifest_sha256,
    expected_weight_manifest_sha256=None,
    recompute=False,
):
    """Check weight integrity/binding, and optionally recompute every array."""
    source, out = Path(acquisition_root), Path(weight_root)
    verify_archive(
        source,
        expected_manifest_sha256=expected_acquisition_manifest_sha256,
    )
    if (
        expected_weight_manifest_sha256 is not None
        and _sha256(out / "manifest.json") != expected_weight_manifest_sha256
    ):
        raise ValueError("Externally anchored Step 3 manifest hash mismatch")
    manifest = json.loads((out / "manifest.json").read_text())
    metadata = json.loads((out / "protocol.json").read_text())
    # The protocol must agree with the version pinned when this weight archive
    # was created. Updating the sidecar alone must not hide a changed protocol.
    if _sha256(out / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Step 3 protocol checksum differs from weight manifest")
    if (
        manifest["schema"] != STEP3_SCHEMA
        or metadata["schema"] != STEP3_SCHEMA
        or metadata["diagonal_only"] is not True
        or metadata["full_covariance_included"] is not False
    ):
        raise ValueError("Step 3 oracle protocol or covariance convention changed")
    if metadata["source_manifest_sha256"] != expected_acquisition_manifest_sha256:
        raise ValueError("Step 3 source manifest identity differs")
    if metadata["source_protocol_sha256"] != _sha256(source / "protocol.json"):
        raise ValueError("Step 3 acquisition protocol identity differs")
    cases = json.loads((source / "manifest.json").read_text())["acquisitions"]
    lookup = {(c["category"], c["sample_id"]): c for c in cases}
    if len(manifest["files"]) != 50 or set(lookup) != {
        (r["category"], r["sample_id"]) for r in manifest["files"]
    }:
        raise ValueError("Oracle-weight archive lacks complete case coverage")
    _check_sidecar(
        out, {"manifest.json", "protocol.json"} | {r["file"] for r in manifest["files"]}
    )
    source_protocol = json.loads((source / "protocol.json").read_text())
    for record in manifest["files"]:
        case = lookup[(record["category"], record["sample_id"])]
        filename = out / record["file"]
        if (
            _sha256(filename) != record["sha256"]
            or filename.stat().st_size != record["bytes"]
        ):
            raise ValueError("Oracle file checksum or size mismatch")
        with np.load(filename, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in RECORD_KEYS}
            note = json.loads(str(data["metadata_json"]))
        if note["source_acquisition_sha256"] != case["sha256"]:
            raise ValueError("Weight source measurement hash mismatch")
        if (
            note["source_acquisition_manifest_sha256"]
            != expected_acquisition_manifest_sha256
        ):
            raise ValueError("Step 3 weights bound to another acquisition manifest")
        v, w, iw = (arrays[k] for k in ("variance", "weight", "inverse_std"))
        if v.dtype != np.float64 or not np.isfinite(v).all() or (v <= 0).any():
            raise ValueError("Invalid float64 oracle variance")
        if not (np.array_equal(w, 1 / v) and np.array_equal(iw, 1 / np.sqrt(v))):
            raise ValueError("Invalid inverse-variance or inverse-STD coefficients")
        components = [
            arrays[k]
            for k in (
                "variance_dark",
                "variance_bias_white",
                "variance_bias_drift",
                "variance_beam",
            )
        ]
        np.testing.assert_allclose(v, sum(components), rtol=4e-15, atol=0)
        original, original_metadata = load_master_npz(
            source / case["file"],
            expected_sha256=case["sha256"],
        )
        if (
            note["source_png_sha256"] != original_metadata["source_sha256"]
            or note["source_protocol_id"] != original_metadata["protocol_id"]
        ):
            raise ValueError("Oracle source-image or protocol identity mismatch")
        if not np.array_equal(original.sampled_indices, arrays["sampled_indices"]):
            raise ValueError("Weights are attached to a different sensing operator")
        if recompute:
            fresh, original_metadata, indices = _oracle_for_case(
                case, source, source_protocol
            )
            expected = _arrays(
                fresh,
                original_metadata,
                indices,
                case,
                expected_acquisition_manifest_sha256,
            )
            for key in RECORD_KEYS:
                if arrays[key].tobytes() != expected[key].tobytes():
                    raise AssertionError(f"Oracle analytic regeneration differs: {key}")
    # Internal file and manifest checksums are not an independent trust
    # anchor. Without an externally saved weight-manifest digest, recompute
    # every analytic weight before reporting a successful verification.
    # Check the per-case source bindings first so swaps are reported precisely.
    if expected_weight_manifest_sha256 is None and not recompute:
        raise ValueError(
            "Weight verification requires an external manifest SHA256 "
            "or analytical recomputation (recompute=True)"
        )
    return {
        "status": "PASS",
        "verified_cases": len(manifest["files"]),
        "recomputed_cases": len(manifest["files"]) if recompute else 0,
        "weights_dtype": "float64",
        "source_manifest_sha256": expected_acquisition_manifest_sha256,
        "weight_manifest_sha256": _sha256(out / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("--acquisitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-acquisition-manifest-sha256", required=True)
    parser.add_argument("--expected-weight-manifest-sha256")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "build":
        result = build_weights(
            args.acquisitions,
            args.output,
            expected_acquisition_manifest_sha256=args.expected_acquisition_manifest_sha256,
        )
    else:
        result = verify_weights(
            args.acquisitions,
            args.output,
            expected_acquisition_manifest_sha256=args.expected_acquisition_manifest_sha256,
            expected_weight_manifest_sha256=args.expected_weight_manifest_sha256,
            recompute=args.recompute,
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("files",)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
