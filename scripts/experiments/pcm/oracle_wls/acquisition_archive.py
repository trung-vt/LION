"""Build and verify the 50-case validation archive for oracle WLS/OLS.

Run as a module from the LION root. No test-set generation is provided here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    ARCHIVE_SCHEMA,
    _sha256,
    _write_atomic_new,
    create_float64_acquisition,
    load_master_npz,
    save_master_npz,
    write_json_new,
)

CODE_FILES = (
    "LION/pcm/simulation.py",
    "LION/pcm/noise.py",
    "LION/pcm_tools/noise_generate.py",
    "LION/pcm_tools/measure.py",
    "LION/pcm_tools/pcm_sampling.py",
    "LION/operators/PhotocurrentMapOp.py",
    "scripts/experiments/pcm/oracle_wls/float64_acquisition.py",
    "scripts/experiments/pcm/oracle_wls/acquisition_archive.py",
)
CROSS_DEVICE_ATOL = 2e-12


def validation_rows(manifest_bytes, protocol):
    """Select exactly the approved 10 validation positions per category."""
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"))))
    selected = [r for r in rows if r["split"] == "validation"]
    expected = {(c, p) for c in protocol["data"]["categories"] for p in range(901, 911)}
    actual = [(r["category"], int(r["permutation_position_1based"])) for r in selected]
    if len(actual) != 50 or len(set(actual)) != 50 or set(actual) != expected:
        raise ValueError("Expected precisely 50 validation rows, positions 901-910")
    if len({(r["category"], r["sample_id"]) for r in selected}) != 50:
        raise ValueError("Duplicate validation source image")
    return selected


def _execution(device):
    d = torch.device(device)
    if d.type == "cuda" and "4070" not in torch.cuda.get_device_name(d):
        raise RuntimeError("This Step 2 GPU run is restricted to the approved RTX 4070")
    return {
        "device_type": d.type,
        "device": str(d),
        "gpu_name": torch.cuda.get_device_name(d) if d.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scipy": version("scipy"),
        "statsmodels": version("statsmodels"),
        "pillow": version("Pillow"),
        "cuda_runtime": torch.version.cuda,
        "torch_cpu_threads": torch.get_num_threads(),
    }


def build_archive(parent_protocol, source_manifest, output, *, device="cpu"):
    """Create a new archive; incomplete or existing directories are never overwritten."""
    parent_protocol, source_manifest, output = map(
        Path, (parent_protocol, source_manifest, output)
    )
    protocol = json.loads(parent_protocol.read_text())
    if _sha256(source_manifest) != protocol["data"]["manifest_sha256"]:
        raise ValueError("Pinned image/seed manifest checksum mismatch")
    manifest_bytes = source_manifest.read_bytes()
    rows = validation_rows(manifest_bytes, protocol)
    required_noise = {
        "beam_ar",
        "beam_ma",
        "beam_sigma",
        "bias_sigma",
        "bias_gradient_sigma",
        "dark_sigma",
        "signal_to_bias_ratio",
        "current_to_model_scale",
        "repeat_acquisitions",
    }
    if not required_noise.issubset(protocol["noise"]):
        raise ValueError(
            "The noise parameters must be explicit, not only a preset name"
        )
    if (
        protocol["sampling"]["sampling_ratio_requested"],
        protocol["sampling"]["coarse_j"],
        protocol["noise"]["repeat_acquisitions"],
    ) != (0.2, 6, 1):
        raise ValueError(
            "This archive locks 20% sampling, coarse_j=6, one acquisition repeat"
        )
    execution = _execution(device)
    # Code hashes include the dirty working-tree files actually executed, not only HEAD.
    code_hashes = {p: _sha256(Path(p)) for p in CODE_FILES}
    protocol.update(
        protocol_id="pcm-oracle-wls-ols-v3-step2-storage",
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="STEP_2_VALIDATION_STORAGE_SPECIFICATION",
        revision_of={
            "filename": "parent_float64_protocol.json",
            "sha256": _sha256(parent_protocol),
        },
        storage={
            "archive_schema": ARCHIVE_SCHEMA,
            "split": "validation",
            "cases": 50,
            "test_cases": 0,
            "swap_bitorder": "little",
            "read_only_files": True,
        },
        oracle_policy={
            "primary": "Var(y-Ax | x,A), marginal over random swaps",
            "realised_swaps": "audit/replay only; not used for the primary weights",
            "weights_implemented": False,
        },
        execution=execution,
        executed_source_sha256=code_hashes,
    )
    for field in (
        "acquisitions_created",
        "pilot_validation_acquisitions_created",
        "production_acquisitions_created",
    ):
        protocol.pop(field, None)
    protocol["precision"][
        "shared_simulator_edited"
    ] = "Optional swap recording only; numerical model unchanged"
    protocol["next_step_requirements"] = [
        "Step 3: implement float64 marginal oracle variances and weights",
        "Step 4: Monte Carlo verification of the analytic variances",
        "Steps 5-6: weighted BPDN and validation pilot before freezing 450 test acquisitions",
    ]
    output.mkdir(parents=True, exist_ok=False)
    (
        output / "test"
    ).mkdir()  # Intentionally empty: held-out acquisitions are not generated here.
    _write_atomic_new(
        output / "parent_float64_protocol.json",
        lambda f: f.write(parent_protocol.read_bytes()),
    )
    _write_atomic_new(
        output / "step1_image_and_seed_manifest.csv", lambda f: f.write(manifest_bytes)
    )
    write_json_new(output / "protocol.json", protocol)
    records = []
    for row in rows:
        acquisition, metadata = create_float64_acquisition(
            row=row, protocol=protocol, device=device
        )
        relative = (
            f"validation/{row['category']}/sample_{int(row['sample_id']):04d}_r0.npz"
        )
        path = output / relative
        save_master_npz(path, acquisition, metadata)
        checksum = _sha256(path)
        loaded, loaded_metadata = load_master_npz(path, expected_sha256=checksum)
        for field in ("measurement", "i0", "swap_flags"):
            if not torch.equal(
                getattr(loaded, field).cpu(), getattr(acquisition, field).cpu()
            ):
                raise AssertionError(f"Lossless round trip failed: {relative}, {field}")
        if (
            not np.array_equal(loaded.sampled_indices, acquisition.sampled_indices)
            or loaded_metadata != metadata
        ):
            raise AssertionError(
                f"Round-trip metadata or row-order mismatch: {relative}"
            )
        if (
            _sha256(Path(protocol["data"]["root"]) / row["source_relpath"])
            != row["source_sha256"]
        ):
            raise AssertionError("Source image changed during generation")
        records.append(
            {
                "file": relative,
                "sha256": checksum,
                "bytes": path.stat().st_size,
                "category": row["category"],
                "sample_id": int(row["sample_id"]),
                "permutation_position_1based": int(row["permutation_position_1based"]),
                "source_sha256": row["source_sha256"],
                "swapped_pairs": int(acquisition.swap_flags.sum()),
            }
        )
        print(f"Stored {len(records):02d}/50: {relative}", flush=True)
    if {p: _sha256(Path(p)) for p in CODE_FILES} != code_hashes:
        raise RuntimeError("Executed source files changed during archive generation")
    manifest = {
        "archive_schema": ARCHIVE_SCHEMA,
        "status": "complete",
        "frozen": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(output / "protocol.json"),
        "source_manifest_sha256": _sha256(source_manifest),
        "validation_count": 50,
        "test_count": 0,
        "category_counts": dict(Counter(r["category"] for r in records)),
        "npz_bytes": sum(r["bytes"] for r in records),
        "lossless_round_trips_passed": 50,
        "source_images_unchanged": True,
        "acquisitions": records,
    }
    write_json_new(output / "manifest.json", manifest)
    return manifest


def verify_archive(root, *, device="cpu", replay=False, cpu_crosscheck=False):
    """Verify files and A/y loading; optionally regenerate the same seeded acquisitions."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if _sha256(root / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Archive protocol checksum mismatch")
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        _sha256(root / "parent_float64_protocol.json")
        != protocol["revision_of"]["sha256"]
    ):
        raise ValueError("Parent precision protocol checksum mismatch")
    if (
        _sha256(root / "step1_image_and_seed_manifest.csv")
        != manifest["source_manifest_sha256"]
    ):
        raise ValueError("Image manifest checksum mismatch")
    rows = validation_rows(
        (root / "step1_image_and_seed_manifest.csv").read_bytes(), protocol
    )
    lookup = {(r["category"], int(r["sample_id"])): r for r in rows}
    keys = [(r["category"], r["sample_id"]) for r in manifest["acquisitions"]]
    if (
        len(keys) != 50
        or set(keys) != set(lookup)
        or manifest["validation_count"] != 50
    ):
        raise ValueError("Archive coverage differs from the approved validation split")
    if list((root / "test").rglob("*.npz")) or manifest["test_count"] != 0:
        raise ValueError("Held-out test acquisitions must remain absent in Step 2")
    expected_files = {r["file"] for r in manifest["acquisitions"]}
    actual_files = {
        str(p.relative_to(root)) for p in (root / "validation").rglob("*.npz")
    }
    if expected_files != actual_files:
        raise ValueError("Unlisted or missing validation files")
    if replay:
        _execution(device)
        for path, digest in protocol["executed_source_sha256"].items():
            if _sha256(Path(path)) != digest:
                raise ValueError(f"Replay source differs from recorded code: {path}")
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "verified_files": 0,
        "replayed_files": 0,
        "bit_exact_replays": 0,
        "cpu_crosschecks": 0,
        "max_cpu_gpu_measurement_difference": 0.0,
        "max_cpu_gpu_i0_difference": 0.0,
        "cross_device_atol": CROSS_DEVICE_ATOL,
        "test_acquisitions": 0,
        "checksums_ok": True,
        "same_A_y_loader_ok": True,
    }
    same_backend = torch.device(device).type == protocol["execution"]["device_type"]
    for record in manifest["acquisitions"]:
        path = root / record["file"]
        loaded, metadata = load_master_npz(path, expected_sha256=record["sha256"])
        row = lookup[(record["category"], record["sample_id"])]
        for key in ("category", "source_relpath", "source_sha256", "split"):
            if metadata[key] != row[key]:
                raise ValueError(f"Metadata mismatch: {record['file']}, {key}")
        for key in (
            "sample_id",
            "permutation_position_1based",
            "mask_seed_uint64",
            "noise_seed_uint64",
        ):
            if int(metadata[key]) != int(row[key]):
                raise ValueError(
                    f"Seed/image identity mismatch: {record['file']}, {key}"
                )
        if metadata["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("Protocol identity mismatch")
        if record["bytes"] != path.stat().st_size or record["swapped_pairs"] != int(
            loaded.swap_flags.sum()
        ):
            raise ValueError("Recorded file size or swap count mismatch")
        if (
            loaded.measurement.dtype != torch.float64
            or loaded.op.normalization_factor != 1 / 256
        ):
            raise ValueError("Incorrect dtype or reconstructed forward operator")
        if replay:
            regenerated, _ = create_float64_acquisition(
                row=row, protocol=protocol, device=device
            )
            for field in ("measurement", "i0"):
                a, b = getattr(loaded, field).cpu(), getattr(regenerated, field).cpu()
                if same_backend and a.numpy().tobytes() != b.numpy().tobytes():
                    raise AssertionError(
                        f"Same-backend replay not bit-exact: {record['file']}, {field}"
                    )
                torch.testing.assert_close(a, b, rtol=0, atol=CROSS_DEVICE_ATOL)
            if not torch.equal(
                loaded.swap_flags, regenerated.swap_flags
            ) or not np.array_equal(
                loaded.sampled_indices, regenerated.sampled_indices
            ):
                raise AssertionError("Replay swaps or sampled indices differ")
            result["replayed_files"] += 1
            result["bit_exact_replays"] += int(
                loaded.measurement.numpy().tobytes()
                == regenerated.measurement.cpu().numpy().tobytes()
            )
        if cpu_crosscheck:
            cpu, _ = create_float64_acquisition(
                row=row, protocol=protocol, device="cpu"
            )
            for field, key in (
                ("measurement", "max_cpu_gpu_measurement_difference"),
                ("i0", "max_cpu_gpu_i0_difference"),
            ):
                a, b = getattr(loaded, field).cpu(), getattr(cpu, field).cpu()
                torch.testing.assert_close(a, b, rtol=0, atol=CROSS_DEVICE_ATOL)
                result[key] = max(result[key], float((a - b).abs().max()))
            if not torch.equal(loaded.swap_flags, cpu.swap_flags) or not np.array_equal(
                loaded.sampled_indices, cpu.sampled_indices
            ):
                raise AssertionError("CPU/GPU swaps or masks differ")
            result["cpu_crosschecks"] += 1
        result["verified_files"] += 1
    result["status"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--cpu-crosscheck", action="store_true")
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "build":
        if args.protocol is None or args.manifest is None:
            parser.error("build requires --protocol and --manifest")
        result = build_archive(
            args.protocol, args.manifest, args.root, device=args.device
        )
    else:
        result = verify_archive(
            args.root,
            device=args.device,
            replay=args.replay,
            cpu_crosscheck=args.cpu_crosscheck,
        )
        if args.verification_output:
            write_json_new(args.verification_output, result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "acquisitions"}, indent=2)
    )


if __name__ == "__main__":
    main()
