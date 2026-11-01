"""Float64-only synthetic PCM acquisition for the oracle WLS/OLS study.

The numerical PCM model, dataset loader, and reconstruction solvers are unchanged.
The shared simulator only adds optional recording of existing swap draws.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import (
    PCMAcquisition,
    acquisition_from_prepared_measurement,
    simulate_pcm_acquisition,
)
from LION.pcm_tools.pcm_sampling import sample_indices


ARCHIVE_SCHEMA = "pcm-float64-swaps-v1"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source or manifest file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_png_float64(path: Path) -> torch.Tensor:
    """Read 8-bit PCM PNG pixels into float64 before normalisation."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def create_float64_acquisition(
    *, row: dict, protocol: dict, device: str | torch.device = "cpu"
) -> tuple[PCMAcquisition, dict]:
    """Generate a deterministic float64 acquisition from a pinned Step 1 row."""
    data = protocol["data"]
    sampling = protocol["sampling"]
    noise_settings = protocol["noise"]
    if protocol["precision"]["acquisition_compute_dtype"] != "float64":
        raise ValueError("The master acquisition must use float64")
    if (
        sampling["j_order"],
        sampling["sampling_scheme"],
        sampling.get("multilevel_alpha", 1.0),
    ) != (8, "multilevel", 1.0):
        raise ValueError("Expected the locked J=8 multilevel alpha=1 protocol")

    image_path = Path(data["root"]) / row["source_relpath"]
    if _sha256(image_path) != row["source_sha256"]:
        raise ValueError(f"Source image hash mismatch: {image_path}")
    image64 = load_png_float64(image_path)
    if image64.shape != (1, 1, 256, 256) or image64.dtype != torch.float64:
        raise ValueError("Expected one float64 256x256 PCM image")

    mask_seed = int(row["mask_seed_uint64"])
    noise_seed = int(row["noise_seed_uint64"])
    indices = sample_indices(
        j_order=sampling["j_order"],
        sampling_ratio=sampling["sampling_ratio_requested"],
        coarse_j=sampling["coarse_j"],
        randomising_scheme=sampling["sampling_scheme"],
        rng=np.random.default_rng(mask_seed),
    )
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=noise_settings["current_to_model_scale"],
        j_order=sampling["j_order"],
        n_repeat_acquisitions=noise_settings["repeat_acquisitions"],
    )
    # Fail rather than silently following a later change to the named preset.
    actual_noise = asdict(noise.noise_params) | {
        "signal_to_bias_ratio": noise.signal_to_bias_ratio,
    }
    for key, value in actual_noise.items():
        if key in noise_settings and json.dumps(value) != json.dumps(
            noise_settings[key]
        ):
            raise ValueError(f"Noise preset differs from locked parameter: {key}")
    if noise_settings["preset"] != "PCMNoiseSimulation.npl_default":
        raise ValueError("This experiment uses the locked NPL default preset")
    acquisition = simulate_pcm_acquisition(
        image64,
        j_order=sampling["j_order"],
        sampled_indices=indices,
        noise=noise,
        rng=np.random.default_rng(noise_seed),
        device=device,
        seed=noise_seed,
        record_swaps=True,
    )
    if acquisition.measurement.dtype != torch.float64:
        raise AssertionError("PCM acquisition lost float64 precision")
    metadata = {
        "archive_schema": ARCHIVE_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "split": row["split"],
        "category": row["category"],
        "permutation_position_1based": int(row["permutation_position_1based"]),
        "sample_id": int(row["sample_id"]),
        "source_relpath": row["source_relpath"],
        "source_sha256": row["source_sha256"],
        "mask_seed_uint64": mask_seed,
        "noise_seed_uint64": noise_seed,
        "acquisition_dtype": "float64",
        "measurements": int(acquisition.measurement.numel()),
        "sampled_indices_dtype_in_archive": "uint16",
        "noise_preset": noise_settings["preset"],
        "repeat_acquisitions": noise_settings["repeat_acquisitions"],
        "j_order": sampling["j_order"],
        "sampling_ratio_requested": sampling["sampling_ratio_requested"],
        "coarse_j": sampling["coarse_j"],
        "multilevel_alpha": 1.0,
        "normalization_factor": acquisition.op.normalization_factor,
        "noise_parameters": json.loads(json.dumps(actual_noise)),
        "swap_bitorder": "little",
        "swap_flag_meaning": "0=keep (2k,2k+1); 1=swap (2k+1,2k), zero-based noise trace",
        "noise_realisation_id": 0,
        "rng_bit_generator": "PCG64",
        "oracle_policy": "Primary variances marginalise swaps; saved flags are for audit/replay.",
    }
    return acquisition, metadata


def _write_atomic_new(path: Path, writer) -> None:
    """Publish a complete file atomically without replacing any existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        # Same-filesystem hard-link publication is atomic and fails if the target exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_new(path: Path, value: dict) -> None:
    """Write one read-only JSON record; existing records are never replaced."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_atomic_new(path, lambda handle: handle.write(payload))


def _validate_arrays(y, indices, i0, packed, metadata):
    """Validate the versioned single-image archive and decode its audit flags."""
    if metadata.get("archive_schema") != ARCHIVE_SCHEMA:
        raise ValueError(
            "Not a Step 2 swap-aware master archive (old pilot is retained separately)"
        )
    m = int(metadata["measurements"])
    repeats = int(metadata["repeat_acquisitions"])
    if m <= 0 or repeats <= 0 or y.shape != (1, 1, m):
        raise ValueError("Invalid measurement shape or repeat count")
    if y.dtype != np.float64 or not np.isfinite(y).all():
        raise ValueError("Measurements must be finite float64")
    if i0.dtype != np.float64 or i0.shape != (1, 1) or not np.isfinite(i0).all():
        raise ValueError("I0 must be finite float64 with shape (1,1)")
    if indices.dtype != np.uint16 or indices.shape != (m,):
        raise ValueError("Stored external dyadic indices must be uint16 of length M")
    idx = indices.astype(np.int64)
    if len(np.unique(idx)) != m or np.any(idx[1:] <= idx[:-1]):
        raise ValueError(
            "Indices must be unique and in acquisition order (ascending dyadic rows)"
        )
    if metadata.get("j_order") != 8 or metadata.get("normalization_factor") != 1 / 256:
        raise ValueError("Unexpected Hadamard order or normalisation")
    coarse = 4 ** int(metadata["coarse_j"])
    if m < coarse or not np.array_equal(idx[:coarse], np.arange(coarse)):
        raise ValueError("Fully sampled coarse block is missing")
    if m != int(float(metadata["sampling_ratio_requested"]) * 65536):
        raise ValueError("Measurement count differs from the sampling fraction")
    if metadata.get("swap_bitorder") != "little":
        raise ValueError("Unknown swap bit order")
    if packed.dtype != np.uint8 or packed.shape != (repeats, (m + 7) // 8):
        raise ValueError("Packed swap flags have the wrong dtype or shape")
    bits = np.unpackbits(packed, axis=-1, bitorder="little")
    if np.any(bits[:, m:]):
        raise ValueError("Nonzero padding bits in packed swaps")
    return bits[:, :m].astype(bool).reshape(1, 1, repeats, m)


def save_master_npz(path: Path, acquisition: PCMAcquisition, metadata: dict) -> None:
    """Store float64 measurements and the actual swap flags, without overwriting."""
    y = acquisition.measurement.detach().cpu().numpy()
    indices = np.asarray(acquisition.sampled_indices)
    if not np.issubdtype(indices.dtype, np.integer) or np.any(
        (indices < 0) | (indices >= 65536)
    ):
        raise ValueError("External dyadic indices cannot be represented as uint16")
    if acquisition.i0 is None or acquisition.swap_flags is None:
        raise ValueError("The noisy master requires I0 and recorded swap flags")
    i0 = acquisition.i0.detach().cpu().numpy()
    flags = acquisition.swap_flags.detach().cpu().numpy()
    expected_shape = (
        1,
        1,
        int(metadata["repeat_acquisitions"]),
        int(metadata["measurements"]),
    )
    if flags.dtype != np.bool_ or flags.shape != expected_shape:
        raise ValueError("Expected boolean swap flags for every repeat and measurement")
    packed = np.packbits(flags.reshape(expected_shape[-2:]), axis=-1, bitorder="little")
    indices = indices.astype(np.uint16)
    _validate_arrays(y, indices, i0, packed, metadata)
    _write_atomic_new(
        path,
        lambda handle: np.savez_compressed(
            handle,
            measurement=y,
            sampled_indices=indices,
            i0=i0,
            swap_flags_packed=packed,
            metadata_json=json.dumps(metadata, sort_keys=True),
        ),
    )


def load_master_npz(
    path: Path,
    *,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[PCMAcquisition, dict]:
    """Load the identical stored A and y; never regenerate noise or recast the master."""
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError(f"Acquisition checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        y, indices, i0, packed = (
            data[k]
            for k in (
                "measurement",
                "sampled_indices",
                "i0",
                "swap_flags_packed",
            )
        )
    flags = _validate_arrays(y, indices, i0, packed, metadata)
    acquisition = acquisition_from_prepared_measurement(
        torch.from_numpy(y),
        j_order=8,
        sampled_indices=indices.astype(np.int64),
        device=device,
        seed=int(metadata["noise_seed_uint64"]),
    )
    return (
        replace(
            acquisition,
            i0=torch.from_numpy(i0).to(device),
            swap_flags=torch.from_numpy(flags),
        ),
        metadata,
    )


def solver_measurement(
    acquisition: PCMAcquisition, *, dtype: torch.dtype
) -> torch.Tensor:
    """Cast only the completed differential measurement at the solver boundary."""
    # A float64 solver also receives an independent copy, not the master tensor.
    return acquisition.measurement.to(dtype=dtype).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--category", required=True)
    parser.add_argument("--permutation-position", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if _sha256(args.manifest) != protocol["data"]["manifest_sha256"]:
        raise ValueError("Step 1 image and seed manifest was modified")
    selected = [
        row
        for row in rows
        if row["split"] == args.split
        and row["category"] == args.category
        and int(row["permutation_position_1based"]) == args.permutation_position
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one pinned image/seed manifest row")
    acquisition, metadata = create_float64_acquisition(
        row=selected[0], protocol=protocol, device=args.device
    )
    save_master_npz(args.output, acquisition, metadata)
    print(f"Saved {args.output}: {acquisition.measurement.numel()} float64 values")


if __name__ == "__main__":
    main()
