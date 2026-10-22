"""Export saved 100% PCM reconstructions as NPL-style .txt.map files.

LEGACY/UNSCALED: exports old clipped normalized-image results.
Do not send these files to NPL. Use export_unseen_rawtxt_scaled_100pct.py
and its verified exports_for_npl_v2 output instead.

The original NPL arrays determine the canvas, active bounds and row order.
No training, new reconstruction, inferred physical calibration, or flip is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

DATA = Path("/mnt/store_ext4/store/LION/ttv22/pcm/data/unseen_data")
RUNS = Path(
    "/mnt/store_ext4/store/LION/ttv22/pcm/results/20261021_npl_100pct_noiseless_ordered_quick"
)
DEFAULT_OUT = RUNS / "exports_for_npl"
METHODS = (
    "pnp_admm_drunet",
    "pnp_admm_dwtdnet_tiny",
    "pnp_gs_deepinv_drunet",
    "pnp_gs_deepinv_dwtdnet_tiny",
    "supervised_utv",
    "self_supervised_utv",
    "spgl1",
    "scalar_tv",
    "zero_filled",
)
COARSE_J = {"cigs": 6, "perovskite": 7}


def checksum(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_prediction(specimen, method):
    """Read the already-completed 100% reconstruction from its saved array."""
    sample_run = RUNS / specimen / f"{specimen}_100pct_noiseless_ordered_matched"
    assert (RUNS / specimen / "COMPLETED").is_file(), sample_run
    case = sample_run / f"artifacts/trial_001/s_1.00_cJ_{COARSE_J[specimen]}"
    if method == "zero_filled":
        found = list((case / "zero_filled").glob("*.npy"))
        assert len(found) == 1, found
        source = found[0]
    else:
        source = case / method / "recons/recon.npy"
    assert source.is_file(), source
    arr = np.load(source, allow_pickle=False).squeeze()
    assert arr.ndim == 2 and np.isfinite(arr).all(), source
    return arr, source


def save_map(file, active, bounds, canvas_shape, factor):
    """Write a full-size tab-delimited canvas without changing row orientation."""
    y0, y1, x0, x1 = bounds
    assert active.shape == (y1 - y0, x1 - x0)
    canvas = np.zeros(canvas_shape, dtype=np.float64)
    canvas[y0:y1, x0:x1] = np.asarray(active, dtype=np.float64) * factor
    # Same scientific-notation text convention as original NPL .txt.map files.
    np.savetxt(file, canvas, fmt="%.8E", delimiter="\t")
    checked = np.loadtxt(file, dtype=np.float64)
    assert checked.shape == canvas_shape
    assert not np.any(checked[:y0]) and not np.any(checked[y1:])
    assert not np.any(checked[y0:y1, :x0]) and not np.any(checked[y0:y1, x1:])
    # 9 significant digits in the NPL-style text representation.
    assert np.allclose(
        checked[y0:y1, x0:x1],
        canvas[y0:y1, x0:x1],
        rtol=3e-8,
        atol=1e-16 if abs(factor) < 1e-4 else 1e-9,
    )
    return {
        "path": str(file),
        "sha256": checksum(file),
        "size_bytes": file.stat().st_size,
        "min": float(canvas.min()),
        "max": float(canvas.max()),
        "full_shape": list(canvas_shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output
    assert not output.exists(), f"Refusing to overwrite: {output}"
    staging = output.with_name(output.name + "_staging")
    assert not staging.exists(), f"Existing staging directory: {staging}"
    manifest = json.loads((DATA / "derived/manifest.json").read_text())
    staging.mkdir(parents=True)
    export_records = []
    for specimen in ("cigs", "perovskite"):
        sample_info = manifest["cells"][specimen]
        source = Path(sample_info["file"])
        assert source.is_file() and checksum(source) == sample_info["sha256"]
        original = np.loadtxt(source, dtype=np.float64)
        shape = tuple(sample_info["canvas_shape"])
        bounds = sample_info["active_bounds_yx_exclusive"]
        scale = float(sample_info["normalization_p995_negative_current"])
        y0, y1, x0, x1 = bounds
        assert shape == (768, 1024) and original.shape == shape
        normalized = np.load(
            DATA / "derived" / f"{specimen}_active_full.npy", allow_pickle=False
        )
        expected = np.clip(np.maximum(-original[y0:y1, x0:x1], 0.0) / scale, 0.0, 1.0)
        assert np.max(np.abs(normalized - expected)) < 5e-8
        # The preprocessing has no vertical flip. A flip here would corrupt
        # the original NPL .txt.map row positions.
        assert np.mean(np.abs(normalized[::-1] - expected)) > 0.02
        assert normalized.shape == (y1 - y0, x1 - x0)
        prefix = source.name.removesuffix(".txt.map")
        assert prefix and source.name == prefix + ".txt.map"
        folder = staging / specimen
        folder.mkdir()
        raw_copy = folder / "original_reference_unchanged" / source.name
        raw_copy.parent.mkdir()
        shutil.copy2(source, raw_copy)
        assert checksum(raw_copy) == sample_info["sha256"]
        local_records = []
        for method in METHODS:
            recon, recon_source = get_prediction(specimen, method)
            assert recon.shape == normalized.shape
            # Predictions, including negative and >1 values, are NOT clipped.
            for unit_tag, factor in (
                ("normalized_model_units", 1.0),
                ("approx_original_signed_numeric_units", -scale),
            ):
                dest_dir = folder / unit_tag
                dest_dir.mkdir(exist_ok=True)
                exported = dest_dir / (
                    f"{prefix}_100pct_ordered_no_extra_noise_"
                    f"{method}_{unit_tag}.txt.map"
                )
                info = save_map(exported, recon, bounds, shape, factor)
                record = {
                    "specimen": specimen,
                    "method": method,
                    "representation": unit_tag,
                    "original_map_prefix": prefix,
                    "source_prediction": str(recon_source),
                    "source_prediction_sha256": checksum(recon_source),
                    "source_map": str(source),
                    "source_map_sha256": sample_info["sha256"],
                    "active_bounds_yx_exclusive": bounds,
                    "canvas_shape": list(shape),
                    "normalization_p995_source_numeric_units": scale,
                    "rescaling_factor": factor,
                    "no_flip_no_rotation": True,
                    "additional_noise": "none",
                    "sampling_fraction": 1.0,
                    "sampling_order": "first-in-order Hadamard, simulated",
                    "output_relative_to_export": str(exported.relative_to(staging)),
                    **info,
                }
                local_records.append(record)
                export_records.append(record)
                print(specimen, method, unit_tag, "OK", flush=True)
        readme = folder / "README_FOR_NPL.txt"
        readme.write_text(
            "100% ordered PCM reconstructions from the two previously supplied NPL maps\n"
            "==========================================================================\n\n"
            f"Specimen: {specimen}\n"
            f"Source map: {source.name}\n"
            f"Original canvas: {shape[0]} rows x {shape[1]} columns\n"
            f"Active bounds [y0,y1,x0,x1), zero-based: {bounds}\n\n"
            "Every exported .txt.map has the ORIGINAL NPL 768x1024 canvas shape,\n"
            "scientific-notation values separated by tabs, and the identical\n"
            "original row/column order. There is NO flip, rotation, or transpose.\n"
            "Values outside the active rectangle are zero. NPL's preview GUI\n"
            "displays y increasing UPWARD; our early Matplotlib figures used\n"
            "origin=upper and therefore looked upside down. Do not flip these\n"
            "exported numerical files to match those early figures.\n\n"
            "normalized_model_units/: model predictions, dimensionless, not\n"
            "clipped or renormalized for export. Not measured currents.\n\n"
            "approx_original_signed_numeric_units/: the SAME model predictions\n"
            f"multiplied by -{scale:.13E}, the original specimen's 99.5th\n"
            "percentile of negative-current magnitude. Values resemble the\n"
            "original signed current scale but ARE NOT CALIBRATED measurements.\n"
            "Source preprocessing discarded positive currents and clipped\n"
            "values above its percentile before model inference, so multiplying\n"
            "back cannot recover the original physical values exactly.\n\n"
            "original_reference_unchanged/: an exact copy of NPL's original\n"
            ".txt.map, included solely to check row order and canvas layout.\n\n"
            "These reconstructions use simulated 100% Hadamard measurements\n"
            "of the previously measured, normalized full inverse with NO added\n"
            "synthetic noise. They are not new physical NPL acquisitions.\n\n"
            "See manifest.json for original source paths and SHA256 hashes.\n"
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "specimen": specimen,
                    "source_map_sha256": sample_info["sha256"],
                    "source_map": str(source),
                    "original_map_reference_copy": str(raw_copy.relative_to(folder)),
                    "active_bounds_yx_exclusive": bounds,
                    "normalization_p995_original_numeric_units": scale,
                    "files": local_records,
                },
                indent=2,
            )
            + "\n"
        )
        archive = staging / f"{prefix}_100pct_ordered_no_added_noise_exports.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    z.write(file, arcname=file.relative_to(folder))
        with ZipFile(archive) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == 2 * len(METHODS) + 3
        print("ARCHIVE", archive.name, archive.stat().st_size, "bytes", flush=True)
    assert len(export_records) == 2 * len(METHODS) * 2
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "NPL-style text maps of SAVED 100% PCM reconstructions",
                "warning": "Not newly measured currents, not physical calibration",
                "format": "768x1024 tab-separated %.8E scientific notation, no header",
                "original_NPL_row_order_retained": True,
                "NPL_preview_y_axis_increases_upward": True,
                "rescaling_is_approximate_not_calibrated": True,
                "source_preprocessing": "positive current dropped; bright magnitudes clipped at source p99.5",
                "export_files": export_records,
            },
            indent=2,
        )
        + "\n"
    )
    staging.rename(output)
    print("EXPORTED", len(export_records), "NPL-style text maps to", output)


if __name__ == "__main__":
    main()
