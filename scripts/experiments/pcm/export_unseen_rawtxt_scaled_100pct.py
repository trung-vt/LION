"""Export the newly corrected raw-text-map PCM reconstructions in NPL format.

Saved reconstructions already have physical current units, because the PCM
operator is normalized by r_high during inference and rescaled afterward.
The original NPL current sign is recovered using data_sign = -1. No new
simulation, reconstruction, fitting, or inference is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

import numpy as np
import yaml

ROOT = Path("/mnt/store_ext4/store/LION/ttv22/pcm")
RESULTS = ROOT / "results/20261021_npl_rawtxt_scaled_1p8e9_4p0e9_v1"
SOURCES = ROOT / "data/unseen_data/derived/manifest.json"
DEST = RESULTS / "exports_for_npl_v2"
METHODS = (
    "zero_filled",
    "spgl1",
    "supervised_utv",
    "self_supervised_utv",
    "scalar_tv",
    "pnp_admm_drunet",
    "pnp_admm_dwtdnet_tiny",
    "pnp_gs_deepinv_drunet",
    "pnp_gs_deepinv_dwtdnet_tiny",
)
CONFIG = {"cigs": (1.8e-9, 6), "perovskite": (4e-9, 7)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_map(method, sample, case_dir, coarse):
    sample_dir = case_dir / f"artifacts/trial_001/s_1.00_cJ_{coarse}"
    if method == "zero_filled":
        paths = list((sample_dir / "zero_filled").glob("*.npy"))
        assert len(paths) == 1, paths
        path = paths[0]
    else:
        path = sample_dir / method / "recons/recon.npy"
    arr = np.load(path, allow_pickle=False).squeeze()
    assert arr.ndim == 2 and np.isfinite(arr).all(), path
    return arr, path


def save_map(path, active, shape, bounds):
    y0, y1, x0, x1 = bounds
    assert active.shape == (y1 - y0, x1 - x0), (path, active.shape, bounds)
    canvas = np.zeros(shape, dtype=np.float64)
    canvas[y0:y1, x0:x1] = active.astype(np.float64)
    np.savetxt(path, canvas, fmt="%.8E", delimiter="\t")
    check = np.loadtxt(path, dtype=np.float64)
    assert check.shape == shape
    assert np.allclose(check[y0:y1, x0:x1], active, rtol=3e-8, atol=2e-16)
    assert np.count_nonzero(check) == np.count_nonzero(check[y0:y1, x0:x1])
    return path


def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to replace prior NPL exports: {DEST}")
    stage = DEST.with_name(DEST.name + "_staging")
    if stage.exists():
        raise FileExistsError(f"Existing staging directory: {stage}")
    records = []
    meta = json.loads(SOURCES.read_text())
    stage.mkdir(parents=True)
    specimen_archives = []
    for sample, (r_high, coarse) in CONFIG.items():
        data = meta["cells"][sample]
        original = Path(data["file"])
        assert original.is_file() and sha256(original) == data["sha256"]
        full = np.loadtxt(original)
        shape = tuple(data["canvas_shape"])
        bounds = data["active_bounds_yx_exclusive"]
        y0, y1, x0, x1 = bounds
        assert full.shape == shape == (768, 1024)
        prefix = original.name.removesuffix(".txt.map")
        assert prefix and prefix + ".txt.map" == original.name
        case_dir = (
            RESULTS
            / sample
            / "s_1p00"
            / f"{sample}_s1p00_ordered_no_added_noise_rawtxt_scaled"
        )
        assert (RESULTS / sample / "s_1p00" / "COMPLETED").is_file()
        c = yaml.safe_load((case_dir / "config.yaml").read_text())
        scaling = c["scaling"]
        assert scaling["r_high"] == r_high and scaling["r_low"] == 0.0
        assert scaling["data_sign"] == -1.0 and scaling["divide_by"] == 1.0
        assert c["data"]["data_type"] == "image_txt_map"
        assert c["noise"] is None
        assert c["plot"]["color"]["clim"] == [0.0, r_high] or tuple(
            c["plot"]["color"]["clim"]
        ) == (0.0, r_high)
        signed_ref = -full[y0:y1, x0:x1]
        assert signed_ref.shape == (1 << (coarse + 2),) * 2
        folder = stage / sample
        folder.mkdir()
        for method in METHODS:
            active, source = numeric_map(method, sample, case_dir, coarse)
            assert active.shape == signed_ref.shape
            # Recon arrays are already rescaled to sign-corrected raw units.
            # Changing the sign once reproduces NPL's original signed convention.
            for variant, values in (
                ("signed_source_current_units", -active),
                (
                    "dimensionless_normalized_model_units",
                    np.asarray(active, dtype=np.float64) / r_high,
                ),
            ):
                subdir = folder / variant
                subdir.mkdir(exist_ok=True)
                target = (
                    subdir
                    / f"{prefix}_100pct_ordered_no_extra_noise_{method}_{variant}.txt.map"
                )
                save_map(target, values, shape, bounds)
                rec = {
                    "sample": sample,
                    "method": method,
                    "variant": variant,
                    "r_high": r_high,
                    "source_prediction": str(source),
                    "source_prediction_sha256": sha256(source),
                    "original_map": str(original),
                    "original_map_sha256": data["sha256"],
                    "shape": list(shape),
                    "active_bounds_yx_exclusive": bounds,
                    "conversion": "-reconstruction_physical_current"
                    if variant == "signed_source_current_units"
                    else "reconstruction_physical_current / r_high",
                    "output_filename": str(target.relative_to(stage)),
                    "sha256": sha256(target),
                    "size_bytes": target.stat().st_size,
                    "range": [float(np.min(values)), float(np.max(values))],
                }
                records.append(rec)
                print("EXPORTED", sample, method, variant, flush=True)
        readme = folder / "README_FOR_NPL.txt"
        readme.write_text(
            f"NPL map: {original.name}\n"
            f"Active bounds (y0,y1,x0,x1)={bounds}\n"
            f"Hadamard simulation: 100% ordered, no added synthetic noise.\n"
            f"Preset r_high={r_high:.13E}, r_low=0, divide_by=1, data_sign=-1.\n\n"
            "The files retain the original 768x1024 tab-separated scientific\n"
            "notation canvas and the original row/column order, with no flips.\n"
            "The signed_source_current_units folder contains -1 times the\n"
            "PHYSICAL-SCALE model reconstructions (matching original NPL current\n"
            "sign), not a newly acquired or independently calibrated current map.\n"
            "The dimensionless_normalized_model_units folder divides model\n"
            "reconstructions by the preset r_high. No prediction was clipped.\n"
            "The input full inversion already contains measurement noise.\n"
            "No actual compressed NPL acquisition was used or acquired.\n"
            "NPL's GUI shows y increasing upward. Earlier research figures\n"
            "used imshow origin=upper, but the saved arrays were never flipped.\n"
            "See ../manifest.json for the original and reconstruction checksums.\n"
        )
        archive = stage / f"{prefix}_100pct_rawtxt_scaled_reconstructions.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    z.write(file, arcname=file.relative_to(folder))
        with ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist()) == 2 * len(METHODS) + 1
        specimen_archives.append(archive)
        print("SPECIMEN_ZIP", archive.name, archive.stat().st_size, flush=True)
    manifest = {
        "r_high": {k: v[0] for k, v in CONFIG.items()},
        "original_signed_current_units": True,
        "original_array_orientation_preserved": True,
        "input_data_type": "image_txt_map",
        "source_is_measured_full_inverse_not_true_ground_truth": True,
        "acquisition_is_simulated_100pct_no_extra_noise": True,
        "no_posthoc_current_scaling_applied_to_inference": True,
        "files": records,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = stage / "NPL_corrected_rawtxt_scaled_100pct_CIGS_perovskite.zip"
    with ZipFile(archive, "w", compression=ZIP_STORED) as z:
        for path in specimen_archives:
            z.write(path, arcname=path.name)
        z.write(stage / "manifest.json", arcname="manifest.json")
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == 3
    assert len(records) == 2 * len(METHODS) * 2
    stage.rename(DEST)
    print("COMPLETE", len(records), "NPL-compatible maps at", DEST)


if __name__ == "__main__":
    main()
