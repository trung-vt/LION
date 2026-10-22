"""Refresh figures from saved PCM arrays only; never rerun acquisition or inference.

Creates a ZIP containing the original PNG renders for rollback, then uses
existing LION plotting/colormap helpers to update all previews, figures and
absolute-error grids. Numeric results, checkpoints, source maps and frozen
measurements are checksum-verified and left unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from LION.pcm.randomized.archive_unseen_reconstructions import (
    _montage,
    METHODS,
)
from LION.pcm.randomized.data import sha256, write_json
from LION.pcm.randomized.figure_style import image_style, save_pcm_image
from LION.pcm.randomized.monitoring import image_grid
from LION.pcm.randomized.unseen_analysis import generate


def _replace_pcm_png(path: Path, values: np.ndarray, kind: str) -> None:
    with TemporaryDirectory(prefix="pcm-display-", dir=path.parent) as folder:
        fresh = Path(folder) / path.name
        save_pcm_image(fresh, values, kind=kind)
        os.replace(fresh, path)


def _legacy_render_backup(root: Path, case_archive: Path, dataset: Path) -> Path:
    target = root / "visuals_before_afmhot_viridis_20261014.zip"
    if target.exists():
        return target
    entries = []
    for category in ("images", "abs_errors", "montages"):
        entries.extend((case_archive / category).rglob("*.png"))
    entries += list((root / "reconstructions").glob("*.png"))
    entries += list((root / "analysis" / "figures").glob("*.png"))
    entries += list((dataset.parent / "derived").glob("*.png"))
    if len(entries) < 500:
        raise RuntimeError(f"Unexpectedly small visual backup: {len(entries)} PNGs")
    target_tmp = target.with_suffix(".zip.tmp")
    with ZipFile(target_tmp, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(entries):
            # Explicit filesystem ancestors; no files outside the known output
            # and generated NPL data-preview directories.
            if path.is_relative_to(root):
                member = "experiment/" + str(path.relative_to(root))
            else:
                member = "data_previews/" + path.name
            z.write(path, member)
    os.replace(target_tmp, target)
    return target


def regenerate(root: Path, unseen_frozen: Path, *, data_manifest: Path) -> dict:
    root = root.resolve()
    case_archive = root / "complete_image_archive"
    protocol = json.loads((case_archive / "protocol.json").read_text())
    if protocol["status"] != "completed" or protocol["reconstructions"] != 216:
        raise ValueError("A complete saved 216-reconstruction archive is required")
    frozen = json.loads(unseen_frozen.read_text())
    if frozen["split"] != "unseen" or frozen["number_of_acquisitions"] != 24:
        raise ValueError("The saved frozen unseen acquisitions are incomplete")
    metrics_path = case_archive / "all_reconstructions.csv"
    original_metrics_path = root / "reconstruction_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 216:
        raise ValueError(f"Expected 216 saved numeric metrics: {len(metrics)}")
    unique = {r["case_file"] for r in frozen["acquisitions"]}
    if len(unique) != 24:
        raise ValueError("Source acquisitions are duplicated")
    # Check authoritative numeric data and identifiers before any image edits.
    immutable = [unseen_frozen, metrics_path, original_metrics_path]
    for r in frozen["acquisitions"]:
        immutable.append(unseen_frozen.parent / r["case_file"])
    original_hashes = {str(p): sha256(p) for p in immutable}
    for r in frozen["acquisitions"]:
        if sha256(unseen_frozen.parent / r["case_file"]) != r["case_file_sha256"]:
            raise RuntimeError("Frozen acquisition checksum mismatch")
    for row in metrics.itertuples():
        name = Path(row.case_file).stem
        path = case_archive / "arrays" / name / (row.method + ".npy")
        if sha256(path) != row.array_sha256:
            raise RuntimeError(f"Numeric reconstruction checksum mismatch: {path}")
    # Preserve pre-change PNG renderings for possible visual rollback.
    backup = _legacy_render_backup(root, case_archive, data_manifest)
    image_count = error_count = montage_count = 0
    sorted_rows = metrics.groupby("case_file", sort=False)
    saved_labels = {r["case_file"]: r for r in frozen["acquisitions"]}
    for case_file, group in sorted_rows:
        meta = saved_labels[case_file]
        name = Path(case_file).stem
        with np.load(unseen_frozen.parent / case_file, allow_pickle=False) as f:
            reference = f["reference"].copy()
        predictors = {"reference": reference}
        _replace_pcm_png(
            case_archive / "images" / name / "reference.png",
            reference,
            "reconstruction",
        )
        image_count += 1
        rows = group.to_dict("records")
        if set(r["method"] for r in rows) != set(METHODS):
            raise ValueError(f"Incomplete reconstruction grid for {name}")
        for item in rows:
            m = item["method"]
            values = np.load(
                case_archive / "arrays" / name / (m + ".npy"), allow_pickle=False
            )
            predictors[m] = values
            _replace_pcm_png(
                case_archive / "images" / name / (m + ".png"),
                values,
                "reconstruction",
            )
            image_count += 1
            _replace_pcm_png(
                case_archive / "abs_errors" / name / (m + ".png"),
                np.abs(np.asarray(values) - np.asarray(reference)),
                "absolute_error",
            )
            error_count += 1
        with TemporaryDirectory(
            prefix="pcm-grids-", dir=case_archive / "montages"
        ) as temp:
            for error in (False, True):
                _montage(Path(temp), name, predictors, rows, error=error)
                suffix = "_abs_error" if error else "_recon"
                os.replace(
                    Path(temp) / (name + suffix + ".png"),
                    case_archive / "montages" / (name + suffix + ".png"),
                )
                montage_count += 1
        if np.isclose(meta["sampling_ratio"], 0.20) and meta["scenario"] in (
            "noiseless",
            "pcm_default_10ms",
            "pcm_short_100us",
        ):
            domain = (
                "cigs_active_full" if meta["case"] == 0 else "perovskite_center_256"
            )
            title = f"{domain}_{meta['scenario']}_r020"
            panels = [("Reference", reference)]
            for method in METHODS:
                match = next(row for row in rows if row["method"] == method)
                panels.append(
                    (f"{method} ({float(match['psnr']):.1f} dB)", predictors[method])
                )
            target = root / "reconstructions" / (title + ".png")
            with TemporaryDirectory(prefix="pcm-overview-", dir=target.parent) as temp:
                tmp = Path(temp) / target.name
                image_grid(tmp, [(domain, panels)])
                os.replace(tmp, target)
        print(f"Re-rendered display-only case {name}", flush=True)

    # The derived source images are presentation aids, not screenshot originals.
    doc = json.loads(data_manifest.read_text())
    for item in doc["cases"]:
        npy = Path(item["path"])
        preview = npy.with_suffix(".png")
        _replace_pcm_png(preview, np.load(npy, allow_pickle=False), "reconstruction")

    # This function re-plots recorded CSV metrics and the source-pair PNG,
    # without calling any model, CUDA operator or synthetic acquisition.
    plots = generate(root)
    for path, saved_hash in original_hashes.items():
        if sha256(Path(path)) != saved_hash:
            raise RuntimeError(
                "Immutable numeric source was unexpectedly modified: " + path
            )
    if (image_count, error_count, montage_count) != (240, 216, 48):
        raise AssertionError((image_count, error_count, montage_count))
    recon_style = image_style("reconstruction")
    error_style = image_style("absolute_error")
    report = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Use existing PCM AFM reconstruction/viridis error display instead of duplicate grayscale/magma rendering",
        "numeric_data_untouched": True,
        "no_inference_or_training_or_new_measurements": True,
        "original_visuals_zip": str(backup),
        "source_frozen_manifest_sha256": sha256(unseen_frozen),
        "numeric_source_sha256": original_hashes,
        "source_array_sha256_preserved": True,
        "images_recolored": image_count,
        "error_images_recolored": error_count,
        "montages_recolored": montage_count,
        "reconstruction_cmap": "afmhot",
        "reconstruction_range": [recon_style[1], recon_style[2]],
        "absolute_error_cmap": "viridis",
        "absolute_error_range": [error_style[1], error_style[2]],
        "parameter_map_cmap": "magma (unchanged in existing LION PCM config)",
        "source_plot_metrics": plots,
    }
    write_json(root / "display_revision.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate(
                args.root, args.frozen_manifest, data_manifest=args.source_manifest
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
