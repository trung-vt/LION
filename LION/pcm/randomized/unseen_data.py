"""Prepare two NPL-supplied unseen PCM map exports, without modifying their data.

The .map files are full 768x1024 canvases containing compact active rectangles.
This is an out-of-distribution descriptive benchmark, not a training loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from LION.pcm.randomized.figure_style import save_pcm_image

SOURCES = {
    "cigs": "CIGS_R2_C1_CTF_256x256_50mA__V0.000.txt.map",
    "perovskite": "perovskiteCell1_256x256_50mA__V0.900.txt.map",
}
# Inferred from the order and visual appearances of screenshots. No original
# screenshot metadata identifies a source, so "_not_sure" is intentional.
PREVIEW_NAMES = {
    "image.png": "CIGS_R2_C1_full_canvas_preview_not_sure.png",
    "image copy.png": "CIGS_R2_C1_cropped_preview_not_sure.png",
    "image copy 2.png": "perovskiteCell1_full_canvas_preview_not_sure.png",
    "image copy 3.png": "perovskiteCell1_cropped_preview_not_sure.png",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(folder: Path) -> dict:
    folder = folder.resolve()
    archive = folder / "Unseen_current_mapping_datasets_for_paper.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    raw = folder / "raw_maps"
    raw.mkdir(exist_ok=True)
    with ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            if member.filename not in SOURCES.values():
                raise ValueError("Unexpected/unsafe archive entry: " + member.filename)
            dest = raw / member.filename
            if not dest.exists():
                dest.write_bytes(z.read(member))
            elif dest.read_bytes() != z.read(member):
                raise RuntimeError(f"Existing extracted map differs from ZIP: {dest}")

    for previous, target in PREVIEW_NAMES.items():
        src, dst = folder / previous, folder / target
        if src.exists() and dst.exists():
            raise RuntimeError(f"Both original and renamed previews exist: {src} {dst}")
        if src.exists():
            src.rename(dst)
        if not dst.exists():
            raise FileNotFoundError(f"Expected preview: {dst}")

    derived = folder / "derived"
    derived.mkdir(exist_ok=True)
    record = {
        "status": "unseen: not used for training or hyperparameter selection",
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "previews": PREVIEW_NAMES,
        "sign_convention": "display positive magnitude max(-raw_current, 0)",
        "normalization": "divide each full active rectangle by its positive-current-magnitude 99.5th percentile; clip display/reconstruction tensors to [0,1]; all 256x256 crops of the same cell share this scale",
        "calibration": "unit-free contrast normalization for reconstruction comparisons, not an ampere-calibrated forward model",
        "cells": {},
        "cases": [],
    }
    for name, file in SOURCES.items():
        path = raw / file
        full = np.loadtxt(path)
        if full.shape != (768, 1024) or not np.isfinite(full).all():
            raise ValueError(f"Unexpected or non-finite map {path}: {full.shape}")
        nonzero = full != 0
        rr, cc = np.nonzero(nonzero)
        y0, y1, x0, x1 = (
            int(rr.min()),
            int(rr.max()) + 1,
            int(cc.min()),
            int(cc.max()) + 1,
        )
        roi = full[y0:y1, x0:x1]
        if roi.shape not in ((256, 256), (512, 512)):
            raise ValueError(f"Unexpected active box {roi.shape} {name}")
        magnitude = np.maximum(-roi, 0)
        positives = magnitude[magnitude > 0]
        if len(positives) == 0:
            raise ValueError(f"No negative current magnitude {name}")
        scale = float(np.percentile(positives, 99.5))
        normalized = np.clip(magnitude / scale, 0, 1).astype("float32")
        entry = {
            "file": str(path),
            "sha256": digest(path),
            "canvas_shape": list(full.shape),
            "active_bounds_yx_exclusive": [y0, y1, x0, x1],
            "active_shape": list(roi.shape),
            "active_nonzero_fraction": float(np.mean(roi != 0)),
            "active_negative_fraction": float(np.mean(roi < 0)),
            "active_positive_fraction": float(np.mean(roi > 0)),
            "signed_current_min": float(np.min(roi)),
            "signed_current_max": float(np.max(roi)),
            "normalization_p995_negative_current": scale,
            "clipped_fraction": float(np.mean(magnitude / scale > 1)),
        }
        record["cells"][name] = entry
        slices = [("active_full", normalized, [0, len(roi), 0, len(roi)])]
        if normalized.shape[0] == 512:
            slices.append(
                ("center_256", normalized[128:384, 128:384], [128, 384, 128, 384])
            )
            for label, y, x in (
                ("top_left_256", 0, 0),
                ("top_right_256", 0, 256),
                ("bottom_left_256", 256, 0),
                ("bottom_right_256", 256, 256),
            ):
                slices.append(
                    (
                        label,
                        normalized[y : y + 256, x : x + 256],
                        [y, y + 256, x, x + 256],
                    )
                )
        for label, values, box in slices:
            target = derived / f"{name}_{label}.npy"
            # Never silently change an earlier exported image.
            if target.exists():
                if not np.array_equal(np.load(target), values):
                    raise FileExistsError(f"Derived crop differs: {target}")
            else:
                np.save(target, values, allow_pickle=False)
            png = target.with_suffix(".png")
            if not png.exists():
                save_pcm_image(png, values, kind="reconstruction")
            record["cases"].append(
                {
                    "cell": name,
                    "label": label,
                    "path": str(target),
                    "sha256": digest(target),
                    "shape": list(values.shape),
                    "crop_yx_on_active_exclusive": box,
                    "reference": label == "active_full",
                }
            )
    manifest = derived / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != record:
            raise FileExistsError(
                f"Manifest changed; inspect before overwriting {manifest}"
            )
    else:
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    r = prepare(args.folder)
    print(json.dumps({"cells": r["cells"], "cases": r["cases"]}, indent=2))


if __name__ == "__main__":
    main()
