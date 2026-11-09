"""Produce category PDF galleries containing every unique test45 reconstruction.

Each image is shown against the same ground truth and original noisy PCM
acquisition. Exactly 21 panels per image: target, ten historical baselines,
five independent ADMM tune-1 configurations, and five GS tune-1 configurations.
One category => 9 pages. Merge 5 category PDFs to obtain all 45 pages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    BASE,
    CATEGORIES,
)
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen

PREVIOUS = BASE / "ssdu_pnp_tuning_20261108_v1"
OLD_RAW = {
    "gs_pnp_oracle": PREVIOUS / "test450/v1/raw/gs_oracle_step6",
    "gs_pnp_ssdu5": PREVIOUS / "test450/v1/raw/gs_ssdu",
    "gs_pnp_expanded5": PREVIOUS / "next_gs_ssdu_sweep_v1/test450/raw",
}
OLD_NAMES = (
    ("zero_filled", "Zero-filled"),
    ("spgl1_bp", "SPGL1-BP"),
    ("spgl1_bpdn", "SPGL1-BPDN"),
    ("pnp_admm_oracle", "PnP-ADMM oracle"),
    ("gs_pnp_oracle", "GS-PnP oracle"),
    ("tv", "Scalar TV"),
    ("utv_c60", "UTV-C60"),
    ("pnp_admm_ssdu5", "PnP-ADMM SSDU-5"),
    ("gs_pnp_ssdu5", "GS-PnP SSDU-5"),
    ("gs_pnp_expanded5", "GS-PnP expanded SSDU-5"),
)


def png_target(category, sid):
    image = (
        BASE
        / "PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
        / category
        / f"image{sid:04d}.png"
    )
    assert image.is_file(), image
    return (
        np.asarray(Image.open(image).convert("L"), dtype=np.uint8).astype(np.float32)
        / 255.0
    )


def old_array(name, category, sid):
    if name in OLD_RAW:
        path = OLD_RAW[name] / f"{category}_{sid:04d}.npy"
    elif name == "pnp_admm_ssdu5":
        path = (
            OUTPUT
            / "test45/baselines/raw/pnp_admm_oracle"
            / f"{category}_{sid:04d}.npy"
        )
    else:
        path = OUTPUT / "test45/baselines/raw" / name / f"{category}_{sid:04d}.npy"
    assert path.is_file(), path
    result = np.load(path, allow_pickle=False)
    assert result.shape == (256, 256) and np.isfinite(result).all()
    return result


def main(category):
    assert category in CATEGORIES
    calibs, tests, configs = assert_all_frozen()
    subset = sorted(
        [x for x in tests if x["category"] == category], key=lambda x: x["sample_id"]
    )
    cids = sorted(int(x["sample_id"]) for x in calibs if x["category"] == category)
    assert len(subset) == 9 and len(cids) == 5
    scores = json.loads((OUTPUT / "analysis/test45_metrics.json").read_text())
    cmap = make_colormap("afmhot", 0.8)
    out = OUTPUT / "analysis/galleries" / f"{category}_9_pages.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42}
    )
    with PdfPages(out) as pdf:
        for j, case in enumerate(subset, 1):
            sid = int(case["sample_id"])
            same = scores[f"{category}_{sid:04d}"]["metrics"]
            panels = [("target", "Target", png_target(category, sid), None)]
            for name, label in OLD_NAMES:
                panels.append((name, label, old_array(name, category, sid), same[name]))
            for method, short in (("admm", "ADMM"), ("gs", "GS")):
                for cid in cids:
                    rawpath = (
                        OUTPUT
                        / "test45"
                        / method
                        / category
                        / "raw"
                        / f"{cid:04d}"
                        / f"{sid:04d}.npy"
                    )
                    recpath = (
                        OUTPUT
                        / "test45"
                        / method
                        / category
                        / "records"
                        / f"{cid:04d}"
                        / f"{sid:04d}.json"
                    )
                    assert rawpath.is_file() and recpath.is_file()
                    arr = np.load(rawpath, allow_pickle=False)
                    metric = json.loads(recpath.read_text())["metrics"]
                    panels.append(
                        (f"{method}_{cid}", f"{short} tune1 ({cid:04d})", arr, metric)
                    )
            assert len(panels) == 21
            fig, axs = plt.subplots(3, 7, figsize=(23.8, 14.2), layout="constrained")
            fig.suptitle(
                f"{category} {sid:04d} | 20% NPL-default PCM | 21 matched image panels",
                fontsize=17,
                weight="bold",
            )
            for (key, label, arr, metric), ax in zip(panels, axs.flat):
                assert arr.shape == (256, 256) and np.isfinite(arr).all()
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                caption = (
                    "Ground truth"
                    if metric is None
                    else f'PSNR {float(metric["psnr"]):.2f} dB | SSIM {float(metric["ssim"]):.3f}\n'
                    f'HaarPSI {float(metric["haarpsi"]):.3f} | MSE {float(metric["mse"]):.2e}'
                )
                ax.set_title(label + "\n" + caption, fontsize=7.8)
            fig.text(
                0.5,
                0.009,
                f"{category}, page {j}/9. AFM-hot cap 0.8 and fixed [0,1] intensity.",
                fontsize=8,
                ha="center",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("TUNE1_GALLERY", category, sid, j, flush=True)
    print("TUNE1_GALLERY_COMPLETE", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    a = ap.parse_args()
    main(a.category)
