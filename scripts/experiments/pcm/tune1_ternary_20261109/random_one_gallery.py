"""Make 45 pages: target and the ten requested methods, one fixed 20% PCM acquisition."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import torch

from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import oldbench
from scripts.experiments.pcm.tune1_ternary_20261109.prepare import OUTPUT, CATEGORIES
from scripts.experiments.pcm.tune1_ternary_20261109.random_one_analysis import (
    ROOT,
    METHODS,
)

PREV = OUTPUT.parent / "ssdu_pnp_tuning_20261108_v1"
BASE = OUTPUT / "test45/baselines/raw"
OLD = {
    "gs_pnp_oracle": PREV / "test450/v1/raw/gs_oracle_step6",
    "gs_pnp_ssdu5": PREV / "test450/v1/raw/gs_ssdu",
}


def raw_path(method, category, sid, choice):
    if method in OLD:
        return OLD[method] / f"{category}_{sid:04d}.npy"
    if method == "pnp_admm_ssdu5":
        return BASE / "pnp_admm_oracle" / f"{category}_{sid:04d}.npy"
    if method == "pnp_admm_tune1":
        cal = choice["selected_calibrations"]["admm"][category]["calibration_sample_id"]
        return ROOT / "recon/admm" / category / "raw" / f"{cal:04d}" / f"{sid:04d}.npy"
    if method == "gs_pnp_tune1":
        cal = choice["selected_calibrations"]["gs"][category]["calibration_sample_id"]
        return ROOT / "recon/gs" / category / "raw" / f"{cal:04d}" / f"{sid:04d}.npy"
    return BASE / method / f"{category}_{sid:04d}.npy"


def main(category):
    assert category in CATEGORIES
    selection = json.loads((ROOT / "selection.json").read_text())
    manifest = json.loads((OUTPUT / "prepared/test45.json").read_text())
    cases = sorted(
        (x for x in manifest["cases"] if x["category"] == category),
        key=lambda x: x["sample_id"],
    )
    assert len(cases) == 9
    csvfile = ROOT / "analysis/test45_10_methods_per_image.csv"
    with csvfile.open(newline="") as f:
        rows = {(x["category"], int(x["sample_id"])): x for x in csv.DictReader(f)}
    assert len(rows) == 45
    assert len(METHODS) == 10
    cmap = make_colormap("afmhot", 0.8)
    folder = ROOT / "analysis/galleries"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{category}_9_Pages.pdf"
    if output.exists():
        raise FileExistsError(output)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9.2, "pdf.fonttype": 42}
    )
    with PdfPages(
        output,
        metadata={
            "Title": f"PCM test45: {category}, ten methods with PSNR SSIM HaarPSI MSE"
        },
    ) as writer:
        for page, case in enumerate(cases, 1):
            sid = int(case["sample_id"])
            truthfile = oldbench.DATA / category / f"image{sid:04d}.png"
            gt = (
                np.asarray(Image.open(truthfile).convert("L"), dtype=np.uint8).astype(
                    np.float32
                )
                / 255.0
            )
            assert gt.shape == (256, 256)
            imgs = [("target", "Target", gt)]
            data = rows[category, sid]
            for key, name in METHODS:
                fname = raw_path(key, category, sid, selection)
                assert fname.is_file(), fname
                im = np.load(fname, allow_pickle=False)
                assert im.shape == (256, 256) and np.isfinite(im).all()
                imgs.append((key, name, im))
            assert len(imgs) == 11
            fig, axs = plt.subplots(3, 4, figsize=(16.5, 11.9), layout="constrained")
            fig.suptitle(
                f"{category} | test image {sid:04d} | 20% Hadamard + NPL-default noise",
                fontsize=15,
                weight="bold",
            )
            for i, ax in enumerate(axs.flat):
                if i >= len(imgs):
                    ax.axis("off")
                    continue
                key, label, image = imgs[i]
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if key == "target":
                    caption = "Ground truth"
                else:
                    if key == "spgl1_bp":
                        # Historical SPGL1-BP is not bitwise replayable on
                        # all samples. Show the metric of the *displayed*
                        # image, while report tables retain archived metrics.
                        scored = oldbench.score(
                            torch.from_numpy(np.ascontiguousarray(image))[None, None],
                            torch.from_numpy(np.ascontiguousarray(gt))[None, None],
                            float(gt.max() - gt.min()),
                        )
                    else:
                        scored = {
                            k: float(data[key + "_" + k])
                            for k in ("psnr", "ssim", "haarpsi", "mse")
                        }
                    cap = (
                        f'PSNR {scored["psnr"]:.2f} dB | SSIM {scored["ssim"]:.4f}\n'
                        f'HaarPSI {scored["haarpsi"]:.4f} | MSE {scored["mse"]:.2e}'
                    )
                    if key.endswith("_tune1"):
                        family = "admm" if key == "pnp_admm_tune1" else "gs"
                        cid = selection["selected_calibrations"][family][category][
                            "calibration_sample_id"
                        ]
                        label = label + f" ({cid:04d})"
                    caption = cap
                ax.set_title(label + "\n" + caption, pad=4, fontsize=8.5)
            writer.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("GALLERY", category, sid, page, "/9", flush=True)
    print("CATEGORY_VISUAL_COMPLETE", output, output.stat().st_size, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    args = ap.parse_args()
    main(args.category)
