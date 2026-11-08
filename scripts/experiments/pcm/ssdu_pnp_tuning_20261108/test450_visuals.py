"""Show 14 qualitative fixed-PCM test cases with all 11 frozen methods plus target.

Reuses previously stored historical raw image arrays for seven older methods;
uses exact new final-z output arrays for GS-PnP oracle/SSDU methods. All images
on a page share one original noisy measurement and common AFM-hot [0,1].
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_reconstruct import (
    OUT,
    ROOT,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_report import METHODS

CASES = (
    ("Reference", 21),
    ("Reference", 66),
    ("Reference", 188),
    ("Reference", 727),
    ("CIGS", 710),
    ("CIGS", 727),
    ("CIGS", 805),
    ("CIGS", 821),
    ("CIGS", 873),
    ("Perovskite", 31),
    ("Perovskite", 79),
    ("Perovskite", 98),
    ("Perovskite", 260),
    ("Perovskite", 288),
)
OLD = ROOT / "ten_methods_matched_450_20261105_v1/visual_14_paired_exact_v1/raw"
MAPPING = {
    "zero_filled": "zero_filled",
    "spgl1_bp": "spgl1_bp",
    "spgl1_bpdn": "bpdn",
    "pnp_admm_oracle": "pnp_tuned",
    "tv": "tv",
    "utv_c60": "C",
    "pnp_admm_ssdu5": "pnp_tuned",
    "pnp_admm_ssdu50": "pnp_tuned",
}
NEW = {
    "gs_pnp_oracle": "gs_oracle_step6",
    "gs_pnp_ssdu5": "gs_ssdu",
    "gs_pnp_ssdu50": "gs_ssdu",
}


def main():
    results = OUT / "merged/all_450_11_method_metrics.csv"
    assert results.is_file()
    with results.open(newline="") as f:
        merged = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    assert len(merged) == 450
    assert len(CASES) == 14
    cmap = make_colormap("afmhot", 0.8)
    target = OUT / "figures/14_examples_all_11_methods.pdf"
    assert not target.exists()
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10.0, "pdf.fonttype": 42}
    )
    with PdfPages(
        target, metadata={"Title": "Fourteen PCM cases: 11 matched methods and target"}
    ) as pdf:
        for index, (cat, sid) in enumerate(CASES, 1):
            oldfile = OLD / f"{cat}_{sid:04d}.npz"
            assert oldfile.exists(), oldfile
            with np.load(oldfile, allow_pickle=False) as arr:
                old = {k: arr[k].copy() for k in arr.files}
            pair = merged[cat, sid]
            images = [("Target", old["ground_truth"], None)]
            for m, label, _ in METHODS:
                if m in MAPPING:
                    im = old[MAPPING[m]]
                else:
                    im = np.load(
                        OUT / "raw" / NEW[m] / f"{cat}_{sid:04d}.npy",
                        allow_pickle=False,
                    )
                assert im.shape == (256, 256) and np.isfinite(im).all(), (cat, sid, m)
                images.append((label, im, m))
            assert len(images) == 12
            fig, axs = plt.subplots(2, 6, figsize=(18.8, 8.6), layout="constrained")
            fig.suptitle(
                f"{cat} | image {sid:04d} | 20% PCM NPL-noise | 1 target + 11 methods",
                fontsize=16,
                weight="bold",
            )
            for (title, im, m), ax in zip(images, axs.flat):
                ax.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if m is None:
                    subtitle = "Ground truth"
                else:
                    ps = float(pair[f"{m}_psnr"])
                    ss = float(pair[f"{m}_ssim"])
                    subtitle = f"PSNR {ps:.2f} dB | SSIM {ss:.3f}"
                ax.set_title(title + "\n" + subtitle, fontsize=9.2, weight="medium")
            fig.text(
                0.5,
                0.013,
                f"Page {index}/14. AFM-hot capped at 0.8. Same intensity range [0,1] for all panels.",
                fontsize=8.5,
                ha="center",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("VISUAL", index, "/14", cat, sid, flush=True)
    print("VISUAL_PDF", target, target.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
