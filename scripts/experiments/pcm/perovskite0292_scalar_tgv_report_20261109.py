"""Publication-ready comparison of MRPro-backed scalar TGV versus PCM baselines.

The experiment uses one frozen Perovskite 0292 20% NPL-default acquisition.
TGV was tuned per-image using ground truth PSNR, so this is oracle analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

from LION.utils.metrics import compute_metrics
from LION.visualize.plotting_functions import make_colormap

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1"
FROZEN = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
ADMM = ROOT / "bad_images_for_utv_20261109_v3_23/raw_new9/Perovskite_0292.npy"
UTV = (
    ROOT
    / "perovskite0292_c60_ternary_with_baselines_20261109_v1/ternary_selected_reconstruction.npz"
)
REFERENCE_METRICS = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def score(pred, ground):
    r = compute_metrics(
        recon=torch.from_numpy(pred)[None, None].float(),
        target=torch.from_numpy(ground)[None, None].float(),
        data_range=1.0,
    )
    return {k: float(r[k]) for k in METRICS}


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run():
    with np.load(FROZEN, allow_pickle=False) as z:
        ground = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
    with np.load(UTV, allow_pickle=False) as z:
        utv = z["reconstruction"].copy()
        rho = float(z["rho"])
        np.testing.assert_array_equal(ground, z["ground_truth"])
        np.testing.assert_array_equal(y, z["measurement"])
        np.testing.assert_array_equal(indices, z["indices"])
    pnp = np.load(ADMM, allow_pickle=False)
    tgv = np.load(OUT / "best_tgv.npy", allow_pickle=False)
    chosen = json.loads((OUT / "selected.json").read_text())
    assert ground.shape == pnp.shape == utv.shape == tgv.shape == (256, 256)
    assert all(np.isfinite(a).all() for a in (ground, pnp, utv, tgv))
    m = [
        r
        for r in read_rows(REFERENCE_METRICS)
        if r["category"] == "Perovskite" and int(r["sample_id"]) == 292
    ]
    assert len(m) == 1
    benchmark = m[0]
    assert sha(y) == benchmark["measurement_sha256"]
    assert sha(indices) == benchmark["pattern_sha256"]

    results = [
        {"name": "Ground truth", "image": ground, "scores": None},
        {"name": "PnP-ADMM-tuned-5", "image": pnp, "scores": score(pnp, ground)},
        {
            "name": f"UTV C60 (best rho={rho:.5f})",
            "image": utv,
            "scores": score(utv, ground),
        },
        {"name": "Scalar TGV (MRPro-PDHG)", "image": tgv, "scores": score(tgv, ground)},
    ]
    mse = np.mean((pnp.astype(np.float64) - ground.astype(np.float64)) ** 2)
    np.testing.assert_allclose(
        mse, float(benchmark["pnp_admm_ssdu5_mse"]), rtol=5e-5, atol=2e-8
    )
    assert abs(results[-1]["scores"]["psnr"] - chosen["selected"]["psnr"]) < 0.0001
    assert results[-1]["scores"]["psnr"] > results[2]["scores"]["psnr"]
    for r in results:
        if r["scores"] is not None:
            print("METRICS", r["name"], r["scores"], flush=True)
    alpha1 = float(chosen["selected"]["alpha1"])
    alpha0 = float(chosen["selected"]["alpha0"])
    iters = int(chosen["selected"]["iterations"])
    assert iters == 8192

    all_rows = []
    for name in (
        "coarse_512.csv",
        "fine_512.csv",
        "local_grid_512.csv",
        "validate_2048.csv",
        "validate_4096.csv",
        "validate_8192.csv",
        "local_top5_2048.csv",
        "local_top3_8192.csv",
    ):
        path = OUT / name
        if path.is_file():
            for row in read_rows(path):
                all_rows.append({**row, "table": name})
    assert len(all_rows) > 80
    with (OUT / "all_tuning_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["table", *[k for k in all_rows[0] if k != "table"]],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(all_rows)

    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10.5, "pdf.fonttype": 42}
    )
    pdfpath = OUT / "Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.pdf"
    pngpath = OUT / "Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.png"

    fig = plt.figure(figsize=(18.1, 7.55))
    g = fig.add_gridspec(
        1, 4, left=0.035, right=0.970, top=0.775, bottom=0.195, wspace=0.125
    )
    fig.suptitle(
        "Perovskite 0292 | Scalar TGV compared with PnP-ADMM and optimised C60",
        y=0.984,
        fontsize=19.2,
        weight="bold",
    )
    fig.text(
        0.5,
        0.924,
        "One frozen 20% Hadamard acquisition  |  NPL-default PCM noise  |  identical GT",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.878,
        "TGV parameters selected using this image's ground truth PSNR (single-image oracle)",
        ha="center",
        fontsize=10.6,
    )
    for i, x in enumerate(results):
        ax = fig.add_subplot(g[0, i])
        ax.imshow(
            x["image"],
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        m = x["scores"]
        caption = (
            "Reference image"
            if m is None
            else (
                f"PSNR {m['psnr']:.3f} dB  |  SSIM {m['ssim']:.3f}"
                f"\nHaarPSI {m['haarpsi']:.3f}"
            )
        )
        ax.set_title(
            x["name"] + "\n" + caption,
            fontsize=10.9,
            pad=10,
            weight="bold" if i == 3 else "normal",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5 if i == 3 else 0.5)
            spine.set_color("#222" if i == 3 else "#666")
    fig.text(
        0.035,
        0.148,
        f"TGV: alpha1 = {alpha1:.7g}, alpha0 = {alpha0:.7g}, "
        f"alpha0/alpha1 = {alpha0/alpha1:.4f}, {iters:,} PDHG iterations",
        fontsize=11.5,
    )
    fig.text(
        0.035,
        0.109,
        f"UTV C60: global rho = {rho:.7f}; C60 original map, 512 PDHG iterations."
        "    PnP-ADMM: fixed tuned-5 parameters.",
        fontsize=10.2,
    )
    fig.text(
        0.035,
        0.074,
        "TGV uses the existing LION AdaptiveTGVPCMNetwork with MRPro-based PDHG"
        " and componentwise L1 TGV terms; the two weights were tuned independently.",
        fontsize=9.6,
    )
    cb = fig.add_axes((0.32, 0.034, 0.36, 0.016))
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cb,
        orientation="horizontal",
    )
    fig.text(
        0.705,
        0.040,
        "Shared display [0,1]; metrics on unclipped images",
        fontsize=9,
        ha="left",
    )
    fig.savefig(pngpath, dpi=140, facecolor="white")
    # Preserve vector text when also exporting the PDF.

    # Provide a separate optimisation summary and fine-detail close-up on page 2.
    fig2 = plt.figure(figsize=(18.1, 10.5))
    outer = fig2.add_gridspec(
        2,
        4,
        left=0.04,
        right=0.96,
        top=0.87,
        bottom=0.10,
        wspace=0.12,
        hspace=0.34,
        height_ratios=[1.05, 1.1],
    )
    fig2.suptitle(
        "Perovskite 0292 | Zoomed texture and TGV parameter selection",
        y=0.975,
        fontsize=18.2,
        weight="bold",
    )
    fig2.text(
        0.5,
        0.934,
        "Crop: rows 40:185, columns 65:205; same region and intensity scale in every method",
        ha="center",
        fontsize=11.6,
    )
    for j, r in enumerate(results):
        a = fig2.add_subplot(outer[0, j])
        a.imshow(
            r["image"][40:185, 65:205],
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(r["name"], fontsize=11.5, pad=7)
    a = fig2.add_subplot(outer[1, :2])
    grid = [r for r in all_rows if int(float(r["iterations"])) == 512]
    p1 = np.asarray([float(r["alpha1"]) for r in grid])
    p0 = np.asarray([float(r["alpha0"]) for r in grid])
    psnr = np.asarray([float(r["psnr"]) for r in grid])
    scatter = a.scatter(
        p1, p0, c=psnr, cmap="viridis", s=75, alpha=0.82, edgecolors="none"
    )
    a.scatter(
        [alpha1],
        [alpha0],
        marker="*",
        s=280,
        c="red",
        edgecolor="black",
        linewidth=0.6,
        label="Selected TGV",
    )
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("alpha1 = weight on grad(u) - v")
    a.set_ylabel("alpha0 = weight on symgrad(v)")
    a.set_title(
        "Oracle PSNR over scanned TGV parameters (512 PDHG iterations)", fontsize=11.4
    )
    a.grid(alpha=0.18)
    a.legend(loc="upper left", fontsize=9)
    fig2.colorbar(scatter, ax=a, shrink=0.8, label="PSNR (dB)")
    a2 = fig2.add_subplot(outer[1, 2:])
    a2.axis("off")
    txt = [
        "TGV parameter search and convergence",
        f"Coarse: 35 pairs at 512 iterations",
        f"Refined grid: {len(read_rows(OUT/'fine_512.csv'))} pairs at 512 iterations",
        f"Local refinement: {len(read_rows(OUT/'local_grid_512.csv'))} pairs at 512 iterations",
        "Validation: strongest candidates at 2,048 / 4,096 / 8,192 iterations",
        "",
        f"Selected alpha1 = {alpha1:.9g}",
        f"Selected alpha0 = {alpha0:.9g}",
        f"PSNR = {results[-1]['scores']['psnr']:.4f} dB",
        f"SSIM = {results[-1]['scores']['ssim']:.4f}",
        f"HaarPSI = {results[-1]['scores']['haarpsi']:.4f}",
        "",
        f"Difference vs UTV C60: {results[-1]['scores']['psnr']-results[2]['scores']['psnr']:+.4f} dB",
        f"Difference vs PnP-ADMM: {results[-1]['scores']['psnr']-results[1]['scores']['psnr']:+.4f} dB",
        "",
        "Oracle tuning: GT is used to choose alpha1 and alpha0.",
        "Comparison across methods has different iteration costs.",
    ]
    a2.text(
        0.03,
        0.97,
        "\n".join(txt),
        transform=a2.transAxes,
        fontsize=11,
        linespacing=1.35,
        va="top",
    )
    fig2.text(
        0.5,
        0.044,
        "Reference: Bredies, Kunisch and Pock (2010), Total Generalized Variation,"
        " DOI 10.1137/090769521. MRPro: docs.mrpro.rocks",
        ha="center",
        fontsize=9.3,
    )
    fig2.text(
        0.5,
        0.020,
        "The tuned TGV penalty is componentwise L1 on gradient and symmetrised gradient;"
        " it is not a globally optimal TGV setting.",
        ha="center",
        fontsize=8.9,
    )

    with PdfPages(
        pdfpath,
        metadata={
            "Title": "Perovskite 0292: MRPro-backed TGV vs matched PCM methods",
            "Author": "PCM experiments",
            "Subject": "One frozen 20% NPL acquisition, PSNR-selected TGV weights",
        },
    ) as pdf:
        pdf.savefig(fig, dpi=190)
        pdf.savefig(fig2, dpi=185)
    plt.close(fig)
    plt.close(fig2)
    with (OUT / "comparison_metrics.csv").open("w", newline="") as f:
        rows = [{"method": r["name"], **r["scores"]} for r in results if r["scores"]]
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    protocol = {
        "case": "Perovskite_0292",
        "data": "frozen 20% multilevel Hadamard, default NPL noise",
        "measurement_sha256": sha(y),
        "sampled_indices_sha256": sha(indices),
        "solver": "existing LION.models.AdaptiveTGVPCMNetwork using LION.classical_algorithms.pdhg_mrpro and MRPro operators",
        "objective": "0.5||Au-y||^2 + alpha1||grad(u)-v||_1 + alpha0||symgrad(v)||_1",
        "norm": "anisotropic/componentwise L1 (per LION implementation)",
        "parameters_selected": chosen["selected"],
        "tuning": "PSNR oracle on same image; independent positive scalar alpha1 and alpha0",
        "pdhg_tau": 0.18,
        "pdhg_sigma": 0.18,
        "pdhg_theta": 1.0,
        "TGV_iterations": iters,
        "pnp_admm": "frozen tuned-5 parameters, no oracle",
        "utv_rho": rho,
        "comparison_metrics": {r["name"]: r["scores"] for r in results if r["scores"]},
        "fairness": "different algorithm iterations and per-iteration cost; compare image quality only, not speed",
        "references": [
            "https://doi.org/10.1137/090769521",
            "https://docs.mrpro.rocks/",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        "REPORT",
        pdfpath,
        pdfpath.stat().st_size,
        "PNG",
        pngpath,
        pngpath.stat().st_size,
        flush=True,
    )


if __name__ == "__main__":
    run()
