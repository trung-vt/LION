"""Update Perovskite 0292 matched comparison with GT-tuned scalar TV.

Retains original frozen PnP-ADMM tuned-5, GT-optimised UTV C60 and
GT-tuned scalar TGV-2. Adds a PSNR-tuned scalar TV reconstruction using the
same normalized Hadamard measurement and no new noise realization.
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
TGV = ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1"
TV = ROOT / "perovskite0292_scalar_tv_oracle_20261109_v1"
OUT = ROOT / "perovskite0292_scalar_tgv_with_tv_oracle_20261109_v2"
FROZEN = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
ADMM = ROOT / "bad_images_for_utv_20261109_v3_23/raw_new9/Perovskite_0292.npy"
UTV = (
    ROOT
    / "perovskite0292_c60_ternary_with_baselines_20261109_v1/ternary_selected_reconstruction.npz"
)
BENCHMARK = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def compute(x, gt):
    rec = torch.from_numpy(x)[None, None].float()
    target = torch.from_numpy(gt)[None, None].float()
    values = compute_metrics(recon=rec, target=target, data_range=1.0)
    return {k: float(values[k]) for k in METRICS}


def rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    with np.load(FROZEN, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        idx = z["sampled_indices"].copy()
    with np.load(UTV, allow_pickle=False) as z:
        u = z["reconstruction"].copy()
        rho = float(z["rho"])
        np.testing.assert_array_equal(gt, z["ground_truth"])
        np.testing.assert_array_equal(y, z["measurement"])
        np.testing.assert_array_equal(idx, z["indices"])
    pnp = np.load(ADMM, allow_pickle=False)
    scalar_tv = np.load(TV / "tv_best_scalar.npy", allow_pickle=False)
    scalar_tgv = np.load(TGV / "best_tgv.npy", allow_pickle=False)
    tv_data = json.loads((TV / "summary.json").read_text())
    tgv_data = json.loads((TGV / "selected.json").read_text())
    tv_lambda = float(tv_data["best"]["lambda"])
    a1, a0 = float(tgv_data["selected"]["alpha1"]), float(
        tgv_data["selected"]["alpha0"]
    )
    assert int(tv_data["best"]["iterations"]) == 8192
    assert int(tgv_data["selected"]["iterations"]) == 8192
    test_row = next(
        row
        for row in rows(BENCHMARK)
        if row["category"] == "Perovskite" and int(row["sample_id"]) == 292
    )
    assert digest(y) == test_row["measurement_sha256"]
    assert digest(idx) == test_row["pattern_sha256"]
    assert all(
        a.shape == gt.shape == (256, 256) and np.isfinite(a).all()
        for a in (pnp, u, scalar_tv, scalar_tgv)
    )
    np.testing.assert_allclose(
        np.mean((pnp.astype(np.float64) - gt.astype(np.float64)) ** 2),
        float(test_row["pnp_admm_ssdu5_mse"]),
        rtol=5e-5,
        atol=1e-8,
    )
    data = [
        ("Ground truth", gt, None),
        ("PnP-ADMM-tuned-5", pnp, compute(pnp, gt)),
        (f"UTV C60 (rho={rho:.5f})", u, compute(u, gt)),
        (f"Scalar TV (best lambda={tv_lambda:.5g})", scalar_tv, compute(scalar_tv, gt)),
        (
            f"Scalar TGV-2 (best alpha1={a1:.4g}, alpha0={a0:.4g})",
            scalar_tgv,
            compute(scalar_tgv, gt),
        ),
    ]
    np.testing.assert_allclose(
        data[3][2]["psnr"], tv_data["best"]["psnr"], atol=2e-5, rtol=0
    )
    np.testing.assert_allclose(
        data[4][2]["psnr"], tgv_data["selected"]["psnr"], atol=2e-5, rtol=0
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.pdf"
    png = OUT / "Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.png"
    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # Page 1: same scale, one row of five full images.
    fig = plt.figure(figsize=(22, 8.2))
    fig.suptitle(
        "Perovskite 0292 | Best scalar TV and TGV-2 compared with PnP-ADMM and C60",
        fontsize=20.5,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.930,
        "Frozen 20% multilevel Hadamard measurement | NPL-default noise | all methods share the identical observation",
        ha="center",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.897,
        "TV, TGV-2 and C60 scalings are tuned for this image using ground-truth PSNR (oracle comparison)",
        ha="center",
        fontsize=10.5,
    )
    gs = fig.add_gridspec(
        1, 5, left=0.027, right=0.973, top=0.78, bottom=0.245, wspace=0.09
    )
    for i, (label, image, score) in enumerate(data):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(
            image,
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            origin="upper",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            label
            if score is None
            else (
                f"{label}\nPSNR {score['psnr']:.3f} dB | SSIM {score['ssim']:.3f}\nHaarPSI {score['haarpsi']:.3f}"
            )
        )
        ax.set_title(
            title, fontsize=9.9, pad=9, weight="bold" if i in (3, 4) else "normal"
        )
        for spine in ax.spines.values():
            spine.set_color("#333" if i >= 3 else "#666")
            spine.set_linewidth(1.35 if i >= 3 else 0.55)
    fig.text(
        0.033,
        0.174,
        f"Scalar TV: lambda = {tv_lambda:.7g}, 8,192 PDHG iterations, MRPro-backed componentwise TV.",
        fontsize=11,
    )
    fig.text(
        0.033,
        0.139,
        f"Scalar TGV-2: alpha1 = {a1:.7g}, alpha0 = {a0:.7g}, 8,192 PDHG iterations, MRPro-backed componentwise TGV.",
        fontsize=11,
    )
    fig.text(
        0.033,
        0.101,
        f"UTV C60: rho = {rho:.7f}, 512 PDHG iterations. PnP-ADMM: frozen tuned-5 parameters.",
        fontsize=10.7,
    )
    fig.text(
        0.033,
        0.069,
        "All PSNR/SSIM/HaarPSI metrics use raw (unclipped) images. Colour display fixed to [0,1].",
        fontsize=9.7,
    )
    cax = fig.add_axes((0.37, 0.029, 0.27, 0.015))
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    fig.savefig(png, dpi=135)

    # Page 2: same close-up area and parameter tuning diagnostics.
    fig2 = plt.figure(figsize=(22, 12))
    fig2.suptitle(
        "Perovskite 0292 | Texture crop and scalar TV / TGV-2 tuning",
        fontsize=20,
        y=0.980,
        weight="bold",
    )
    fig2.text(
        0.5,
        0.935,
        "Crop: rows 40:185, columns 65:205 | identical crop and [0,1] display scale in all methods",
        ha="center",
        fontsize=11,
    )
    grid = fig2.add_gridspec(
        2,
        10,
        left=0.035,
        right=0.965,
        top=0.875,
        bottom=0.12,
        wspace=0.22,
        hspace=0.37,
        height_ratios=(1, 1.05),
    )
    for j, (label, image, score) in enumerate(data):
        ax = fig2.add_subplot(grid[0, 2 * j : 2 * j + 2])
        ax.imshow(
            image[40:185, 65:205], cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10.3, pad=8)
    # TV scalar parameter scan at fixed 512 iterations.
    tv_scan = rows(TV / "coarse_512.csv") + rows(TV / "fine_512.csv")
    scan_data = {}
    for r in tv_scan:
        scan_data[round(float(r["lambda"]), 10)] = (
            float(r["lambda"]),
            float(r["psnr"]),
        )
    ordered = sorted(scan_data.values())
    ax = fig2.add_subplot(grid[1, :3])
    ax.plot(
        [r[0] for r in ordered],
        [r[1] for r in ordered],
        marker="o",
        markersize=3.3,
        linewidth=1.1,
        label="512 PDHG iterations",
    )
    ax.scatter(
        [tv_lambda],
        [float(tv_data["best"]["psnr"])],
        marker="*",
        s=130,
        color="red",
        edgecolor="black",
        linewidth=0.5,
        label="Chosen (8192 iters)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("TV scalar lambda")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("TV scalar parameter scan", fontsize=11.5)
    # TGV two-dimensional parameter scan, all evaluated at 512 steps.
    tgv_scan = (
        rows(TGV / "coarse_512.csv")
        + rows(TGV / "fine_512.csv")
        + rows(TGV / "local_grid_512.csv")
    )
    ax2 = fig2.add_subplot(grid[1, 3:7])
    x = np.array([float(r["alpha1"]) for r in tgv_scan])
    yv = np.array([float(r["alpha0"]) for r in tgv_scan])
    zs = np.array([float(r["psnr"]) for r in tgv_scan])
    points = ax2.scatter(x, yv, c=zs, s=40, alpha=0.85, cmap="viridis")
    ax2.scatter(
        [a1],
        [a0],
        marker="*",
        s=170,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        label="Chosen TGV-2",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("TGV alpha1")
    ax2.set_ylabel("TGV alpha0")
    ax2.set_title("TGV-2 parameter scan (512 steps)", fontsize=11.5)
    ax2.grid(alpha=0.18)
    ax2.legend(loc="upper left", fontsize=8)
    fig2.colorbar(points, ax=ax2, shrink=0.68, label="PSNR (dB)")
    # Compact textual comparison and method selection warning.
    ax3 = fig2.add_subplot(grid[1, 7:])
    ax3.axis("off")
    tvm = data[3][2]
    tgvm = data[4][2]
    pm = data[1][2]
    um = data[2][2]
    lines = [
        "Selected scalar regularisation",
        f"TV lambda = {tv_lambda:.7g}",
        f"TGV-2 alpha1 = {a1:.7g}",
        f"TGV-2 alpha0 = {a0:.7g}",
        "",
        f"TV PSNR: {tvm['psnr']:.3f} dB",
        f"TGV-2 PSNR: {tgvm['psnr']:.3f} dB",
        f"TV - TGV: {tvm['psnr']-tgvm['psnr']:+.3f} dB",
        "",
        f"TV vs PnP: {tvm['psnr']-pm['psnr']:+.3f} dB",
        f"TV vs C60: {tvm['psnr']-um['psnr']:+.3f} dB",
        "",
        "TV and TGV are both GT-tuned",
        "for this single image.",
    ]
    ax3.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        fontsize=10.2,
        transform=ax3.transAxes,
        linespacing=1.35,
    )
    fig2.text(
        0.5,
        0.069,
        f"Archived validation-tuned TV (lambda=0.011; 1024 iterations): PSNR {float(test_row['tv_psnr']):.3f} dB"
        " | not the oracle scalar TV selected for this comparison.",
        ha="center",
        fontsize=9.6,
    )
    fig2.text(
        0.5,
        0.033,
        "TGV-2 and TV implementations: existing LION AdaptiveTGVPCMNetwork / AdaptiveTVPCMNetwork"
        " with MRPro-backed PDHG. Scores and plots refer to the same frozen PCM measurement.",
        ha="center",
        fontsize=9,
    )

    with PdfPages(
        pdf,
        metadata={
            "Title": "Perovskite 0292: scalar TV, scalar TGV-2, PnP-ADMM and learned C60",
            "Author": "PCM reconstruction experiments",
        },
    ) as pages:
        pages.savefig(fig, dpi=170)
        pages.savefig(fig2, dpi=170)
    plt.close(fig)
    plt.close(fig2)

    compare = []
    for label, image, score in data:
        if score is not None:
            compare.append({"method": label, **score})
    with (OUT / "comparison_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compare[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(compare)
    np.savez_compressed(
        OUT / "five_way_raw_arrays.npz",
        ground_truth=gt,
        measurement=y,
        sampled_indices=idx,
        pnp_admm_tuned5=pnp,
        utv_c60_best=u,
        scalar_tv_best=scalar_tv,
        scalar_tgv2_best=scalar_tgv,
        utv_rho=np.float64(rho),
        tv_lambda=np.float64(tv_lambda),
        tgv_alpha1=np.float64(a1),
        tgv_alpha0=np.float64(a0),
        tv_iterations=np.int64(8192),
        tgv_iterations=np.int64(8192),
    )
    protocol = {
        "case": "Perovskite_0292",
        "measurement_sha256": digest(y),
        "sampled_indices_sha256": digest(idx),
        "new_method": "PSNR-tuned scalar TV with existing MRPro-backed PDHG",
        "tv_lambda": tv_lambda,
        "tv_iterations": 8192,
        "tgv_alpha1": a1,
        "tgv_alpha0": a0,
        "tgv_iterations": 8192,
        "utv_rho": rho,
        "utv_iterations": 512,
        "validation_tuned_tv_psnr": float(test_row["tv_psnr"]),
        "score_source": "ground truth, single-image oracle",
        "comparison_metrics": {
            r["method"]: {k: r[k] for k in METRICS} for r in compare
        },
        "original_tgv_pdf": str(
            TGV / "Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.pdf"
        ),
        "tv_search_summary": str(TV / "summary.json"),
        "tgv_search_summary": str(TGV / "selected.json"),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    assert pdf.stat().st_size > 100000
    print("NEW_PDF", pdf, pdf.stat().st_size, flush=True)
    print("NEW_CSV", OUT / "comparison_metrics.csv")
    for r in compare:
        print("METRIC_ROW", r)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
