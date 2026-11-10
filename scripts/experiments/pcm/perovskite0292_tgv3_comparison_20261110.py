"""Render an exact-acquisition comparison of scalar TGV3, TGV2, TV, UTV and PnP.

The TGV-3 source implementation lives on feature/pcm-tgv3-mrpro-20261109.
TGV3, TGV2, TV and C60 multipliers are single-image PSNR oracle choices.
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

from LION.visualize.plotting_functions import make_colormap
from LION.utils.metrics import compute_metrics

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_scalar_tgv3_oracle_20261110_v1"
GT_ARCHIVE = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
PNP = ROOT / "bad_images_for_utv_20261109_v3_23/raw_new9/Perovskite_0292.npy"
UTV = (
    ROOT
    / "perovskite0292_c60_ternary_with_baselines_20261109_v1/ternary_selected_reconstruction.npz"
)
TV = ROOT / "perovskite0292_scalar_tv_oracle_20261109_v1/tv_best_scalar.npy"
TV_META = ROOT / "perovskite0292_scalar_tv_oracle_20261109_v1/summary.json"
TGV2 = ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1/best_tgv.npy"
TGV2_META = ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1/selected.json"
TGV3 = OUT / "tgv3_best_scalar.npy"
TGV3_META = OUT / "summary.json"
METRIC450 = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
METRIC = ("psnr", "ssim", "haarpsi", "mse")


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def score(a, gt):
    values = compute_metrics(
        recon=torch.from_numpy(a)[None, None].float(),
        target=torch.from_numpy(gt)[None, None].float(),
        data_range=1.0,
    )
    return {m: float(values[m]) for m in METRIC}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run():
    with np.load(GT_ARCHIVE, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
    with np.load(UTV, allow_pickle=False) as z:
        utv = z["reconstruction"].copy()
        rho = float(z["rho"])
        np.testing.assert_array_equal(z["ground_truth"], gt)
        np.testing.assert_array_equal(z["measurement"], y)
        np.testing.assert_array_equal(z["indices"], indices)
    tgv3_meta = json.loads(TGV3_META.read_text())
    tv_meta = json.loads(TV_META.read_text())
    tgv2_meta = json.loads(TGV2_META.read_text())
    row = next(
        r
        for r in read_csv(METRIC450)
        if r["category"] == "Perovskite" and int(r["sample_id"]) == 292
    )
    assert digest(y) == row["measurement_sha256"]
    assert digest(indices) == row["pattern_sha256"]
    assert digest(y) == tgv3_meta["measurement_sha256"]
    assert digest(indices) == tgv3_meta["sampled_indices_sha256"]

    images = [
        ("Ground truth", gt, None),
        ("PnP-ADMM-tuned-5", np.load(PNP, allow_pickle=False), None),
        ("UTV C60", utv, {"rho": rho}),
        (
            "Scalar TV",
            np.load(TV, allow_pickle=False),
            {"lambda": float(tv_meta["best"]["lambda"]), "iterations": 8192},
        ),
        (
            "Scalar TGV-2",
            np.load(TGV2, allow_pickle=False),
            {
                "alpha1": float(tgv2_meta["selected"]["alpha1"]),
                "alpha0": float(tgv2_meta["selected"]["alpha0"]),
                "iterations": 8192,
            },
        ),
        (
            "Scalar TGV-3",
            np.load(TGV3, allow_pickle=False),
            {
                "alpha2": float(tgv3_meta["weights"][0]),
                "alpha1": float(tgv3_meta["weights"][1]),
                "alpha0": float(tgv3_meta["weights"][2]),
                "iterations": 8192,
            },
        ),
    ]
    assert len(images) == 6
    assert all(
        a.shape == gt.shape == (256, 256) and bool(np.isfinite(a).all())
        for _, a, _ in images
    )
    np.testing.assert_allclose(
        np.mean((images[1][1].astype("float64") - gt.astype("float64")) ** 2),
        float(row["pnp_admm_ssdu5_mse"]),
        rtol=5e-5,
        atol=1e-8,
    )

    table = []
    for i, (name, image, settings) in enumerate(images):
        if i == 0:
            continue
        stats = score(image, gt)
        table.append({"method": name, **stats})
        if name == "Scalar TGV-3":
            np.testing.assert_allclose(
                stats["psnr"], tgv3_meta["final_metrics"]["psnr"], rtol=0, atol=3e-5
            )
        print("METHOD", name, settings, stats, flush=True)
    with (OUT / "comparison_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)

    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "font.size": 11,
            "figure.facecolor": "white",
        }
    )
    pdfpath = OUT / "Perovskite_0292_TGV3_vs_TGV2_TV_UTV_PnP.pdf"
    pngpath = OUT / "Perovskite_0292_TGV3_vs_TGV2_TV_UTV_PnP.png"
    fig = plt.figure(figsize=(22, 7.65))
    fig.suptitle(
        "Perovskite 0292 | Third-order TGV against existing PCM methods",
        fontsize=20.4,
        weight="bold",
        y=0.983,
    )
    fig.text(
        0.5,
        0.935,
        "Same stored 20% Hadamard acquisition, NPL-default noise and ground truth",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.906,
        "TV, TGV-2, TGV-3 and C60 multipliers tuned on this one image's ground-truth PSNR",
        ha="center",
        fontsize=10.4,
    )
    gs = fig.add_gridspec(
        1, 6, left=0.025, right=0.975, top=0.805, bottom=0.245, wspace=0.08
    )
    for idx, (name, image, settings) in enumerate(images):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(
            image, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", rasterized=True
        )
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            name
            if idx == 0
            else (
                f"{name}\nPSNR {table[idx-1]['psnr']:.3f} dB | SSIM {table[idx-1]['ssim']:.3f}"
                f"\nHaarPSI {table[idx-1]['haarpsi']:.3f}"
            )
        )
        ax.set_title(
            title, fontsize=10.6, pad=9, weight="bold" if idx == 5 else "normal"
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.6 if idx == 5 else 0.55)
            spine.set_color("#222" if idx == 5 else "#666")
    alpha2, alpha1, alpha0 = tgv3_meta["weights"]
    fig.text(
        0.028,
        0.172,
        f"TGV-3: alpha2={alpha2:.6g}, alpha1={alpha1:.6g}, alpha0={alpha0:.6g}; "
        "8,192 MRPro-backed PDHG iterations; anisotropic full-tensor L1 penalty.",
        fontsize=11.2,
    )
    fig.text(
        0.028,
        0.136,
        f"TGV-2: alpha1={tgv2_meta['selected']['alpha1']:.6g}, "
        f"alpha0={tgv2_meta['selected']['alpha0']:.6g}; "
        f"TV: lambda={tv_meta['best']['lambda']:.6g}; both at 8,192 iterations.",
        fontsize=10.8,
    )
    fig.text(
        0.028,
        0.101,
        f"UTV C60: rho={rho:.6g}, 512 PDHG iterations; PnP-ADMM uses fixed tuned-5 parameters.",
        fontsize=10.6,
    )
    fig.text(
        0.028,
        0.063,
        "Images displayed with fixed [0,1] intensity limits; PSNR/SSIM/HaarPSI use raw unclipped arrays.",
        fontsize=9.8,
    )
    cbax = fig.add_axes((0.40, 0.023, 0.22, 0.017))
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cbax,
        orientation="horizontal",
    )
    fig.savefig(pngpath, dpi=150)

    fig2 = plt.figure(figsize=(22, 10.8))
    fig2.suptitle(
        "Perovskite 0292 | Cropped texture and TGV-3 three-weight search",
        fontsize=20.3,
        weight="bold",
        y=0.975,
    )
    fig2.text(
        0.5,
        0.936,
        "Same crop for every method: rows 40:185, columns 65:205; fixed display [0,1]",
        fontsize=11.4,
        ha="center",
    )
    gs2 = fig2.add_gridspec(
        2,
        12,
        left=0.034,
        right=0.965,
        top=0.884,
        bottom=0.11,
        wspace=0.18,
        hspace=0.40,
        height_ratios=(1.02, 1.08),
    )
    for j, (name, image, settings) in enumerate(images):
        ax = fig2.add_subplot(gs2[0, 2 * j : 2 * j + 2])
        ax.imshow(
            image[40:185, 65:205], cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=10.6, pad=8)
    coarse = read_csv(OUT / "coarse_512.csv")
    scatter_ax = fig2.add_subplot(gs2[1, :5])
    ratios = np.array([float(r["alpha0"]) / float(r["alpha1"]) for r in coarse])
    a2_values = np.array([float(r["alpha2"]) for r in coarse])
    psnr_values = np.array([float(r["psnr"]) for r in coarse])
    scat = scatter_ax.scatter(
        a2_values, ratios, c=psnr_values, cmap="viridis", s=50, alpha=0.78
    )
    scatter_ax.scatter(
        [alpha2],
        [alpha0 / alpha1],
        marker="*",
        s=260,
        c="red",
        edgecolor="black",
        linewidth=0.8,
        label="Final selected",
    )
    scatter_ax.set_xscale("log")
    scatter_ax.set_yscale("log")
    scatter_ax.set_xlabel("alpha2 = gradient mismatch weight")
    scatter_ax.set_ylabel("alpha0 / alpha1")
    scatter_ax.set_title("Coarse 512-iteration TGV-3 scan", fontsize=12)
    scatter_ax.legend(loc="lower left", fontsize=8)
    scatter_ax.grid(alpha=0.2)
    fig2.colorbar(scat, ax=scatter_ax, shrink=0.8, label="PSNR (dB)")

    ax2 = fig2.add_subplot(gs2[1, 5:8])
    for name in (
        "coarse_512.csv",
        "coarse_shortlist_2048.csv",
        "local_1024.csv",
        "shortlist_4096.csv",
        "finalists_8192.csv",
    ):
        records = read_csv(OUT / name)
        if records:
            ax2.scatter(
                [int(float(r["iterations"])) for r in records],
                [float(r["psnr"]) for r in records],
                s=29,
                alpha=0.45,
                label=name.replace(".csv", ""),
            )
    ax2.axhline(
        float(tgv3_meta["final_metrics"]["psnr"]),
        linestyle="--",
        linewidth=1,
        color="black",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("PDHG iterations, different candidates")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Oracle search by iteration budget", fontsize=11.4)
    ax2.grid(alpha=0.18)
    ax2.legend(loc="lower right", fontsize=7.8)

    ax3 = fig2.add_subplot(gs2[1, 8:])
    ax3.axis("off")
    metric_map = {r["method"]: r for r in table}
    difference_tgv2 = (
        metric_map["Scalar TGV-3"]["psnr"] - metric_map["Scalar TGV-2"]["psnr"]
    )
    difference_tv = metric_map["Scalar TGV-3"]["psnr"] - metric_map["Scalar TV"]["psnr"]
    difference_pnp = (
        metric_map["Scalar TGV-3"]["psnr"] - metric_map["PnP-ADMM-tuned-5"]["psnr"]
    )
    text = [
        "Scalar TGV-3: selected on GT PSNR",
        f"alpha2 = {alpha2:.8g}",
        f"alpha1 = {alpha1:.8g}",
        f"alpha0 = {alpha0:.8g}",
        f"PSNR = {metric_map['Scalar TGV-3']['psnr']:.4f} dB",
        f"SSIM = {metric_map['Scalar TGV-3']['ssim']:.4f}",
        f"HaarPSI = {metric_map['Scalar TGV-3']['haarpsi']:.4f}",
        "",
        f"PSNR gain vs TGV-2: {difference_tgv2:+.4f} dB",
        f"PSNR gain vs TV: {difference_tv:+.4f} dB",
        f"PSNR gap vs PnP: {difference_pnp:+.4f} dB",
        "",
        "Finite three-weight parameter scan.",
        "Different methods have different",
        "per-iteration computational costs.",
        "No convergence guarantee is claimed",
        "for LION's existing relaxation rule.",
    ]
    ax3.text(
        0.01,
        0.98,
        "\n".join(text),
        transform=ax3.transAxes,
        va="top",
        fontsize=11.2,
        linespacing=1.42,
    )
    fig2.text(
        0.5,
        0.058,
        "TGV-3 uses existing LION/MRPro PDHG with a new 4x3 operator. "
        "The shared solver uses interpolation relaxation, not standard CP extrapolation.",
        ha="center",
        fontsize=9.7,
    )
    fig2.text(
        0.5,
        0.028,
        "Bredies, Kunisch and Pock (2010), Total Generalized Variation, Eq. (3.6), DOI: 10.1137/090769521.",
        ha="center",
        fontsize=9,
    )

    with PdfPages(
        pdfpath,
        metadata={
            "Title": "Perovskite 0292: Scalar TGV3 vs TGV2, TV, C60, PnP-ADMM",
            "Author": "PCM experiments",
        },
    ) as pages:
        pages.savefig(fig, dpi=180)
        pages.savefig(fig2, dpi=180)
    plt.close(fig)
    plt.close(fig2)

    np.savez_compressed(
        OUT / "six_way_raw_arrays.npz",
        ground_truth=gt,
        measurement=y,
        sampled_indices=indices,
        pnp_admm_tuned5=images[1][1],
        utv_c60_best=images[2][1],
        scalar_tv_best=images[3][1],
        scalar_tgv2_best=images[4][1],
        scalar_tgv3_best=images[5][1],
        utv_rho=np.float64(rho),
        tv_lambda=np.float64(tv_meta["best"]["lambda"]),
        tgv2_alpha1=np.float64(tgv2_meta["selected"]["alpha1"]),
        tgv2_alpha0=np.float64(tgv2_meta["selected"]["alpha0"]),
        tgv3_alpha2=np.float64(alpha2),
        tgv3_alpha1=np.float64(alpha1),
        tgv3_alpha0=np.float64(alpha0),
    )
    protocol = {
        "case": "Perovskite_0292",
        "status": "complete",
        "measurement_sha256": digest(y),
        "sampled_indices_sha256": digest(indices),
        "method": "TGV3 scalar parameters optimized by ground-truth PSNR",
        "stopping": "chosen final reconstruction after 8192 PDHG iterations",
        "tgv3_weights": tgv3_meta["weights"],
        "tgv3_metrics": tgv3_meta["final_metrics"],
        "comparison": {r["method"]: {m: r[m] for m in METRIC} for r in table},
        "output_pdf": str(pdfpath),
        "oracle_warning": "All scalar variational and C60 map scaling values were tuned per image using GT",
        "solver_warning": "Existing LION PDHG relaxation differs from textbook Chambolle-Pock extrapolation",
    }
    (OUT / "report_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        "PDF_READY",
        pdfpath,
        pdfpath.stat().st_size,
        "PNG",
        pngpath.stat().st_size,
        flush=True,
    )


if __name__ == "__main__":
    run()
