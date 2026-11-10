"""Append verified n=256/512/1024 checkpoint comparisons to the six-page PCM report.

All rendering and PDF merging take place on tuni. Raw checkpoint arrays
were generated with identical frozen acquisition and local solver copies.
"""
from __future__ import annotations
import csv, json, subprocess
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
CHK = ROOT / "perovskite0292_relaxation_three_methods_checkpoints_20261110_v1"
OLD = (
    ROOT
    / "perovskite0292_relaxation_three_methods_archive_20261110_v1/Perovskite_0292_TV_TGV2_TGV3_Relaxation_Experiment_Report.pdf"
)
NPZ = CHK / "checkpoint_reconstructions_and_residuals.npz"
SUP = CHK / "checkpoint_supplement.pdf"
FINAL = CHK / "Perovskite_0292_TV_TGV2_TGV3_Relaxation_Experiment_Report_UPDATED.pdf"
STEPS = (256, 512, 1024)
METHODS = (("TV", "tv"), ("TGV-2", "tgv2"), ("TGV-3", "tgv3"))
VARS = (("LION_existing", "Existing LION"), ("canonical_CP", "Canonical CP"))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
with (CHK / "checkpoint_metrics.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
record = {(r["method"], r["variant"], int(r["iterations"])): r for r in rows}
assert len(rows) == 18
with np.load(NPZ, allow_pickle=False) as z:
    images = {k: z[k].copy() for k in z.files}
gt = images["ground_truth"]
indices = images["sampled_indices"]
assert gt.shape == (256, 256)
all_error = np.concatenate(
    [
        images[f"{short}_{rule}_n{n}_signed_error"].reshape(-1)
        for _, short in METHODS
        for rule, _ in VARS
        for n in STEPS
    ]
)
all_measured = np.concatenate(
    [
        images[f"{short}_{rule}_n{n}_measurement_residual"]
        for _, short in METHODS
        for rule, _ in VARS
        for n in STEPS
    ]
)
error_cap = float(np.percentile(np.abs(all_error), 99.5))
measurement_cap = float(np.percentile(np.abs(all_measured), 99.5))
assert error_cap > 0 and measurement_cap > 0


def colorbar(fig, cap, label, cmap="coolwarm", left=0.31, width=0.38, y=0.062):
    ax = fig.add_axes([left, y, width, 0.016])
    cb = fig.colorbar(
        ScalarMappable(norm=TwoSlopeNorm(vcenter=0, vmin=-cap, vmax=cap), cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label(label, fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)


def header(fig, num, title, subtitle):
    fig.text(
        0.045,
        0.974,
        f"{num}  {title}",
        weight="bold",
        size=19.2,
        va="top",
        color="#172b3d",
    )
    fig.text(0.045, 0.932, subtitle, size=10.8, va="top", color="#4e6479")
    fig.add_artist(
        plt.Line2D(
            [0.045, 0.957],
            [0.901, 0.901],
            transform=fig.transFigure,
            color="#c0d0df",
            lw=1,
        )
    )


def foot(fig, text):
    fig.text(0.045, 0.023, text, fontsize=8.2, color="#606d7a", va="bottom")


def f(r, name, nd=3):
    return f"{float(r[name]):.{nd}f}"


def title_for(method, variant, n):
    row = record[method, variant, n]
    return (
        f"{method} | {dict(VARS)[variant]}\n"
        f"PSNR {f(row,'psnr',3)} dB  |  SSIM {f(row,'ssim',3)}  |  HaarPSI {f(row,'haarpsi',3)}\n"
        f"Elapsed {f(row,'seconds',3)} s  |  objective {f(row,'objective',3)}"
    )


def visual_page(pdf, n, page):
    fig = plt.figure(figsize=(17.6, 12.4))
    header(
        fig,
        page,
        f"Reconstructions at {n:,} PDHG iterations",
        "Each cell: full 256x256 image (left) and the identical texture crop (right). "
        "Same [0,1] display range.",
    )
    grid = fig.add_gridspec(
        3, 2, left=0.045, right=0.96, top=0.800, bottom=0.105, hspace=0.38, wspace=0.17
    )
    for i, (method, short) in enumerate(METHODS):
        for j, (rule, rulelabel) in enumerate(VARS):
            sub = grid[i, j].subgridspec(1, 2, wspace=0.06)
            im = images[f"{short}_{rule}_n{n}"]
            ax = fig.add_subplot(sub[0, 0])
            zoom = fig.add_subplot(sub[0, 1])
            ax.imshow(im, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            zoom.imshow(
                im[40:185, 65:205],
                cmap="magma",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            for a in (ax, zoom):
                a.set_xticks([])
                a.set_yticks([])
            ax.set_title(title_for(method, rule, n), size=9.8, pad=8)
            zoom.set_title("Crop: rows 40-184, cols 65-204", size=9.3, pad=8)
    fig.text(
        0.50,
        0.069,
        "Ground-truth PSNR/SSIM/HaarPSI use raw, unclipped arrays. "
        "The same stored measurement and weights are used throughout.",
        ha="center",
        size=10.2,
    )
    foot(
        fig,
        "The ground-truth image and 8,192-iteration comparison appear in the first six report pages.",
    )
    pdf.savefig(fig, dpi=155)
    plt.close(fig)


def spatial_error_page(pdf, n, page):
    fig = plt.figure(figsize=(16.2, 12.1))
    header(
        fig,
        page,
        f"Signed image error at {n:,} iterations",
        "Each image shows x(reconstruction) - x(ground truth). Shared symmetric colour limits "
        "across ALL methods and iteration checkpoints.",
    )
    grid = fig.add_gridspec(
        3, 2, left=0.09, right=0.92, top=0.858, bottom=0.128, hspace=0.54, wspace=0.23
    )
    for i, (method, short) in enumerate(METHODS):
        for j, (rule, rlabel) in enumerate(VARS):
            r = record[method, rule, n]
            arr = images[f"{short}_{rule}_n{n}_signed_error"]
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(
                arr,
                cmap="coolwarm",
                vmin=-error_cap,
                vmax=error_cap,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{method} | {rlabel}\n"
                f"PSNR {f(r,'psnr',3)} dB | image RMSE {float(r['mse'])**0.5:.5f}",
                size=11,
                pad=7,
            )
    colorbar(fig, error_cap, "Signed image error (reconstruction - GT)")
    fig.text(
        0.5,
        0.104,
        f"Colour limits: +/-{error_cap:.4f} (99.5th percentile pooled "
        "across all checkpoint image errors). Extreme values clip for readability.",
        ha="center",
        size=9.4,
    )
    foot(
        fig,
        "This is a spatial image-error map, not the measurement-domain data residual.",
    )
    pdf.savefig(fig, dpi=155)
    plt.close(fig)


def measured_residual_page(pdf, n, page):
    fig = plt.figure(figsize=(16.2, 12.1))
    header(
        fig,
        page,
        f"Measurement residuals at {n:,} iterations",
        "Residual Ax-y at sampled Hadamard-pattern indices. Unmeasured indices are grey. "
        "This is NOT a spatial image.",
    )
    grid = fig.add_gridspec(
        3, 2, left=0.08, right=0.92, top=0.858, bottom=0.125, hspace=0.54, wspace=0.22
    )
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad("#d3d8df")
    for i, (method, short) in enumerate(METHODS):
        for j, (rule, rlabel) in enumerate(VARS):
            r = record[method, rule, n]
            residual = images[f"{short}_{rule}_n{n}_measurement_residual"]
            assert len(residual) == len(indices)
            grid_coeff = np.full(256 * 256, np.nan, dtype=np.float32)
            grid_coeff[indices] = residual
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(
                grid_coeff.reshape(256, 256),
                cmap=cmap,
                vmin=-measurement_cap,
                vmax=measurement_cap,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{method} | {rlabel}\n"
                f"Data residual RMSE {f(r,'measurement_residual_rmse',5)} | "
                f"fidelity {f(r,'data_fidelity',3)}",
                size=10.7,
                pad=7,
            )
    colorbar(fig, measurement_cap, "Normalized Hadamard measurement residual (Ax-y)")
    fig.text(
        0.5,
        0.104,
        f"Shared limits +/-{measurement_cap:.4f} across checkpoints. "
        "Hadamard coefficient indexing does not preserve spatial pixel positions.",
        ha="center",
        size=9.4,
    )
    foot(
        fig,
        "Only the 13,107 measured coefficients are coloured; 80% unsampled locations remain grey.",
    )
    pdf.savefig(fig, dpi=155)
    plt.close(fig)


def metrics_page(pdf, page=7):
    fig = plt.figure(figsize=(17.8, 12.3))
    header(
        fig,
        page,
        "Checkpoint metrics and acquisition protocol",
        "Perovskite 0292 | frozen 20% NPL-default PCM acquisition | fixed scalar regularisation weights",
    )
    fig.text(
        0.053,
        0.864,
        "18 independently restarted runs = 3 methods x 2 relaxation rules x "
        "3 iteration budgets. Each row is verified against the archived reconstruction metrics.",
        size=11.2,
    )
    table = []
    for n in STEPS:
        for method, _ in METHODS:
            for rule, name in VARS:
                r = record[method, rule, n]
                table.append(
                    [
                        f"{n:,}",
                        method,
                        "LION" if rule == "LION_existing" else "CP",
                        f(r, "psnr", 4),
                        f(r, "ssim", 4),
                        f(r, "haarpsi", 4),
                        f(r, "objective", 3),
                        f(r, "measurement_residual_rmse", 5),
                        f(r, "seconds", 3),
                    ]
                )
    ax = fig.add_axes([0.045, 0.27, 0.91, 0.64])
    ax.axis("off")
    t = ax.table(
        cellText=table,
        colLabels=[
            "Steps",
            "Method",
            "Rule",
            "PSNR (dB)",
            "SSIM",
            "HaarPSI",
            "Objective",
            "Data RMSE",
            "Elapsed (s)",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.085, 0.125, 0.073, 0.118, 0.103, 0.11, 0.127, 0.145, 0.115],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.69)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#cfdae4")
        cell.set_linewidth(0.5)
        if i == 0:
            cell.set_facecolor("#e4ecf5")
            cell.get_text().set_weight("bold")
        elif i in (1, 7, 13):
            cell.set_facecolor("#edf3fb")
        else:
            cell.set_facecolor("#fafcff" if i % 2 else "white")
    fig.text(
        0.052,
        0.143,
        "Definition: image residual = reconstructed image - GT. "
        "Measurement residual = A(reconstruction) - observed data.",
        size=10.1,
    )
    fig.text(
        0.052,
        0.111,
        "All timings were measured on tuni's RTX 4090; each run starts "
        "from zero. Step sizes tau=sigma=0.18; no weights were retuned.",
        size=10.1,
    )
    foot(
        fig,
        "Do not compare TV, TGV-2 and TGV-3 objective values with each other: they are different functionals.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def conclusions_page(pdf, page=17):
    fig = plt.figure(figsize=(15.5, 11))
    header(
        fig,
        page,
        "Runtime and convergence implications",
        "Observed checkpoint timings on one GPU, with both relaxation rules and fixed regularisation parameters",
    )
    grid = fig.add_gridspec(
        2, 3, left=0.075, right=0.955, bottom=0.275, top=0.83, hspace=0.43, wspace=0.27
    )
    for j, (method, _) in enumerate(METHODS):
        for i, (metric, label) in enumerate(
            [("objective", "Objective"), ("psnr", "PSNR (dB)")]
        ):
            ax = fig.add_subplot(grid[i, j])
            for v, desc, mark in [(VARS[0][0], "LION", "o-"), (VARS[1][0], "CP", "s-")]:
                rr = [record[method, v, n] for n in STEPS]
                ax.plot(
                    [float(r["seconds"]) for r in rr],
                    [float(r[metric]) for r in rr],
                    mark,
                    label=desc,
                )
            ax.set_title(f"{method} | {label}", size=11.8)
            ax.set_xlabel("Elapsed seconds")
            ax.set_ylabel(label)
            if metric == "psnr":
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.grid(alpha=0.22)
            if i == 0:
                ax.legend(fontsize=8.6)
    stats = []
    for method, _ in METHODS:
        rL = record[method, "LION_existing", 1024]
        rC = record[method, "canonical_CP", 1024]
        stats.append(
            (
                method,
                round(float(rL["ms_per_iteration"]), 3),
                round(float(rC["ms_per_iteration"]), 3),
                round(float(rC["objective"]) - float(rL["objective"]), 5),
                round(float(rC["psnr"]) - float(rL["psnr"]), 6),
            )
        )
    fig.text(
        0.075,
        0.219,
        "At 1,024 iterations (canonical CP minus existing LION):",
        size=12.3,
        weight="bold",
    )
    for idx, (method, a, b, obj, psnr) in enumerate(stats):
        fig.text(
            0.080,
            0.183 - idx * 0.032,
            f"{method:6s}    LION {a:.3f} ms/step     CP {b:.3f} ms/step"
            f"    objective change {obj:+.5f}    PSNR change {psnr:+.6f} dB",
            fontsize=10.6,
            family="DejaVu Sans Mono",
        )
    fig.text(
        0.075,
        0.051,
        "Canonical CP changes the convergence trajectory rather than the cost of a PDHG iteration. "
        "Objective thresholds and image quality must be assessed separately.",
        size=10.2,
    )
    foot(
        fig,
        "Single GT-tuned case. Runtime is noisy for very short runs; compare at matched iteration budgets.",
    )
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(
    SUP,
    metadata={
        "Title": "Perovskite 0292 - relaxation checkpoint supplement",
        "Author": "PCM experiments",
    },
) as book:
    metrics_page(book, 7)
    page = 8
    for n in STEPS:
        visual_page(book, n, page)
        page += 1
        spatial_error_page(book, n, page)
        page += 1
        measured_residual_page(book, n, page)
        page += 1
    conclusions_page(book, page)
assert SUP.stat().st_size > 100000
subprocess.run(["pdfunite", str(OLD), str(SUP), str(FINAL)], check=True)
check = subprocess.run(
    ["pdfinfo", str(FINAL)], capture_output=True, text=True, check=True
).stdout
assert "Pages:           17" in check, check
print(
    "REPORT_DONE",
    str(FINAL),
    "bytes",
    FINAL.stat().st_size,
    "supplement",
    SUP.stat().st_size,
    "signed_error_limit",
    error_cap,
    "measurement_residual_limit",
    measurement_cap,
)
