"""Build PDF + CSV report from frozen U-TV checkpoint ratio-sweep arrays."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from LION.config.LogConfig import ResultOutputConfig
from LION.config.config_utils import load_yaml, make_dataclass
from LION.pcm.pcm_result_writer import PCMResultRenderer
from LION.visualize.plotting_functions import make_colormap

ROOT = Path("/store/LION/ttv22/pcm/results/20261025_utv_ratio_sweep_snapshot")
DEFAULT_RENDER_CONFIG = (
    Path(__file__).resolve().parents[4] / "config/pcm/pcm_utv_ratio_sweep_render.yaml"
)
RATIOS = (5, 10, 20, 30, 40, 50)
METRICS = ("mse", "psnr", "ssim", "haarpsi")
PALETTE = plt.get_cmap("tab10")


def load_render_config(path: Path) -> ResultOutputConfig:
    """Parse the standard LION log.final_result plot specification."""
    config = make_dataclass(ResultOutputConfig, load_yaml(path)["log"]["final_result"])
    spec = config.reconstruction
    if spec is None or spec.vmin is None or spec.vmax is None:
        raise ValueError(
            "Specify shared reconstruction vmin and vmax in the render YAML."
        )
    if not np.isfinite((spec.vmin, spec.vmax)).all() or spec.vmin >= spec.vmax:
        raise ValueError("Require finite reconstruction vmin < vmax.")
    if spec.cmap is None or "png" not in config.image_formats:
        raise ValueError("Set reconstruction.cmap and include png in image_formats.")
    return config


def rows_read():
    with (ROOT / "metrics_per_image.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300, len(rows)
    for row in rows:
        for field in ("sampling_percent", "sample_id"):
            row[field] = int(row[field])
        for field in METRICS:
            row[field] = float(row[field])
    assert (
        len(set((r["category"], r["sample_id"], r["sampling_percent"]) for r in rows))
        == 300
    )
    return rows


def aggregate(rows, output_dir):
    groups = defaultdict(list)
    for r in rows:
        groups[("ALL", r["sampling_percent"])].append(r)
        groups[(r["category"], r["sampling_percent"])].append(r)
    out = []
    for (category, ratio), group in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        item = {"category": category, "sampling_percent": ratio, "n_images": len(group)}
        for metric in METRICS:
            arr = np.asarray([r[metric] for r in group])
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_sd"] = float(arr.std(ddof=1))
        out.append(item)
    with (output_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return out


def formatted_summary(data, category):
    return [
        next(
            x for x in data if x["category"] == category and x["sampling_percent"] == r
        )
        for r in RATIOS
    ]


def title_page(pdf, summary, manifest):
    d = formatted_summary(summary, "ALL")
    fig = plt.figure(figsize=(16.54, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(
        0.055,
        0.92,
        "NOISY U-TV  |  5%-50% sampling",
        fontsize=27,
        weight="bold",
        va="top",
    )
    fig.text(
        0.055,
        0.865,
        "Held-out PCM evaluation • checkpoint from epoch 5, step 22,500",
        fontsize=15,
    )
    notes = [
        "Model: 4-level U-Net, 128 initial filters, 512 PDHG iterations",
        "Evaluation GPU: RTX 4070 (training continues separately on RTX 4090)",
        "Five PCM categories; 10 held-out TEST images per category (50 images total)",
        "Six sample ratios, coarse J=6; 300 reconstructions",
        "Fresh, reproducibly seeded noise draw per image and sampling ratio",
        "Dark / bias / beam sigma sampled from the SAME variable-noise ranges used in training",
        "One noise realisation per image per ratio; results are descriptive, not final test-set estimates",
        "Metrics computed by the existing LION compute_metrics implementation (unchanged)",
        "The learned scale-factor key is shared across all ratios (as in training).",
        "5% falls below the fully sampled J6 block; existing sampler selects a subset from J6.",
    ]
    y = 0.80
    for note in notes:
        fig.text(0.064, y, "•  " + note, fontsize=12.3)
        y -= 0.038
    ax = fig.add_axes([0.055, 0.19, 0.89, 0.26])
    ax.axis("off")
    tab = [
        [
            f"{r['sampling_percent']}%",
            f"{r['mse_mean']:.5f}",
            f"{r['psnr_mean']:.2f}",
            f"{r['ssim_mean']:.3f}",
            f"{r['haarpsi_mean']:.3f}",
        ]
        for r in d
    ]
    table = ax.table(
        cellText=tab,
        colLabels=[
            "Sampling",
            "Mean MSE",
            "Mean PSNR (dB)",
            "Mean SSIM",
            "Mean HaarPSI",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d9e3")
        cell.set_facecolor(
            "#e5edf7" if row == 0 else "#f7f8fa" if row % 2 else "#ffffff"
        )
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.text(
        0.055,
        0.105,
        "Source: snapshot best_val_loss_s0.20_cJ6.pt; saved training checkpoint is NOT accessed during testing.",
        fontsize=10,
        color="#4b5563",
    )
    fig.text(
        0.055,
        0.068,
        "All category averages use equal numbers of samples. See metrics_summary.csv and metrics_per_image.csv.",
        fontsize=10,
        color="#4b5563",
    )
    pdf.savefig(fig)
    plt.close(fig)


def metric_page(pdf, summary):
    fig, axs = plt.subplots(2, 2, figsize=(16.54, 11.69))
    fig.suptitle("Sampling ratio response by category", fontsize=24, fontweight="bold")
    cats = ["ALL", "CIGS", "Organic", "Perovskite", "Reference", "Silicone"]
    for ax, metric in zip(axs.flat, ("psnr", "ssim", "haarpsi", "mse")):
        for k, cat in enumerate(cats):
            dat = formatted_summary(summary, cat)
            vals = [d[f"{metric}_mean"] for d in dat]
            ax.plot(
                RATIOS,
                vals,
                marker="o",
                lw=2.1 if cat == "ALL" else 1.2,
                linestyle="-" if cat == "ALL" else "--",
                color="black" if cat == "ALL" else PALETTE(k - 1),
                label=cat,
                alpha=1 if cat == "ALL" else 0.77,
            )
        ax.set_xlabel("Subsampled Hadamard measurements (%)")
        ax.set_ylabel(
            {"psnr": "PSNR (dB)", "ssim": "SSIM", "haarpsi": "HaarPSI", "mse": "MSE"}[
                metric
            ]
        )
        ax.set_xticks(RATIOS)
        ax.grid(alpha=0.21)
    axs[0, 0].legend(loc="best", fontsize=10)
    fig.subplots_adjust(
        top=0.88, bottom=0.09, left=0.08, right=0.96, wspace=0.18, hspace=0.27
    )
    pdf.savefig(fig)
    plt.close(fig)


def _crop_by_grad(im, side=96):
    im = np.asarray(im)
    best = (-float("inf"), 0, 0)
    for y in range(0, 256 - side + 1, 16):
        for x in range(0, 256 - side + 1, 16):
            a = im[y : y + side, x : x + side]
            score = float(
                np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
            )
            if score > best[0]:
                best = (score, x, y)
    return best[1], best[2], side, side


def image_pages(pdf, rows, manifest, output_dir, render_config):
    spec = render_config.reconstruction
    assert spec is not None and spec.vmin is not None and spec.vmax is not None
    vmin, vmax = float(spec.vmin), float(spec.vmax)
    cmap = make_colormap(spec.cmap, spec.cmap_max)
    renderer = PCMResultRenderer(render_config)
    lookup = {(r["category"], r["sample_id"], r["sampling_percent"]): r for r in rows}
    arrays = sorted((ROOT / "arrays").glob("*.npz"))
    assert len(arrays) in (15, 17), len(arrays)
    figroot = output_dir / "individual_png"
    for file in arrays:
        orig = file.stem
        extra = "_train" in orig or "_val" in orig
        category = orig.split("_image")[0]
        sid = int(orig.split("_image")[1].split("_")[0])
        data = np.load(file)
        assert all(f"ratio_{r:02d}" in data for r in RATIOS), file
        im = data["ground_truth"]
        assert im.shape == (256, 256)
        crops = [(None, False), (_crop_by_grad(im), True)]
        if category == "CIGS" and sid == 674:
            crops = [(None, False), ((78, 82, 97, 97), True)]
        note = (
            "TRAINING SPLIT - qualitative illustration ONLY (excluded from test metrics)"
            if "train" in orig.lower()
            else "VALIDATION SPLIT - qualitative illustration ONLY (excluded from test metrics)"
            if "val" in orig.lower()
            else "HELD-OUT TEST SPLIT"
        )
        dest = figroot / category
        dest.mkdir(exist_ok=True, parents=True)
        for k, img in [
            ("ground_truth", im),
            *[(f"sample_{r:02d}pct", data[f"ratio_{r:02d}"]) for r in RATIOS],
        ]:
            renderer.render_scalar_image(
                values=img,
                out_dir=dest,
                filename_stem=f"{orig}_{k}",
                label=k,
                spec=spec,
            )
        for (crop, iszoom) in crops:
            fig, axs = plt.subplots(2, 4, figsize=(16.54, 11.69))
            mode = (
                "Local detail (automatically selected gradient-rich 96px crop)"
                if iszoom
                else "Full 256x256 field"
            )
            if category == "CIGS" and sid == 674 and iszoom:
                mode = "Thin 1-pixel line detail (manually specified known location)"
            fig.suptitle(
                f"{category} / image{sid:04d}   |   {mode}",
                fontsize=21,
                fontweight="bold",
                y=0.982,
            )
            fig.text(
                0.055,
                0.933,
                note,
                fontsize=11.2,
                color="#8f3f1a" if extra else "#334155",
            )
            images = [("Ground truth", im)]
            images += [
                (f"{ratio}% sampling", data[f"ratio_{ratio:02d}"]) for ratio in RATIOS
            ]
            for i, (label, arr) in enumerate(images):
                ax = axs.flat[i]
                if crop is not None:
                    x, y, w, h = crop
                    arr = arr[y : y + h, x : x + w]
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(label, fontsize=15)
                ax.axis("off")
                if i == 0 and not iszoom:
                    x, y, w, h = _crop_by_grad(im)
                    if category == "CIGS" and sid == 674:
                        x, y, w, h = (78, 82, 97, 97)
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            w,
                            h,
                            fill=False,
                            edgecolor="#28ddfa",
                            lw=2,
                        )
                    )
            ax = axs.flat[-1]
            ax.axis("off")
            stats = [
                f"CHECKPOINT: epoch 5 / step 22500",
                f"Test ratios: {', '.join(map(str,RATIOS))}%",
                f"Display limits: [{vmin:g}, {vmax:g}]",
            ]
            if not extra:
                stats += ["", "PSNR / SSIM / HaarPSI"]
                for ratio in RATIOS:
                    r = lookup[(category, sid, ratio)]
                    stats += [
                        f"{ratio:>2}%   {r['psnr']:>5.2f} dB  {r['ssim']:.3f}   {r['haarpsi']:.3f}"
                    ]
            else:
                stats += ["", "Visual only; not used in", "held-out aggregate metrics."]
            ax.text(
                0.01,
                0.99,
                "\n".join(stats),
                transform=ax.transAxes,
                va="top",
                fontsize=11.6,
                fontfamily="DejaVu Sans Mono",
                linespacing=1.5,
            )
            if iszoom:
                x, y, w, h = crop
                fig.text(
                    0.06,
                    0.073,
                    f"Crop: x={x}:{x+w}, y={y}:{y+h}. All panels use the same coordinates and intensity scale.",
                    fontsize=10,
                    color="#334155",
                )
            else:
                fig.text(
                    0.06,
                    0.073,
                    "Cyan box indicates the crop shown on the following page. No per-image autoscaling is used.",
                    fontsize=10,
                    color="#334155",
                )
            cb = fig.add_axes([0.15, 0.041, 0.70, 0.014])
            norm = plt.Normalize(vmin, vmax)
            bar = matplotlib.colorbar.ColorbarBase(
                cb,
                cmap=cmap,
                norm=norm,
                orientation="horizontal",
            )
            ticks = [vmin, (vmin + vmax) / 2, vmax]
            labels = [format(v, spec.cbar_tick_format) for v in ticks]
            if spec.cbar_first_tick_leq:
                labels[0] = "≤ " + labels[0]
            if spec.cbar_last_tick_geq:
                labels[-1] = "≥ " + labels[-1]
            bar.set_ticks(ticks, labels=labels)
            cb.set_xlabel(
                f"PCM intensity (shared display limits [{vmin:g}, {vmax:g}])",
                fontsize=10,
            )
            fig.subplots_adjust(
                left=0.045, right=0.955, bottom=0.11, top=0.90, wspace=0.06, hspace=0.25
            )
            pdf.savefig(fig)
            plt.close(fig)
        print("RENDERED_PDF_SAMPLE", category, sid, "extra", extra, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-config", type=Path, default=DEFAULT_RENDER_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "rendered_configurable"
    )
    args = parser.parse_args()
    render_config = load_render_config(args.render_config)
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT.resolve():
        raise ValueError("Use a separate plotting output dir to protect source data.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.render_config, output_dir / "render_config_used.yaml")
    rows = rows_read()
    manifest = json.loads((ROOT / "evaluation_manifest.json").read_text())
    summary = aggregate(rows, output_dir)
    document = output_dir / "utv_sampling_5_to_50_percent_report.pdf"
    with PdfPages(document) as pdf:
        title_page(pdf, summary, manifest)
        metric_page(pdf, summary)
        image_pages(pdf, rows, manifest, output_dir, render_config)
        info = pdf.infodict()
        info["Title"] = "Noisy U-TV reconstruction at 5%-50% Hadamard sampling"
        info["Author"] = "LION PCM experiment"
        info[
            "Subject"
        ] = "Checkpoint epoch 5, 10 held-out test images per category and qualitative examples"
    print("CREATED_PDF", document, "bytes", document.stat().st_size, flush=True)
    print("SUMMARY_TABLE")
    for row in formatted_summary(summary, "ALL"):
        print(row, flush=True)


if __name__ == "__main__":
    main()
