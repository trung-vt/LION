"""Visual comparison for the 14 hand-selected bad UTV C60 PCM test cases.

Reuses frozen 20% NPL-default acquisitions. No new measurements or
reconstruction runs are performed. Baseline and reconstruction identities are
verified against the fixed 450-case manifest and numerical metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from LION.visualize.plotting_functions import make_colormap

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUTPUT = ROOT / "bad_images_for_utv_20261109_v1"
CASES = OrderedDict(
    [
        ("Reference", (21, 66, 188, 727)),
        ("CIGS", (710, 727, 805, 821, 873)),
        ("Perovskite", (31, 79, 98, 260, 288)),
    ]
)
ARCHIVE = ROOT / "ten_methods_matched_450_20261105_v1/visual_14_paired_exact_v1/raw"
ORIGINAL = ROOT / "pcm_utv_fixed20_npl_default_20261103/cases"
SSDU = ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1"
METRICS_CSV = SSDU / "merged/all_450_11_method_metrics.csv"
GS_RAW = SSDU / "raw/gs_ssdu"

# The November 8 report establishes that ADMM SSDU-5 and original validation
# tuning selected identical parameters; these numerical arrays are equal.
METHODS = [
    ("gt", "Ground truth", "ground_truth", None),
    ("zero_filled", "Zero-fill", "zero_filled", "zero_filled"),
    ("spgl1_bpdn", "SPGL1-BPDN", "bpdn", "spgl1_bpdn"),
    ("pnp_admm_ssdu5", "PnP-ADMM-tuned-5", "pnp_tuned", "pnp_admm_ssdu5"),
    ("gs_pnp_ssdu5", "GS-PnP-tuned-5", None, "gs_pnp_ssdu5"),
    ("tv", "Scalar TV", "tv", "tv"),
    ("utv_c60", "UTV C60", "C", "utv_c60"),
]


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_metrics() -> dict[tuple[str, int], dict]:
    with METRICS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    table = {(r["category"], int(r["sample_id"])): r for r in rows}
    assert len(rows) == 450 and len(table) == 450
    return table


def read_case(cat: str, sid: int, benchmark: dict):
    archived = ARCHIVE / f"{cat}_{sid:04d}.npz"
    original = ORIGINAL / f"{cat}_{sid:04d}.npz"
    gsfile = GS_RAW / f"{cat}_{sid:04d}.npy"
    assert archived.is_file() and original.is_file() and gsfile.is_file()
    row = benchmark[(cat, sid)]
    with np.load(original, allow_pickle=False) as z:
        measurement_sha = sha(z["measurement"])
        pattern_sha = sha(z["sampled_indices"])
        assert float(z["sampling_ratio"]) == 0.2
    assert measurement_sha == row["measurement_sha256"], (
        cat,
        sid,
        "measurement mismatch",
    )
    assert pattern_sha == row["pattern_sha256"], (cat, sid, "sampling pattern mismatch")
    with np.load(archived, allow_pickle=False) as z:
        images = {
            method: z[arr].copy() for method, _, arr, _ in METHODS if arr is not None
        }
    images["gs_pnp_ssdu5"] = np.load(gsfile, allow_pickle=False)
    gt = images["gt"]
    assert gt.shape == (256, 256)
    metrics = {}
    for key, _, _, metric_key in METHODS:
        im = images[key]
        assert im.shape == (256, 256), (cat, sid, key, im.shape)
        assert np.isfinite(im).all(), (cat, sid, key)
        if metric_key is None:
            continue
        stats = {
            m: float(row[f"{metric_key}_{m}"])
            for m in ("psnr", "ssim", "haarpsi", "mse")
        }
        mse_from_raw = float(
            np.mean((im.astype(np.float64) - gt.astype(np.float64)) ** 2)
        )
        np.testing.assert_allclose(
            mse_from_raw,
            stats["mse"],
            rtol=3e-5,
            atol=1e-8,
            err_msg=f"stored reconstruction does not match metrics: {cat}_{sid:04d} {key}",
        )
        metrics[key] = stats
    return (
        images,
        metrics,
        {
            "category": cat,
            "sample_id": sid,
            "sample_name": f"{cat}_{sid:04d}",
            "measurement_sha256": measurement_sha,
            "pattern_sha256": pattern_sha,
            "archived_arrays_path": str(archived),
            "archived_acquisition_path": str(original),
            "gs_tuned5_path": str(gsfile),
        },
    )


def write_pdf(pages: list):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    dest = OUTPUT / "PCM_Bad_UTV_14_20pct_NPL_7_Methods.pdf"
    with PdfPages(
        dest,
        metadata={
            "Title": "14 bad UTV C60 PCM images: fixed 20 percent NPL 7-method comparison",
            "Author": "PCM evaluation",
            "Subject": "Matched stored acquisitions, identical display scale 0-1",
        },
    ) as pdf:
        for i, (images, metrics, meta) in enumerate(pages, 1):
            fig = plt.figure(figsize=(16.55, 10.2))
            gs = fig.add_gridspec(
                2,
                4,
                left=0.045,
                right=0.955,
                top=0.845,
                bottom=0.125,
                wspace=0.20,
                hspace=0.29,
            )
            name = meta["sample_name"]
            fig.text(
                0.5,
                0.975,
                f"Bad images for UTV | {name}",
                ha="center",
                va="top",
                fontsize=21,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.929,
                "Stored 20% multilevel Hadamard sampling | NPL-default PCM noise"
                " | one matched acquisition per image",
                ha="center",
                va="top",
                fontsize=11.6,
            )
            for pos, (key, label, _, metric_key) in enumerate(METHODS):
                ax = fig.add_subplot(gs[pos // 4, pos % 4])
                im = images[key]
                ax.imshow(
                    im,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    origin="upper",
                    rasterized=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#555555")
                    spine.set_linewidth(0.7)
                if metric_key is None:
                    caption = "Reference image"
                else:
                    s = metrics[key]
                    caption = (
                        f"PSNR {s['psnr']:.2f} dB  |  SSIM {s['ssim']:.3f}"
                        f"\nHaarPSI {s['haarpsi']:.3f}"
                    )
                ax.set_title(
                    f"{label}\n{caption}", fontsize=10, pad=8, fontweight="normal"
                )
                if key == "utv_c60":
                    for spine in ax.spines.values():
                        spine.set_color("#444444")
                        spine.set_linewidth(1.7)
            note_ax = fig.add_subplot(gs[1, 3])
            note_ax.set_axis_off()
            note_ax.text(
                0.055,
                0.79,
                "Comparison notes",
                transform=note_ax.transAxes,
                fontsize=13.0,
                fontweight="bold",
            )
            note_ax.text(
                0.055,
                0.68,
                "Zero-fill, SPGL1-BPDN, PnP-ADMM-\n"
                "tuned-5, GS-PnP-tuned-5, scalar TV\n"
                "and UTV C60 use the same frozen\n"
                "measurement and sampled indices.\n\n"
                "Scalar TV: lambda=0.011, 1024 iterations.\n"
                "UTV: original C60 checkpoint.\n"
                "All metrics from matched 450-image test.",
                transform=note_ax.transAxes,
                fontsize=10.0,
                va="top",
                linespacing=1.35,
                color="#333333",
            )
            cax = fig.add_axes((0.27, 0.061, 0.46, 0.014))
            cb = fig.colorbar(
                ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                cax=cax,
                orientation="horizontal",
            )
            cb.set_label(
                "Reconstruction value (displayed, clipped to [0, 1])", fontsize=10
            )
            cb.ax.tick_params(labelsize=9)
            fig.text(
                0.035,
                0.011,
                f"Acquisition SHA256: {meta['measurement_sha256'][:20]}..."
                f"   |   Pattern SHA256: {meta['pattern_sha256'][:20]}...",
                fontsize=8,
                color="#555555",
                va="bottom",
            )
            fig.text(
                0.970,
                0.011,
                f"{i} / {len(pages)}",
                ha="right",
                fontsize=9.5,
                va="bottom",
            )
            pdf.savefig(fig, dpi=175)
            plt.close(fig)
            print("PDF_PAGE", i, len(pages), name, flush=True)
    return dest


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = load_metrics()
    pages = []
    rows = []
    for cat, ids in CASES.items():
        for sid in ids:
            images, metrics, meta = read_case(cat, sid, table)
            pages.append((images, metrics, meta))
            for k, title, arrkey, metric_key in METHODS:
                if metric_key is not None:
                    rows.append(
                        {
                            "category": cat,
                            "sample_id": f"{sid:04d}",
                            "method": title,
                            **metrics[k],
                            "measurement_sha256": meta["measurement_sha256"],
                            "pattern_sha256": meta["pattern_sha256"],
                        }
                    )
    assert len(pages) == 14 and len(rows) == 14 * 6
    pdf = write_pdf(pages)
    metricsfile = OUTPUT / "PCM_Bad_UTV_14_20pct_NPL_7_Methods_metrics.csv"
    with metricsfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "sample_id",
                "method",
                "psnr",
                "ssim",
                "haarpsi",
                "mse",
                "measurement_sha256",
                "pattern_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "name": "bad images for UTV",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Hand-selected 14 cases with unwanted detail/contrast loss in UTV C60.",
        "sampling_ratio": 0.20,
        "noise_model": "NPL default",
        "measurement_policy": "reuse exact stored acquisitions, no resimulation",
        "methods": [item[1] for item in METHODS],
        "notes": {
            "tuned5_source": "SSDU PnP tuning 2026-11-08 frozen 5-sample selections",
            "admm_tuned5_note": "SSDU-5 selected same rho,sigma as original validation-tuned ADMM; arrays reused, no test oracle applied",
            "gs_tuned5_note": "Use GS SSDU-5 reconstructions (not older GS oracle and not expanded GS)",
            "tv": "lambda=0.011; 1024 iterations",
            "utv": "original C60, no multiplier sweep",
            "display": "AFM hot colour map with vmin=0,vmax=1; all metrics from unclipped outputs",
        },
        "cases": [m for _, _, m in pages],
        "source_metrics": str(METRICS_CSV),
        "source_visual_archive": str(ARCHIVE),
    }
    manifestfile = OUTPUT / "bad_images_for_utv_dataset_manifest.json"
    manifestfile.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("OUTPUT_PDF", pdf)
    print("OUTPUT_CSV", metricsfile)
    print("OUTPUT_MANIFEST", manifestfile)
    print("PDF_BYTES", pdf.stat().st_size)


if __name__ == "__main__":
    main()
