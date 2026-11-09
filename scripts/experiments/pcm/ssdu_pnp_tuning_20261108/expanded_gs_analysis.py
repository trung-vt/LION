"""Analyse the frozen expanded GS-PnP 450-image retrospective comparison.

Metric aggregation, per-category plots and five replayed qualitative cases.
All hyperparameter selection MUST already be frozen; this module cannot tune.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import ROOT, REPORT
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs_test450 import (
    TEST,
    SELECTION,
    BENCH,
    validate,
    sha_file,
)

PARENT = ROOT.parent
OLD = PARENT / "test450/v1"
VIS_OLD = OLD / "visual5"
OUT = TEST / "analysis"
FIGS = OUT / "figures"
METRICS = ("psnr", "ssim", "haarpsi", "mse")
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
# Nine prior labels (seven reference baselines + two original SSDU-5 methods),
# plus the newly frozen GS-PnP-expanded-5 configuration.
METHODS = (
    ("zero_filled", "Zero-filled"),
    ("spgl1_bp", "SPGL1-BP"),
    ("spgl1_bpdn", "SPGL1-BPDN"),
    ("pnp_admm_oracle", "PnP-ADMM oracle"),
    ("gs_pnp_oracle", "GS-PnP oracle"),
    ("tv", "TV"),
    ("utv_c60", "UTV-C60"),
    ("pnp_admm_ssdu5", "PnP-ADMM tuned-5"),
    ("gs_pnp_ssdu5", "GS-PnP tuned-5"),
    ("gs_pnp_expanded5", "GS-PnP expanded-5"),
)
assert len(METHODS) == 10


def inputs():
    validate()
    selected = json.loads(SELECTION.read_text())
    cfg = selected["winner"]["parameters"]
    expected_sha = sha_file(SELECTION)
    cases = sorted((TEST / "records").glob("*/*.json"))
    assert len(cases) == 450, (len(cases), "Missing reconstructed case records")
    new = {}
    for path in cases:
        row = json.loads(path.read_text())
        key = (row["category"], int(row["sample_id"]))
        assert row["selection_file_sha256"] == expected_sha
        assert row["selected_candidate_id"] == selected["winner"]["candidate_id"]
        assert row["no_test_tuning"] is True
        assert key not in new
        assert all(math.isfinite(float(row["metrics"][m])) for m in METRICS)
        image = TEST / "raw" / f"{key[0]}_{key[1]:04d}.npy"
        assert image.exists()
        assert image.stat().st_size > 100000
        new[key] = row
    assert Counter(c for c, _ in new) == Counter({c: 90 for c in CATEGORIES})
    old_file = OLD / "merged/all_450_11_method_metrics.csv"
    with old_file.open(newline="") as f:
        archived = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    assert len(archived) == 450 and set(archived) == set(new)
    merged = {}
    for key in sorted(new):
        row = archived[key]
        current = new[key]
        assert current["measurement_sha256"] == row["measurement_sha256"]
        assert current["pattern_sha256"] == row["pattern_sha256"]
        item = {
            "category": key[0],
            "sample_id": key[1],
            "measurement_sha256": row["measurement_sha256"],
            "pattern_sha256": row["pattern_sha256"],
        }
        for method, label in METHODS:
            vals = (
                current["metrics"]
                if method == "gs_pnp_expanded5"
                else {m: float(row[f"{method}_{m}"]) for m in METRICS}
            )
            for m in METRICS:
                item[f"{method}_{m}"] = float(vals[m])
        merged[key] = item
    assert len(merged) == 450
    return selected, merged, new


def stats(merged, new):
    all_stats = {}
    for cat in ("ALL",) + CATEGORIES:
        rows = [r for (c, _), r in merged.items() if cat == "ALL" or cat == c]
        assert len(rows) == (450 if cat == "ALL" else 90)
        all_stats[cat] = {}
        for method, label in METHODS:
            values = {}
            for metric in METRICS:
                arr = np.array(
                    [x[f"{method}_{metric}"] for x in rows], dtype=np.float64
                )
                assert np.isfinite(arr).all()
                values[metric] = {
                    "mean": float(arr.mean()),
                    "sd": float(arr.std(ddof=1)),
                    "median": float(np.median(arr)),
                    "minimum": float(arr.min()),
                    "maximum": float(arr.max()),
                }
            all_stats[cat][method] = values
    paired = {}
    for other in ("gs_pnp_ssdu5", "gs_pnp_oracle", "utv_c60", "pnp_admm_ssdu5"):
        diffs = {}
        for metric in METRICS:
            delta = np.array(
                [
                    r[f"gs_pnp_expanded5_{metric}"] - r[f"{other}_{metric}"]
                    for r in merged.values()
                ]
            )
            diffs[metric] = {
                "mean": float(delta.mean()),
                "sd": float(delta.std(ddof=1)),
                "fraction_greater": float(np.mean(delta > 0)),
                "fraction_equal": float(np.mean(delta == 0)),
            }
        paired["expanded_minus_" + other] = diffs
    durations = np.array(
        [x["diagnostics"]["wall_seconds_with_final_gradient"] for x in new.values()],
        dtype=np.float64,
    )
    timings = {
        "expanded_gs": {
            "n": len(durations),
            "mean_seconds": float(durations.mean()),
            "sd_seconds": float(durations.std(ddof=1)),
            "median_seconds": float(np.median(durations)),
        }
    }
    return all_stats, paired, timings


def write_tables(merged, all_stats, paired, timings):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "all_450_10_methods.csv"
    with out.open("w", newline="") as f:
        fields = list(next(iter(merged.values())))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(merged):
            w.writerow(merged[key])
    for name, body in (
        ("metric_summary.json", all_stats),
        ("paired_differences.json", paired),
        ("runtime_summary.json", timings),
    ):
        (OUT / name).write_text(
            json.dumps(body, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
    return out


def plot_stats(merged, all_stats):
    FIGS.mkdir(parents=True, exist_ok=True)
    labels = [label for m, label in METHODS]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}
    )
    fig, axs = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
    for ax, (met, name) in zip(
        axs.flat,
        (
            ("psnr", "PSNR (dB)"),
            ("ssim", "SSIM"),
            ("haarpsi", "HaarPSI"),
            ("mse", "MSE"),
        ),
    ):
        vals = [
            np.array([r[f"{m}_{met}"] for r in merged.values()]) for m, _ in METHODS
        ]
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_title(name, fontsize=13, weight="semibold")
        ax.set_xticks(range(1, 11), labels, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        if met == "mse":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle("450 matched 20% NPL-noisy PCM cases | 10 methods", fontsize=16)
    fig.savefig(FIGS / "10_methods_450_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)

    def meanpage(cat, pdf):
        fig, axs = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
        for ax, (met, name) in zip(
            axs.flat,
            (
                ("psnr", "PSNR (dB)"),
                ("ssim", "SSIM"),
                ("haarpsi", "HaarPSI"),
                ("mse", "MSE"),
            ),
        ):
            means = [all_stats[cat][m][met]["mean"] for m, _ in METHODS]
            errors = [all_stats[cat][m][met]["sd"] for m, _ in METHODS]
            ax.bar(range(10), means, yerr=errors, capsize=3, alpha=0.85)
            ax.set_title(name, weight="semibold")
            ax.set_xticks(range(10), labels, rotation=40, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if met == "mse":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        fig.suptitle(
            (cat + ": 90 cases" if cat != "ALL" else "All 450 cases")
            + " | Mean and sample SD",
            fontsize=16,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(FIGS / "10_methods_450_mean_sd.pdf") as pdf:
        meanpage("ALL", pdf)
    with PdfPages(FIGS / "10_methods_categories_mean_sd.pdf") as pdf:
        for cat in CATEGORIES:
            meanpage(cat, pdf)
    print("STATS_PLOTS_COMPLETE", flush=True)


def gallery(merged):
    selection = json.loads((VIS_OLD / "selection.json").read_text())
    cases = [(x["category"], int(x["sample_id"])) for x in selection["chosen_cases"]]
    assert cases == [
        ("CIGS", 279),
        ("Organic", 331),
        ("Perovskite", 786),
        ("Reference", 279),
        ("Silicone", 370),
    ]
    cmap = make_colormap("afmhot", 0.8)
    target = FIGS / "Five_Examples_10_Methods_with_Metrics.pdf"
    with PdfPages(target) as pdf:
        for i, (category, sid) in enumerate(cases, 1):
            fp = VIS_OLD / "arrays" / f"{category}_{sid:04d}.npz"
            with np.load(fp, allow_pickle=False) as f:
                archive = {n: f[n].copy() for n in f.files}
            archive["gs_pnp_expanded5"] = np.load(
                TEST / "raw" / f"{category}_{sid:04d}.npy", allow_pickle=False
            )
            data = merged[(category, sid)]
            # 11 image panels: target, ten method labels, one unused grid cell.
            display = [("target", "Target")] + list(METHODS)
            fig, axs = plt.subplots(3, 4, figsize=(19, 13.1), layout="constrained")
            fig.suptitle(
                f"{category} {sid:04d} | 20% NPL noisy PCM | matched acquisition",
                fontsize=16,
                weight="bold",
            )
            for j, ax in enumerate(axs.flat):
                if j >= len(display):
                    ax.axis("off")
                    continue
                k, label = display[j]
                arr = archive[k]
                assert arr.shape == (256, 256) and np.isfinite(arr).all()
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if k == "target":
                    info = "Ground truth"
                else:
                    info = (
                        f'PSNR {data[f"{k}_psnr"]:.2f} dB  SSIM {data[f"{k}_ssim"]:.4f}\n'
                        f'HaarPSI {data[f"{k}_haarpsi"]:.4f}  '
                        f'MSE {data[f"{k}_mse"]:.2e}'
                    )
                ax.set_title(label + "\n" + info, fontsize=9.8)
            fig.text(
                0.5,
                0.011,
                f"{i}/5, same [0,1] image display and AFM-hot cap 0.8.",
                ha="center",
                fontsize=8.5,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("GALLERY_CASE", i, category, sid, flush=True)
    print("GALLERY_PDF", target, target.stat().st_size, flush=True)


def main():
    selected, merged, new = inputs()
    all_stats, paired, timings = stats(merged, new)
    write_tables(merged, all_stats, paired, timings)
    plot_stats(merged, all_stats)
    gallery(merged)
    frozen = {
        "selection_id": selected["winner"]["candidate_id"],
        "selected_params": selected["winner"]["parameters"],
        "metric_summary": str(OUT / "metric_summary.json"),
        "test_cases": 450,
        "categories": list(CATEGORIES),
        "no_test_tuning": True,
        "paired": paired,
        "new_reconstruction_count": len(new),
        "source_selection_sha256": sha_file(SELECTION),
    }
    (OUT / "analysis_complete.json").write_text(
        json.dumps(frozen, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    print(
        "ANALYSIS_COMPLETE",
        {
            "expanded_psnr": all_stats["ALL"]["gs_pnp_expanded5"]["psnr"]["mean"],
            "original_gs_psnr": all_stats["ALL"]["gs_pnp_ssdu5"]["psnr"]["mean"],
            "oracle_gs_psnr": all_stats["ALL"]["gs_pnp_oracle"]["psnr"]["mean"],
            "paired_mean_psnr_vs_original": paired["expanded_minus_gs_pnp_ssdu5"][
                "psnr"
            ]["mean"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
