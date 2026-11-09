"""Matched random-one-per-category PnP test45 experiment analysis.

Ten requested methods. No oracle ADMM label and no expanded GS label.
Read only after both sets of frozen configurations have completed test45.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen

ROOT = OUTPUT / "test45/random_one_per_category_v1"
PREV = OUTPUT.parent / "ssdu_pnp_tuning_20261108_v1"
HIST = PREV / "next_gs_ssdu_sweep_v1/test450/analysis/all_450_10_methods.csv"
METHODS = (
    ("zero_filled", "Zero-filled"),
    ("spgl1_bp", "SPGL1-BP"),
    ("spgl1_bpdn", "SPGL1-BPDN"),
    ("pnp_admm_ssdu5", "PnP-ADMM tuned-5"),
    ("gs_pnp_oracle", "GS-PnP oracle"),
    ("tv", "Scalar TV"),
    ("utv_c60", "UTV-C60"),
    ("gs_pnp_ssdu5", "GS-PnP tuned-5"),
    ("pnp_admm_tune1", "PnP-ADMM Tune-1"),
    ("gs_pnp_tune1", "GS-PnP Tune-1"),
)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def select():
    calib, test, configs = assert_all_frozen()
    x = json.loads((ROOT / "selection.json").read_text())
    assert x["status"] == "frozen_before_test_access"
    assert len(test) == 45 and len(calib) == 25
    assert x["test45_file_sha256"] == digest(OUTPUT / "prepared/test45.json")
    assert x["validation_file_sha256"] == digest(
        OUTPUT / "prepared/validation_calibrations.json"
    )
    assert x["methods_for_comparison"] == [m for m, _ in METHODS]
    assert x["excluded_methods"] == ["pnp_admm_oracle", "gs_pnp_expanded5"]
    assert not x["using_test_metrics_for_selection"]
    assert len(x["selected_calibrations"]) == 2
    for m in ("admm", "gs"):
        for c in CATEGORIES:
            v = x["selected_calibrations"][m][c]
            p = (
                OUTPUT
                / "tuning"
                / m
                / c
                / f'{int(v["calibration_sample_id"]):04d}'
                / "frozen_selection.json"
            )
            assert digest(p) == v["selection_file_sha256"]
            assert json.loads(p.read_text())["winner_config"] == v["frozen_config"]
    assert "BOTH_45_COMPLETE" in (ROOT / "logs/status.txt").read_text()
    return test, x


def merge():
    test, choices = select()
    with HIST.open(newline="") as f:
        prior = {(x["category"], int(x["sample_id"])): x for x in csv.DictReader(f)}
    assert len(prior) == 450
    new = {}
    for case in test:
        key = (case["category"], int(case["sample_id"]))
        original = prior[key]
        assert original["measurement_sha256"] == case["measurement_sha256"]
        assert original["pattern_sha256"] == case["pattern_sha256"]
        row = {
            "category": key[0],
            "sample_id": key[1],
            "measurement_sha256": original["measurement_sha256"],
            "pattern_sha256": original["pattern_sha256"],
        }
        for m, name in METHODS:
            if m.endswith("_tune1"):
                code = "admm" if m == "pnp_admm_tune1" else "gs"
                cal = choices["selected_calibrations"][code][key[0]]
                cid = cal["calibration_sample_id"]
                record = (
                    ROOT
                    / "recon"
                    / code
                    / key[0]
                    / "records"
                    / f"{cid:04d}"
                    / f"{key[1]:04d}.json"
                )
                raw = (
                    ROOT
                    / "recon"
                    / code
                    / key[0]
                    / "raw"
                    / f"{cid:04d}"
                    / f"{key[1]:04d}.npy"
                )
                assert record.exists() and raw.exists(), (record, raw)
                x = json.loads(record.read_text())
                assert x["test_measurement_sha256"] == row["measurement_sha256"]
                assert x["test_pattern_sha256"] == row["pattern_sha256"]
                assert x["frozen_selection_sha256"] == cal["selection_file_sha256"]
                assert x["no_test_driven_parameter_tuning"] is True
                assert x["calibration_id"] == cid and x["test_sample_id"] == key[1]
                img = np.load(raw, mmap_mode="r", allow_pickle=False)
                assert img.shape == (256, 256) and np.isfinite(img).all()
                values = x["metrics"]
                row[m + "_elapsed_seconds"] = float(x["wall_seconds"])
                row[m + "_calibration_id"] = cid
            else:
                values = {s: float(original[f"{m}_{s}"]) for s in METRICS}
            for metric in METRICS:
                value = float(values[metric])
                assert math.isfinite(value)
                row[f"{m}_{metric}"] = value
        assert key not in new
        new[key] = row
    assert len(new) == 45
    return new, choices


def stats(data):
    result = {}
    for cat in ("ALL",) + CATEGORIES:
        items = [x for (c, _), x in data.items() if cat == "ALL" or c == cat]
        assert len(items) == (45 if cat == "ALL" else 9)
        result[cat] = {}
        for m, name in METHODS:
            result[cat][m] = {}
            for metric in METRICS:
                arr = np.array(
                    [float(x[f"{m}_{metric}"]) for x in items], dtype=np.float64
                )
                result[cat][m][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)),
                    "median": float(np.median(arr)),
                    "n": len(arr),
                }
    paired = {}
    for new, prior in (
        ("pnp_admm_tune1", "pnp_admm_ssdu5"),
        ("gs_pnp_tune1", "gs_pnp_ssdu5"),
        ("gs_pnp_tune1", "gs_pnp_oracle"),
    ):
        diffs = {}
        for metric in METRICS:
            arr = np.array(
                [
                    float(x[f"{new}_{metric}"]) - float(x[f"{prior}_{metric}"])
                    for x in data.values()
                ]
            )
            diffs[metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)),
                "median": float(np.median(arr)),
                "fraction_positive": float((arr > 0).mean()),
            }
        paired[new + "_minus_" + prior] = diffs
    return result, paired


def write_outputs(data, choices, summary, paired):
    out = ROOT / "analysis"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    keys = list(next(iter(data.values())))
    with (out / "test45_10_methods_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for x in sorted(data):
            w.writerow(data[x])
    for name, v in (
        ("test45_10_methods_summary.json", summary),
        ("paired_method_differences.json", paired),
        ("random_calibration_choices.json", choices["selected_calibrations"]),
    ):
        (out / name).write_text(
            json.dumps(v, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = [t for m, t in METHODS]

    def plot_category(cat, pdf):
        fig, axs = plt.subplots(2, 2, figsize=(17.5, 10.4), layout="constrained")
        for ax, (metric, nice) in zip(
            axs.flat,
            (
                ("psnr", "PSNR (dB)"),
                ("ssim", "SSIM"),
                ("haarpsi", "HaarPSI"),
                ("mse", "MSE"),
            ),
        ):
            y = [summary[cat][m][metric]["mean"] for m, t in METHODS]
            e = [summary[cat][m][metric]["std"] for m, t in METHODS]
            ax.bar(np.arange(10), y, yerr=e, capsize=3, alpha=0.80)
            ax.set_xticks(np.arange(10), labels, rotation=38, ha="right", fontsize=8)
            ax.set_title(nice, weight="bold")
            ax.grid(axis="y", alpha=0.24)
            if metric == "mse":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        fig.suptitle(
            f"{cat if cat!='ALL' else 'All 45 cases'} | 20% PCM NPL-noisy | mean +/- sample SD",
            fontsize=14,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(figs / "All_45_10_Method_Mean_SD.pdf") as pdf:
        plot_category("ALL", pdf)
    with PdfPages(figs / "Five_Categories_10_Method_Mean_SD.pdf") as pdf:
        for c in CATEGORIES:
            plot_category(c, pdf)
    fig, axs = plt.subplots(2, 2, figsize=(17.5, 10.4), layout="constrained")
    for ax, (metric, nice) in zip(
        axs.flat,
        (
            ("psnr", "PSNR (dB)"),
            ("ssim", "SSIM"),
            ("haarpsi", "HaarPSI"),
            ("mse", "MSE"),
        ),
    ):
        values = [
            np.array([x[f"{m}_{metric}"] for x in data.values()]) for m, n in METHODS
        ]
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_xticks(np.arange(1, 11), labels, rotation=38, ha="right", fontsize=8)
        ax.set_title(nice, weight="bold")
        ax.grid(axis="y", alpha=0.24)
        if metric == "mse":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle("Matched test45 | all ten selected method labels", fontsize=15)
    fig.savefig(figs / "All_45_10_Method_Boxplots.pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        "SUMMARY_COMPLETE",
        {
            m: {
                metric: round(summary["ALL"][m][metric]["mean"], 4)
                for metric in ("psnr", "ssim", "haarpsi")
            }
            for m in (
                "pnp_admm_tune1",
                "gs_pnp_tune1",
                "pnp_admm_ssdu5",
                "gs_pnp_ssdu5",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    values, selection = merge()
    summary, paired = stats(values)
    write_outputs(values, selection, summary, paired)
