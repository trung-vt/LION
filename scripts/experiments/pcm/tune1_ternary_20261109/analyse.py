"""Aggregate tune-1 test45 without pseudoreplicating calibration choices.

Every calibration source defines a separate tune-1 algorithm configuration,
each applied to the same nine test images from its category. Overall means
average the five calibrations for each test image before averaging test45.
No test scores are used to choose calibration hyperparameters.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    BASE,
    CATEGORIES,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen

PARENT = BASE / "ssdu_pnp_tuning_20261108_v1"
HIST = PARENT / "next_gs_ssdu_sweep_v1/test450/analysis/all_450_10_methods.csv"
METRICS = ("psnr", "ssim", "haarpsi", "mse")
BASELINES = (
    ("zero_filled", "Zero-filled"),
    ("spgl1_bp", "SPGL1-BP"),
    ("spgl1_bpdn", "SPGL1-BPDN"),
    ("pnp_admm_oracle", "PnP-ADMM validation oracle"),
    ("gs_pnp_oracle", "GS-PnP validation oracle"),
    ("tv", "TV (validation tuned)"),
    ("utv_c60", "UTV-C60 original"),
    ("pnp_admm_ssdu5", "PnP-ADMM SSDU-5"),
    ("gs_pnp_ssdu5", "GS-PnP SSDU-5"),
    ("gs_pnp_expanded5", "GS-PnP expanded SSDU-5"),
)
ALL_KEYS = [x[0] for x in BASELINES] + ["pnp_admm_tune1", "gs_pnp_tune1"]
LABELS = {k: v for k, v in BASELINES}
LABELS["pnp_admm_tune1"] = "PnP-ADMM tune-1"
LABELS["gs_pnp_tune1"] = "GS-PnP tune-1"
OUT = OUTPUT / "analysis"


def collect():
    calibrations, test, configs = assert_all_frozen()
    assert HIST.exists()
    with HIST.open(newline="") as f:
        history = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    assert len(history) == 450
    cases = {}
    per_calibration = []
    per_pair = []
    for row in test:
        cat = row["category"]
        sid = int(row["sample_id"])
        prior = history[cat, sid]
        assert prior["measurement_sha256"] == row["measurement_sha256"]
        assert prior["pattern_sha256"] == row["pattern_sha256"]
        measures = {}
        for key, label in BASELINES:
            measures[key] = {m: float(prior[f"{key}_{m}"]) for m in METRICS}
        all_sample_calib = [
            int(x["sample_id"]) for x in calibrations if x["category"] == cat
        ]
        assert len(all_sample_calib) == 5
        for method in ("admm", "gs"):
            vectors = []
            for cal_id in all_sample_calib:
                p = (
                    OUTPUT
                    / "test45"
                    / method
                    / cat
                    / "records"
                    / f"{cal_id:04d}"
                    / f"{sid:04d}.json"
                )
                assert p.exists(), ("test45 incomplete", p)
                record = json.loads(p.read_text())
                assert record["no_test_driven_parameter_tuning"] is True
                assert record["test_measurement_sha256"] == row["measurement_sha256"]
                assert record["test_pattern_sha256"] == row["pattern_sha256"]
                selected_file = (
                    OUTPUT
                    / "tuning"
                    / method
                    / cat
                    / f"{cal_id:04d}"
                    / "frozen_selection.json"
                )
                assert record["frozen_selection_sha256"] == digest(selected_file)
                values = {m: float(record["metrics"][m]) for m in METRICS}
                vectors.append(values)
                per_pair.append(
                    {
                        "category": cat,
                        "test_sample_id": sid,
                        "calibration_sample_id": cal_id,
                        "method": method,
                        **values,
                    }
                )
            # This is the expected score over 5 equally likely frozen calibration
            # choices, NOT image ensembling or choosing the best test score.
            name = "pnp_admm_tune1" if method == "admm" else "gs_pnp_tune1"
            measures[name] = {
                m: float(np.mean([q[m] for q in vectors])) for m in METRICS
            }
        cases[cat, sid] = {"category": cat, "sample_id": sid, "metrics": measures}
    assert len(cases) == 45
    assert all(set(v["metrics"]) == set(ALL_KEYS) for v in cases.values())

    cal_stats = {}
    for cat in CATEGORIES:
        for method in ("admm", "gs"):
            for record in [r for r in calibrations if r["category"] == cat]:
                cal_id = int(record["sample_id"])
                subset = [
                    v
                    for v in per_pair
                    if (v["category"], v["method"], v["calibration_sample_id"])
                    == (cat, method, cal_id)
                ]
                assert len(subset) == 9
                selection = configs[method, cat, cal_id]
                cal_stats[f"{method}/{cat}/{cal_id:04d}"] = {
                    "calibration_sample_id": cal_id,
                    "category": cat,
                    "method": method,
                    "frozen_parameters": selection["winner_params"],
                    "calibration_ssdu_mse": selection["winner_ssdu_mse"],
                    "evaluation_samples": 9,
                    "test_metrics": {
                        m: {
                            "mean": float(np.mean([x[m] for x in subset])),
                            "sd": float(np.std([x[m] for x in subset], ddof=1)),
                        }
                        for m in METRICS
                    },
                }
    assert len(cal_stats) == 50
    summaries = {}
    for cat in ("ALL",) + CATEGORIES:
        records = [v for (c, s), v in cases.items() if cat == "ALL" or c == cat]
        assert len(records) == (45 if cat == "ALL" else 9)
        summaries[cat] = {}
        for method in ALL_KEYS:
            summaries[cat][method] = {}
            for metric in METRICS:
                arr = np.asarray([v["metrics"][method][metric] for v in records])
                assert np.isfinite(arr).all()
                summaries[cat][method][metric] = {
                    "mean": float(arr.mean()),
                    "sd_across_unique_test_images": float(arr.std(ddof=1)),
                    "n_unique_test_images": len(arr),
                }
    return cases, per_pair, cal_stats, summaries


def outputs(cases, per_pair, cal_stats, summary):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("per_calibration_9_images.json", cal_stats),
        ("test45_metrics.json", {f"{c}_{sid:04d}": v for (c, sid), v in cases.items()}),
        ("test45_summary.json", summary),
    ):
        (OUT / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
    with (OUT / "per_calibration_per_test_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "test_sample_id",
                "calibration_sample_id",
                "method",
                *METRICS,
            ],
        )
        writer.writeheader()
        for row in sorted(
            per_pair,
            key=lambda x: (
                x["category"],
                x["method"],
                x["calibration_sample_id"],
                x["test_sample_id"],
            ),
        ):
            writer.writerow(row)
    with (OUT / "test45_all_method_metrics.csv").open("w", newline="") as f:
        fields = ["category", "sample_id"] + [
            f"{m}_{metric}" for m in ALL_KEYS for metric in METRICS
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, v in sorted(cases.items()):
            row = {"category": key[0], "sample_id": key[1]}
            row.update(
                {
                    f"{m}_{metric}": v["metrics"][m][metric]
                    for m in ALL_KEYS
                    for metric in METRICS
                }
            )
            writer.writerow(row)
    print("NUMERIC_RESULTS_READY", OUT, flush=True)


def charts(cases, summary):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}
    )
    legends = [LABELS[m] for m in ALL_KEYS]
    fig, axs = plt.subplots(2, 2, figsize=(18, 10.5), layout="constrained")
    for ax, metric in zip(axs.flat, METRICS):
        avg = [summary["ALL"][k][metric]["mean"] for k in ALL_KEYS]
        sd = [
            summary["ALL"][k][metric]["sd_across_unique_test_images"] for k in ALL_KEYS
        ]
        ax.bar(range(len(ALL_KEYS)), avg, yerr=sd, capsize=3)
        ax.set_xticks(
            range(len(ALL_KEYS)), legends, rotation=35, ha="right", fontsize=7.8
        )
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.2)
        if metric == "mse":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "Test45: 12 method labels, mean +/- SD across 45 unique images", fontsize=16
    )
    fig.savefig(OUT / "figures/test45_12_methods_mean_sd.pdf", bbox_inches="tight")
    plt.close(fig)
    with PdfPages(OUT / "figures/test45_12_methods_by_category.pdf") as pdf:
        for cat in CATEGORIES:
            fig, axs = plt.subplots(2, 2, figsize=(18, 10.5), layout="constrained")
            for ax, metric in zip(axs.flat, METRICS):
                avg = [summary[cat][k][metric]["mean"] for k in ALL_KEYS]
                sd = [
                    summary[cat][k][metric]["sd_across_unique_test_images"]
                    for k in ALL_KEYS
                ]
                ax.bar(range(len(ALL_KEYS)), avg, yerr=sd, capsize=3)
                ax.set_xticks(
                    range(len(ALL_KEYS)), legends, rotation=35, ha="right", fontsize=7.8
                )
                ax.set_title(metric.upper())
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(
                f"{cat} (9 test cases), 5 calibration choices per tune-1 method"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print("SCORE_FIGURES_READY", flush=True)


def main():
    cases, per_pair, cal_stats, summary = collect()
    outputs(cases, per_pair, cal_stats, summary)
    charts(cases, summary)
    print(
        "TUNE1_TEST45_ANALYSIS_COMPLETE",
        len(cases),
        len(per_pair),
        "CALIBRATIONS",
        len(cal_stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
