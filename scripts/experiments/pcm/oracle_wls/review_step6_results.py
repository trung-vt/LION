"""Step 6 reviewer analysis and CORRECTLY labelled reproducible figures.

Purely post-hoc descriptive analysis of two immutable 50-case archives.
No reconstructions, ground-truth images, weights or acquisitions are edited.
The x-axis category follows the actual category/sample_id in the manifests,
NOT the order in which pilot and full runs were executed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, pearsonr, wilcoxon

from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    write_json_new,
)

SCHEMA = "pcm-step6-independent-review-20261102-v1"
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
ORIGINAL_ANCHOR = "8e71d0977db94fde758895db24e193517aaec28d2a040e8d5ca55a2436e1c2e1"
TIGHT_ANCHOR = "ff92d1f47bcf25fc1c410d164d3a5989c9d43f6b1810ffc9636fa94b607a4f4a"
EXPECTED_ACQUISITION = (
    "b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac"
)
EXPECTED_WEIGHT = "50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1"
EXPECTED_CALIBRATION = (
    "fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a"
)
SEED = 20261102


def order(records):
    return sorted(
        records,
        key=lambda r: (
            CATEGORIES.index(r["category"]),
            int(r["permutation_position_1based"]),
        ),
    )


def read_frozen(path, sha):
    path = Path(path)
    if _sha256(path / "manifest.json") != sha:
        raise ValueError("Original or tight Step 6 manifest SHA256 changed")
    data = json.loads((path / "manifest.json").read_text())
    if len(data["cases"]) != 50 or data["test_cases"] != 0:
        raise ValueError("Unexpected Step 6 validation count")
    if len({(r["category"], r["sample_id"]) for r in data["cases"]}) != 50:
        raise ValueError("Duplicate validation records")
    if data["source_acquisitions_sha256"] != EXPECTED_ACQUISITION:
        raise ValueError("Wrong frozen acquisition provenance")
    if data["source_weights_sha256"] != EXPECTED_WEIGHT:
        raise ValueError("Wrong Step 3 weight provenance")
    if data["source_calibration_sha256"] != EXPECTED_CALIBRATION:
        raise ValueError("Wrong Step 5 epsilon provenance")
    return data


def _delta(r, name):
    if name == "psnr":
        return float(
            r["methods"]["wls"]["metrics"]["psnr"]
            - r["methods"]["ols"]["metrics"]["psnr"]
        )
    return float(
        r["methods"]["wls"]["metrics"][name] - r["methods"]["ols"]["metrics"][name]
    )


def _gap(r, name):
    x = r["methods"][name]["solver"]
    return x.get("relative_duality_gap", None)


def signed_results(selected):
    """Two-sided exact sign test and a separate paired Wilcoxon test."""
    x = np.array([_delta(r, "psnr") for r in selected], dtype=np.float64)
    n = int(np.count_nonzero(x))
    wins = int(np.sum(x > 0))
    p_sign = (
        float(binomtest(wins, n, p=0.5, alternative="two-sided").pvalue) if n else None
    )
    p_wilcoxon = (
        float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
        if n
        else None
    )
    rng = np.random.default_rng(SEED)
    draw_means = (
        x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
        if len(x)
        else np.array([])
    )
    return {
        "n": int(len(x)),
        "mean_delta_psnr": float(x.mean()) if len(x) else None,
        "median_delta_psnr": float(np.median(x)) if len(x) else None,
        "sd_delta_psnr": float(x.std(ddof=1)) if len(x) > 1 else None,
        "wls_psnr_wins": wins,
        "wls_psnr_losses": int(np.sum(x < 0)),
        "ties": int(np.sum(x == 0)),
        "exact_two_sided_sign_test_p": p_sign,
        "two_sided_wilcoxon_p": p_wilcoxon,
        "independent_paired_bootstrap_mean_95pct": (
            list(map(float, np.quantile(draw_means, [0.025, 0.975])))
            if len(x)
            else None
        ),
    }


def create_analysis(original_path, tight_path, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed Step 6 analysis {out}")
    old = read_frozen(original_path, ORIGINAL_ANCHOR)
    tight = read_frozen(tight_path, TIGHT_ANCHOR)
    expected = {(r["category"], r["sample_id"]) for r in old["cases"]}
    if expected != {(r["category"], r["sample_id"]) for r in tight["cases"]}:
        raise ValueError("Old/tight cases differ")
    old_by = {(r["category"], r["sample_id"]): r for r in old["cases"]}
    tight_rows = order(tight["cases"])
    joint = []
    for t in tight_rows:
        key = t["category"], t["sample_id"]
        o = old_by[key]
        g_ols = _gap(t, "ols")
        g_wls = _gap(t, "wls")
        joint.append(
            {
                "category": key[0],
                "sample_id": key[1],
                "permutation_position_1based": int(t["permutation_position_1based"]),
                "delta_psnr_1e_minus3": _delta(o, "psnr"),
                "delta_psnr_1e_minus6": _delta(t, "psnr"),
                "ols_psnr_1e_minus6": float(t["methods"]["ols"]["metrics"]["psnr"]),
                "wls_psnr_1e_minus6": float(t["methods"]["wls"]["metrics"]["psnr"]),
                "ols_relative_gap_1e_minus6": g_ols,
                "wls_relative_gap_1e_minus6": g_wls,
                "relative_gap_difference": (
                    g_wls - g_ols if g_ols is not None and g_wls is not None else None
                ),
                "both_root_found": (
                    t["methods"]["ols"]["status"] == "PASS"
                    and t["methods"]["wls"]["status"] == "PASS"
                ),
                "both_gaps_below_0point01": (
                    g_ols is not None
                    and g_wls is not None
                    and g_ols < 0.01
                    and g_wls < 0.01
                ),
                "both_gaps_below_0point05": (
                    g_ols is not None
                    and g_wls is not None
                    and g_ols < 0.05
                    and g_wls < 0.05
                ),
            }
        )

    out.mkdir(parents=True, exist_ok=False)
    with (out / "correctly_ordered_50_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(joint[0]))
        writer.writeheader()
        writer.writerows(joint)

    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_opt_tol": float(old["solver"]["opt_tol"]),
        "tighter_opt_tol": float(tight["solver"]["opt_tol"]),
        "historical_manifest_sha256": ORIGINAL_ANCHOR,
        "tighter_manifest_sha256": TIGHT_ANCHOR,
        "signed_psnr": {
            "original_1e_minus3": signed_results(order(old["cases"])),
            "tighter_1e_minus6": signed_results(tight_rows),
            "without_solver_warning_pairs": signed_results(
                [
                    t
                    for t in tight_rows
                    if t["methods"]["ols"]["status"] == "PASS"
                    and t["methods"]["wls"]["status"] == "PASS"
                ]
            ),
            "both_duality_gaps_below_0point01": signed_results(
                [
                    t
                    for t in tight_rows
                    if _gap(t, "ols") is not None
                    and _gap(t, "wls") is not None
                    and _gap(t, "ols") < 0.01
                    and _gap(t, "wls") < 0.01
                ]
            ),
            "both_duality_gaps_below_0point05": signed_results(
                [
                    t
                    for t in tight_rows
                    if _gap(t, "ols") is not None
                    and _gap(t, "wls") is not None
                    and _gap(t, "ols") < 0.05
                    and _gap(t, "wls") < 0.05
                ]
            ),
        },
        "per_category": {},
        "gap_vs_psnr": {
            "available": sum(r["relative_gap_difference"] is not None for r in joint)
        },
        "large_original_gains": {
            "top_3": [
                {
                    "category": r["category"],
                    "sample_id": r["sample_id"],
                    "delta_psnr_1e_minus3": r["delta_psnr_1e_minus3"],
                    "delta_psnr_1e_minus6": r["delta_psnr_1e_minus6"],
                }
                for r in sorted(
                    joint, key=lambda r: r["delta_psnr_1e_minus3"], reverse=True
                )[:3]
            ],
        },
        "statistical_scope": "Exploratory paired tests; no optimality certificate; small per-category sizes",
    }
    for cat in CATEGORIES:
        subset_old = [r for r in old["cases"] if r["category"] == cat]
        subset_tight = [r for r in tight_rows if r["category"] == cat]
        summary["per_category"][cat] = {
            "original": signed_results(subset_old),
            "tight": signed_results(subset_tight),
        }
    obs = [r for r in joint if r["relative_gap_difference"] is not None]
    if len(obs) >= 3:
        summary["gap_vs_psnr"].update(
            {
                "pearson_gap_difference_vs_delta_psnr": float(
                    pearsonr(
                        [r["relative_gap_difference"] for r in obs],
                        [r["delta_psnr_1e_minus6"] for r in obs],
                    ).statistic
                ),
                "pearson_ols_psnr_vs_delta_psnr": float(
                    pearsonr(
                        [r["ols_psnr_1e_minus6"] for r in obs],
                        [r["delta_psnr_1e_minus6"] for r in obs],
                    ).statistic
                ),
                "n_ols_gap_over_point1": sum(
                    r["ols_relative_gap_1e_minus6"] > 0.1 for r in obs
                ),
                "n_wls_gap_over_point1": sum(
                    r["wls_relative_gap_1e_minus6"] > 0.1 for r in obs
                ),
            }
        )
    # Make the plotting category order explicit and visible, independent of
    # pilot-first reconstruction order.
    plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "savefig.dpi": 180})
    xs = np.arange(50)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dif = np.array([r["delta_psnr_1e_minus3"] for r in joint])
    bars = ax.bar(xs, dif, color=np.where(dif > 0, "#3c7aa4", "#bc7050"), width=0.81)
    for x in [9.5, 19.5, 29.5, 39.5]:
        ax.axvline(x, color="#c5cbd0", lw=0.8)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks([4.5, 14.5, 24.5, 34.5, 44.5], CATEGORIES)
    ax.set_xlim(-1, 50)
    ax.set_ylabel("PSNR WLS - OLS (dB)")
    ax.set_title("Historical Step 6: corrected category labels, 50 paired images")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "figure1_corrected_paired_psnr_by_category.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True)
    for ax, name in zip(axes, ("delta_psnr_1e_minus3", "delta_psnr_1e_minus6")):
        vals = np.array([r[name] for r in joint])
        ax.bar(xs, vals, color=np.where(vals > 0, "#3c7aa4", "#bc7050"), width=0.81)
        ax.axhline(0, color="black", lw=0.8)
        for x in [9.5, 19.5, 29.5, 39.5]:
            ax.axvline(x, color="#c5cbd0", lw=0.8)
        ax.set_ylabel("WLS - OLS (dB)")
        ax.grid(axis="y", alpha=0.16)
        ax.set_title(
            "opt_tol = "
            + ("1e-3, original" if name.endswith("minus3") else "1e-6, sensitivity")
        )
    axes[-1].set_xticks([4.5, 14.5, 24.5, 34.5, 44.5], CATEGORIES)
    axes[-1].set_xlim(-1, 50)
    fig.suptitle("Same validation images, correctly grouped by category", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "figure2_original_vs_tighter_psnr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5))
    x = np.array([r["relative_gap_difference"] for r in obs])
    y = np.array([r["delta_psnr_1e_minus6"] for r in obs])
    colors = ["#ba6547" if not r["both_root_found"] else "#2c7b9f" for r in obs]
    ax.scatter(x, y, c=colors, s=33, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SPGL1 relative duality gap: WLS - OLS")
    ax.set_ylabel("PSNR WLS - OLS (dB)")
    ax.set_title("Tighter run: residual accuracy vs optimality-gap imbalance")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(out / "figure3_psnr_vs_duality_gap_difference.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    ct = np.arange(5)
    for shift, key, color, title in [
        (-0.17, "original", "#7d9ec3", "1e-3"),
        (0.17, "tight", "#cc8061", "1e-6"),
    ]:
        means = [summary["per_category"][c][key]["mean_delta_psnr"] for c in CATEGORIES]
        medians = [
            summary["per_category"][c][key]["median_delta_psnr"] for c in CATEGORIES
        ]
        ax.bar(
            ct + shift, means, width=0.30, label=f"Mean {title}", color=color, alpha=0.8
        )
        ax.scatter(ct + shift, medians, color="#23394e", marker="_", s=150, zorder=5)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(ct, CATEGORIES)
    ax.set_ylabel("WLS - OLS PSNR (dB)")
    ax.set_title("Category means (bars) and medians (horizontal marks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure4_category_mean_and_median.png")
    plt.close(fig)

    write_json_new(out / "analysis.json", summary)
    source_paths = (
        "scripts/experiments/pcm/oracle_wls/compare_ols_wls_step6.py",
        "scripts/experiments/pcm/oracle_wls/verify_step6.py",
        "scripts/experiments/pcm/oracle_wls/weighted_bpdn.py",
        "scripts/experiments/pcm/oracle_wls/run_step6_tighter_tolerance.py",
        "scripts/experiments/pcm/oracle_wls/independent_step6_checks.py",
        "LION/pcm/spgl1_bpdn.py",
        "LION/classical_algorithms/spgl1_torch.py",
        "LION/operators/Wavelet2D.py",
        "LION/operators/PhotocurrentMapOp.py",
        "LION/utils/metrics.py",
    )
    packages = (
        "numpy",
        "scipy",
        "torch",
        "torchmetrics",
        "piq",
        "scikit-image",
        "PyWavelets",
        "spgl1",
        "matplotlib",
    )
    provenance = {
        "schema": "pcm-step6-reviewed-provenance-sidecar-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "current_python_version": platform.python_version(),
        "current_distribution_versions": {
            x: importlib.metadata.version(x) for x in packages
        },
        "current_source_sha256": {x: _sha256(Path(x)) for x in source_paths},
        "historical_reconstruction_commits": {
            "opt_tol_1e-3": "693ff44feecf48b8cab19228d4eebbf9f7026aa0",
            "opt_tol_1e-6": "c5e191222381293f0dc22a039aee1cdd915080e9",
        },
        "recorded_execution_source_sha256": {
            "opt_tol_1e-3": old["source_sha256"],
            "opt_tol_1e-6": tight["source_sha256"],
        },
        "tighter_runner_source_sha256": _sha256(
            Path("scripts/experiments/pcm/oracle_wls/run_step6_tighter_tolerance.py")
        ),
        "source_provenance_warning": (
            "This reviewed sidecar records today's inspected dependencies and "
            "historical Git commit IDs. Dependency versions at historical "
            "execution are not necessarily all independently known. "
            "The frozen archive manifests were not modified."
        ),
        "statistical_source_sha256": _sha256(Path(__file__)),
        "original_archive_manifest_sha256": ORIGINAL_ANCHOR,
        "tighter_archive_manifest_sha256": TIGHT_ANCHOR,
    }
    write_json_new(out / "provenance.json", provenance)
    files = sorted(str(x.relative_to(out)) for x in out.iterdir() if x.is_file())
    sums = "".join(f"{_sha256(out/x)}  {x}\n" for x in files)
    (out / "SHA256SUMS").write_text(sums)
    print("REVIEW_ANALYSIS_JSON_SHA256=" + _sha256(out / "analysis.json"))
    print("REVIEW_SIGN_TESTS", json.dumps(summary["signed_psnr"], indent=2))
    print("REVIEW_GAP_CORRELATION", summary["gap_vs_psnr"])
    print("REVIEW_OUTPUT", out)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--tight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    create_analysis(a.original, a.tight, a.output)


if __name__ == "__main__":
    main()
