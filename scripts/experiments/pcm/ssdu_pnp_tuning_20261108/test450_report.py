"""Final 11-method matched 450-image PCM comparison and LaTeX report update.

Requires two new 450-case GS-PnP test runs to complete. No test-driven tuning.
Reuses old benchmark methods only where source acquisition hashes and complete
method records match. Preserves original tuning-only report as version 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_reconstruct import (
    ROOT,
    SSDU,
    OUT,
    BENCH,
    PREV_ADMM,
    PREV_TV,
    BASE,
    CATEGORIES,
    METRICS,
    validate_archive,
    read_frozen,
    shafile,
)

REPORT = SSDU / "report"
REPORT_V1 = REPORT / "archive_tuning_only_v1"
METHODS = (
    ("zero_filled", "Zero-filled", "zero_filled"),
    ("spgl1_bp", "SPGL1-BP", "spgl1_bp"),
    ("spgl1_bpdn", "SPGL1-BPDN", "bpdn"),
    ("pnp_admm_oracle", "PnP-ADMM-oracle", "pnp_tuned"),
    ("gs_pnp_oracle", "GS-PnP-oracle", None),
    ("tv", "TV (val.-tuned)", "tv"),
    ("utv_c60", "UTV-C60", "C"),
    ("pnp_admm_ssdu5", "PnP-ADMM-tuned-5", "pnp_tuned"),
    ("pnp_admm_ssdu50", "PnP-ADMM-tuned-50", "pnp_tuned"),
    ("gs_pnp_ssdu5", "GS-PnP-tuned-5", None),
    ("gs_pnp_ssdu50", "GS-PnP-tuned-50", None),
)
DISPLAY = {k: label for k, label, _ in METHODS}
assert len(METHODS) == 11
SIGN = ("psnr", "ssim", "haarpsi", "mse")


def make_method_values():
    assert (OUT / "status.txt").read_text().startswith("COMPLETED_BOTH_450")
    frozen, _ = read_frozen()
    assert frozen["admm"]["rho"] == 0.5 and frozen["admm"]["sigma"] == 0.075
    bench, *_ = validate_archive()
    data = {}
    for method in ("gs_ssdu", "gs_oracle_step6"):
        r = list((OUT / "records" / method).glob("*/*.json"))
        assert len(r) == 450, (method, len(r))
        records = {}
        for file in r:
            x = json.loads(file.read_text())
            key = (x["category"], int(x["sample_id"]))
            assert key not in records and key in bench
            assert x["measurement_sha256"] == bench[key]["measurement_sha256"]
            assert x["pattern_sha256"] == bench[key]["pattern_sha256"]
            assert x["no_test_retraining"] is True
            assert all(math.isfinite(float(x["metrics"][m])) for m in METRICS)
            img = OUT / "raw" / method / f"{key[0]}_{key[1]:04d}.npy"
            assert img.exists()
            records[key] = x
        assert len(records) == 450 and set(records) == set(bench)
        data[method] = records
    table = {}
    admm_table = {}
    with PREV_ADMM.open(newline="") as f:
        admm_table = {
            (r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)
        }
    tv_table = {}
    with PREV_TV.open(newline="") as f:
        tv_table = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    assert len(admm_table) == len(tv_table) == 450
    base_table = {}
    with (BASE / "all_450_metrics.csv").open(newline="") as f:
        base_table = {
            (r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)
        }
    for key, archived in bench.items():
        record = {
            "category": key[0],
            "sample_id": int(key[1]),
            "measurement_sha256": archived["measurement_sha256"],
            "pattern_sha256": archived["pattern_sha256"],
        }
        for m, _, prefix in METHODS:
            if m == "gs_pnp_oracle":
                scores = data["gs_oracle_step6"][key]["metrics"]
            elif m.startswith("gs_pnp_ssdu"):
                scores = data["gs_ssdu"][key]["metrics"]
            else:
                scores = {
                    metric: float(archived[f"{prefix}_{metric}"]) for metric in METRICS
                }
            for metric in METRICS:
                record[f"{m}_{metric}"] = float(scores[metric])
        assert all(
            np.isfinite(record[f"{m}_{metric}"])
            for m, _, _ in METHODS
            for metric in METRICS
        )
        for metric in METRICS:
            assert (
                record[f"pnp_admm_oracle_{metric}"]
                == record[f"pnp_admm_ssdu5_{metric}"]
                == record[f"pnp_admm_ssdu50_{metric}"]
            )
            assert record[f"gs_pnp_ssdu5_{metric}"] == record[f"gs_pnp_ssdu50_{metric}"]
            assert (
                abs(
                    record[f"zero_filled_{metric}"]
                    - float(base_table[key][f"zero_filled_{metric}"])
                )
                < 1e-8
            )
            assert (
                abs(
                    record[f"pnp_admm_oracle_{metric}"]
                    - float(admm_table[key][f"pnp_{metric}"])
                )
                < 1e-8
            )
            assert abs(record[f"tv_{metric}"] - float(tv_table[key][metric])) < 1e-8
        table[key] = record
    assert len(table) == 450
    return table, data, admm_table, tv_table, bench


def create_stats(table, data, admm_table, tv_table):
    stats = {}
    cats = ("ALL",) + CATEGORIES
    for cat in cats:
        cases = [v for k, v in table.items() if cat == "ALL" or k[0] == cat]
        assert len(cases) == (450 if cat == "ALL" else 90)
        stats[cat] = {}
        for method, _, _ in METHODS:
            vals = {}
            for metric in METRICS:
                raw = np.array(
                    [c[f"{method}_{metric}"] for c in cases], dtype=np.float64
                )
                vals[metric] = {
                    "mean": float(raw.mean()),
                    "std": float(raw.std(ddof=1)),
                    "median": float(np.median(raw)),
                    "minimum": float(raw.min()),
                    "maximum": float(raw.max()),
                    "n": len(raw),
                }
            stats[cat][method] = vals
    # Runtime is reported only when comparable archived records exist.
    times = {}
    for method, _, _ in METHODS:
        if method in ("gs_pnp_oracle", "gs_pnp_ssdu5", "gs_pnp_ssdu50"):
            key = "gs_oracle_step6" if method == "gs_pnp_oracle" else "gs_ssdu"
            arr = [
                float(x["diagnostics"]["wall_seconds_with_final_gradient"])
                for x in data[key].values()
            ]
        elif method in ("pnp_admm_oracle", "pnp_admm_ssdu5", "pnp_admm_ssdu50"):
            arr = [float(x["pnp_wall_seconds"]) for x in admm_table.values()]
        elif method == "tv":
            arr = [float(x["seconds"]) for x in tv_table.values()]
        elif method == "spgl1_bp":
            with (BASE / "all_450_metrics.csv").open(newline="") as f:
                arr = [float(x["spgl1_runtime_s"]) for x in csv.DictReader(f)]
        elif method == "spgl1_bpdn":
            arr = []
            path = ROOT / "ten_methods_matched_450_20261105_v1/records/bpdn"
            for category in CATEGORIES:
                for fp in sorted((path / category).glob("*.json")):
                    rec = json.loads(fp.read_text())
                    key = (rec["category"], int(rec["sample_id"]))
                    assert rec["measurement_sha256"] == table[key]["measurement_sha256"]
                    assert rec["pattern_sha256"] == table[key]["pattern_sha256"]
                    arr.append(float(rec["seconds"]))
        else:
            arr = []
        assert len(arr) in (0, 450), (method, len(arr))
        times[method] = {
            "n": len(arr),
            "mean_seconds": float(np.mean(arr)) if arr else None,
            "std_seconds": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
        }
    paired = {}
    for m, base in (
        ("pnp_admm_ssdu5", "pnp_admm_oracle"),
        ("pnp_admm_ssdu50", "pnp_admm_oracle"),
        ("gs_pnp_ssdu5", "gs_pnp_oracle"),
        ("gs_pnp_ssdu50", "gs_pnp_oracle"),
    ):
        ps = np.array([v[f"{m}_psnr"] - v[f"{base}_psnr"] for v in table.values()])
        paired[m] = {
            "baseline": base,
            "mean_psnr_gain_db": float(ps.mean()),
            "std_psnr_gain_db": float(ps.std(ddof=1)),
            "fraction_better_psnr": float(np.mean(ps > 0)),
            "fraction_equal_psnr": float(np.mean(ps == 0)),
        }
    return stats, times, paired


def dump_metrics(table, stats, times, paired):
    dest = OUT / "merged"
    dest.mkdir(parents=True, exist_ok=True)
    names = list(next(iter(table.values())).keys())
    with (dest / "all_450_11_method_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for key in sorted(table):
            w.writerow(table[key])
    for filename, obj in (
        ("all_450_summary.json", stats),
        ("inference_time_summary.json", times),
        ("paired_psnr_gains.json", paired),
    ):
        (dest / filename).write_text(
            json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    checksum = {
        name: shafile(dest / name)
        for name in (
            "all_450_11_method_metrics.csv",
            "all_450_summary.json",
            "inference_time_summary.json",
            "paired_psnr_gains.json",
        )
    }
    return checksum


def charts(table, stats, paired):
    dest = OUT / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    metrics = [
        ("psnr", "PSNR (dB)"),
        ("ssim", "SSIM"),
        ("haarpsi", "HaarPSI"),
        ("mse", "MSE"),
    ]
    labels = [title for _, title, _ in METHODS]
    fig, ax = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
    for axes, (metric, label) in zip(ax.flat, metrics):
        values = [
            np.array([v[f"{key}_{metric}"] for v in table.values()])
            for key, _, _ in METHODS
        ]
        axes.boxplot(
            values, tick_labels=labels, showfliers=False, whis=1.5, patch_artist=False
        )
        axes.set_title(label, fontsize=13, weight="semibold")
        axes.tick_params(axis="x", labelrotation=40, labelsize=8)
        for tick in axes.get_xticklabels():
            tick.set_ha("right")
        axes.grid(True, axis="y", alpha=0.22)
        if metric == "mse":
            axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "450 fixed PCM test images | all 11 methods | 20% NPL-default noise",
        fontsize=15,
    )
    fig.savefig(dest / "all_450_11_method_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)
    with PdfPages(dest / "category_11_method_means_sd.pdf") as pdf:
        for cat in CATEGORIES:
            fig, ax = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
            for axes, (metric, label) in zip(ax.flat, metrics):
                mean = np.array([stats[cat][m][metric]["mean"] for m, _, _ in METHODS])
                std = np.array([stats[cat][m][metric]["std"] for m, _, _ in METHODS])
                axes.bar(
                    np.arange(len(METHODS)), mean, yerr=std, capsize=2.8, alpha=0.83
                )
                axes.set_xticks(
                    np.arange(len(METHODS)), labels, rotation=40, ha="right", fontsize=8
                )
                axes.set_title(label, weight="semibold")
                axes.grid(True, axis="y", alpha=0.22)
                if metric == "mse":
                    axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            fig.suptitle(
                f"{cat}: 90 paired test images, mean +/- sample SD", fontsize=15
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    # 450-mean ± SD summary, all 11 methods on one landscape page.
    fig, ax = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
    for axes, (metric, label) in zip(ax.flat, metrics):
        avg = np.array([stats["ALL"][m][metric]["mean"] for m, _, _ in METHODS])
        std = np.array([stats["ALL"][m][metric]["std"] for m, _, _ in METHODS])
        axes.bar(np.arange(11), avg, yerr=std, capsize=3, alpha=0.82)
        axes.set_xticks(np.arange(11), labels, rotation=40, ha="right", fontsize=8)
        axes.set_title(label, weight="semibold")
        axes.grid(True, axis="y", alpha=0.22)
        if metric == "mse":
            axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "Overall 450-image mean and standard deviation | same acquisitions", fontsize=15
    )
    fig.savefig(dest / "all_450_11_method_mean_sd.pdf", bbox_inches="tight")
    plt.close(fig)
    return [str(p) for p in sorted(dest.glob("*.pdf"))]


def latex_stats(stats):
    rows = []
    for method, label, _ in METHODS:
        s = stats["ALL"][method]
        cell = lambda m, n: f'{s[m]["mean"]:.{n}f} $\\pm$ {s[m]["std"]:.{n}f}'
        rows.append(
            label.replace("_", r"\_")
            + " & "
            + " & ".join(
                (
                    cell("psnr", 2),
                    cell("ssim", 4),
                    cell("haarpsi", 4),
                    f'{s["mse"]["mean"]:.3e} $\\pm$ {s["mse"]["std"]:.3e}',
                )
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_category_table(stats):
    lines = []
    # Eight distinct method configurations; repeated labels remain in the full results.
    for cat in CATEGORIES:
        d = stats[cat]
        lines.append(
            cat
            + " & "
            + " & ".join(
                f'{d[m]["psnr"]["mean"]:.2f}'
                for m in (
                    "zero_filled",
                    "spgl1_bp",
                    "spgl1_bpdn",
                    "pnp_admm_oracle",
                    "gs_pnp_oracle",
                    "tv",
                    "utv_c60",
                    "gs_pnp_ssdu5",
                )
            )
            + r" \\"
        )
    return "\n".join(lines)


def latex_runtime_table(times):
    lines = []
    for method, label, _ in METHODS:
        r = times[method]
        value = (
            f'{r["mean_seconds"]:.3f} $\\pm$ {r["std_seconds"]:.3f}'
            if r["n"] == 450
            else "not archived"
        )
        lines.append(label + " & " + str(r["n"]) + " & " + value + r" \\")
    return "\n".join(lines)


def update_report(stats, times, paired):
    old = REPORT / "SSDU_PnP_Tuning_Report.tex"
    oldpdf = REPORT / "SSDU_PnP_Tuning_Report.pdf"
    assert old.is_file() and oldpdf.is_file()
    REPORT_V1.mkdir(parents=True, exist_ok=True)
    orig = REPORT_V1 / "SSDU_PnP_Tuning_Report_v1.tex"
    pdf = REPORT_V1 / "SSDU_PnP_Tuning_Report_v1.pdf"
    if not orig.exists():
        shutil.copy2(old, orig)
    if not pdf.exists():
        shutil.copy2(oldpdf, pdf)
    source = orig.read_text()
    source = source.replace("Version 1.0", "Version 2.0")
    source = source.replace(
        "when the 450-image evaluation becomes available",
        "after the 450-image evaluation has been incorporated",
    )
    source = source.replace(
        "Parameter search: completed and frozen \\quad 450-image evaluation: pending",
        "Parameter search: completed and frozen \\quad 450-image evaluation: complete",
    )
    source = source.replace(
        "The agreed \\textbf{450-image test-set evaluation has not yet been run} as part of this experiment.",
        "The agreed \\textbf{450-image test-set evaluation has now been completed} using the frozen configurations. Its detailed results are in Section~5.",
    )
    # Correction of the status line and any other explicit pre-test language.
    source = source.replace(
        "450-image test-set evaluation has not yet been run",
        "450-image test-set evaluation has now been run",
    )
    where = r"\section{Conclusion}"
    assert source.count(where) == 1
    main = stats["ALL"]
    admm_note = (
        f"The two SSDU ADMM selections coincide exactly with the earlier "
        f"validation-tuned oracle settings $\\rho=0.5$, $\\sigma=0.075$ "
        f"and its stopping rule. Therefore their archived 450 test scores are "
        r"\emph{identical by construction}, not independent repeated runs."
    )
    gs_note = (
        f"The two SSDU GS-PnP selections also coincide, so they share one actual "
        f"450-image reconstruction run. The separate prior frozen Step-6 GS-PnP "
        f"oracle was reconstructed on the same matched acquisition set."
    )
    comparison = (
        f"The mean GS-PnP SSDU minus supervised-validation GS-PnP oracle "
        f'PSNR difference is {paired["gs_pnp_ssdu5"]["mean_psnr_gain_db"]:+.3f} dB '
        f'(sample SD {paired["gs_pnp_ssdu5"]["std_psnr_gain_db"]:.3f} dB). '
        f"For ADMM the difference is exactly zero because the parameter "
        f"and numerical solver choices coincide."
    )
    appendix = r"""
\subsection*{Matched 450-image test: all eleven methods}
We reconstructed the fixed 20\% NPL-noisy acquisitions of all 450 test images, 90 per category.
The seven existing comparison methods use verified historical numerical scores for
zero-filled, SPGL1-BP, SPGL1-BPDN, previously oracle-tuned ADMM, scalar TV
($\lambda=0.011$, 1024 iterations) and original UTV-C60, with the same
measurement/mask SHA256 hashes as the new runs. The prior Step-6 GS-PnP
oracle was newly reconstructed as \emph{paper-final $z$}, not substituted by
the separate historical GS-v2 method. All four SSDU selections were frozen
before reading these test metrics, and no test-based parameter adjustment was made.

@@ADMMNOTE@@

@@GSNOTE@@

\begin{table}[htbp]
\centering\scriptsize
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Method} & \textbf{PSNR, dB} & \textbf{SSIM} & \textbf{HaarPSI} & \textbf{MSE}\\
\midrule
@@STATS@@
\bottomrule
\end{tabular}
\caption{Mean $\pm$ sample standard deviation of 450 paired reconstructions.
Scores computed on raw (unclipped) image arrays. Identical parameter sets share
the same test computation but retain all labels.}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../test450/v1/figures/all_450_11_method_boxplots.pdf}
\caption{All 11 methods shown together over the 450 fixed test acquisitions.
The box plots exclude plotted outliers for legibility but include every case
in the descriptive statistics.}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../test450/v1/figures/all_450_11_method_mean_sd.pdf}
\caption{All 11 methods: test-set average and sample standard deviation for
PSNR, SSIM, HaarPSI and MSE.}
\end{figure}

@@COMPARISON@@

\subsection*{Reconstruction time}
The table gives seconds per reconstructed image, from new GPU runs or verified
historical records. Offline tuning time is excluded. The methods were not
all timed on the same GPU; therefore these values are descriptive, rather than
a controlled hardware-speed benchmark. Identical 5/50 selections reuse one
numerical reconstruction, not two independent computations.

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrl@{}}
\toprule
\textbf{Method} & \textbf{Timed cases} & \textbf{Mean $\pm$ SD, seconds/image}\\
\midrule
@@RUNTIME@@
\bottomrule
\end{tabular}
\caption{Available reconstruction times. Missing timings are not estimated.}
\end{table}

\subsection*{Per-category comparison}
Table~\ref{tab:categorypsnr} summarises PSNR means for the seven numerical
choices that are distinct after accounting for the repeated SSDU selections.
The full 11-method per-category metrics, standard deviations and boxplots
are retained as CSV, JSON and a five-page PDF in the test450 results directory.

\begin{table}[htbp]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
\textbf{Category} & \textbf{ZF} & \textbf{SPGL1} & \textbf{BPDN} &
\textbf{ADMM} & \textbf{GS oracle} & \textbf{TV} &
\textbf{C60} & \textbf{GS SSDU}\\
\midrule
@@CATEGORYROWS@@
\bottomrule
\end{tabular}}
\caption{Per-category mean PSNR (dB), with 90 test images in each category.
ADMM and GS SSDU repeated five/50 selections have been collapsed only in
this compact table, not in the full 11-method results.}
\label{tab:categorypsnr}
\end{table}
"""
    for k, v in {
        "@@ADMMNOTE@@": admm_note,
        "@@GSNOTE@@": gs_note,
        "@@STATS@@": latex_stats(stats),
        "@@COMPARISON@@": comparison,
        "@@CATEGORYROWS@@": latex_category_table(stats),
        "@@RUNTIME@@": latex_runtime_table(times),
    }.items():
        assert k in appendix
        appendix = appendix.replace(k, v)
    source = source.replace(where, appendix + "\n" + where)
    # Revise the original conclusion to reflect the new test results.
    needle = "Next, the four frozen SSDU configurations should be evaluated"
    if needle in source:
        source = source.replace(
            "Next, the four frozen SSDU configurations should be evaluated",
            "We have now evaluated the four frozen SSDU configurations",
        )
    conclusion_note = (
        r"""
\paragraph{Test-set conclusion.} The SSDU-5 and SSDU-50 choices match within each
PnP family, so collecting ten validation images per category did not change
the selected grid point. The ADMM SSDU choice is exactly the original
ground-truth-validation ADMM oracle parameter pair; the GS-PnP SSDU
selection is distinct from the earlier Step-6 GS oracle. """
        + comparison
        + r"""
The interpretation is limited by possible temporal PCM noise correlation,
the null-space of undersampled Hadamard imaging and prior exploratory use of
these 450 images in the broader project. No further test-driven tuning is allowed.
"""
    )
    cite_where = r"\renewcommand{\refname}{Primary references}"
    assert source.count(cite_where) == 1
    source = source.replace(cite_where, conclusion_note + "\n" + cite_where)
    destination = REPORT / "SSDU_PnP_Tuning_Report_Test450.tex"
    destination.write_text(source)
    print("UPDATED_REPORT_SOURCE", destination)
    return destination


def main():
    table, data, admm, tv, bench = make_method_values()
    stats, times, paired = create_stats(table, data, admm, tv)
    checks = dump_metrics(table, stats, times, paired)
    figs = charts(table, stats, paired)
    tex = update_report(stats, times, paired)
    manifest = {
        "status": "complete_450_evaluation",
        "num_images": 450,
        "new_gs_ssdu": 450,
        "new_gs_oracle_step6": 450,
        "all_11_methods": list(DISPLAY),
        "prior_methods_reused_after_exact_hash_verification": True,
        "frozen_admm_5_and_50_equal_oracle": True,
        "frozen_gs_5_and_50_equal_each_other": True,
        "input_measurement_csv_sha256": shafile(BENCH),
        "stage1_tuning_versions_remain_frozen": True,
        "merged_files_sha256": checks,
        "new_figures": figs,
        "report_tex": str(tex),
    }
    (OUT / "evaluation_complete.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "EVALUATION_COMPLETE",
        json.dumps(
            {
                "overall_psnr": {
                    m: round(stats["ALL"][m]["psnr"]["mean"], 3) for m in DISPLAY
                },
                "new_vs_oracle_psnr": paired["gs_pnp_ssdu5"],
                "output": str(OUT),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
