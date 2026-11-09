"""Generate the ADMM-only completed tune-1 calibration report.

No test45 image access. The run does not depend on GS-PnP selections.
Produces a versioned LaTeX report, figures, CSV and provenance JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    digest,
)

HERE = Path(__file__).resolve().parent
RESULTS = OUTPUT / "tuning/admm"
REPORT = OUTPUT / "report/admm_tune1_completed"
FIGURES = REPORT / "figures"


def load():
    calibration = json.loads(
        (OUTPUT / "prepared/validation_calibrations.json").read_text()
    )
    assert len(calibration["cases"]) == 25
    saved = json.loads((RESULTS / "admm_tune1_summary.json").read_text())
    assert saved["status"] == "frozen_all_25_admm_tune1_validation_only"
    assert saved["calibrations"] == 25
    rows = []
    fold_times = []
    for item in calibration["cases"]:
        cat = item["category"]
        sid = int(item["sample_id"])
        root = RESULTS / cat / f"{sid:04d}"
        selection_file = root / "frozen_selection.json"
        record = json.loads(selection_file.read_text())
        trace = json.loads((root / "search_trace.json").read_text())
        assert record["stage"] == "validation_selection_frozen_before_test"
        assert record["no_test_metrics_used"] and record["no_image_ground_truth_used"]
        assert trace["no_test_metric_used"] and trace["no_image_ground_truth_used"]
        assert record["validation_source_file_sha256"] == item["source_file_sha256"]
        assert record["fold_file_sha256"] == item["folds_sha256"]
        assert trace["evaluations"] <= 25
        cfg = record["winner_params"]
        scored = list((root / "evaluations").glob("t1_admm_*.json"))
        failures = list((root / "evaluations").glob("*.failure.json"))
        assert len(failures) == 0
        assert len(scored) == trace["evaluations"]
        row = {
            "category": cat,
            "calibration_id": sid,
            "rho": cfg["rho"],
            "sigma": cfg["sigma"],
            "ssdu_mse": record["winner_ssdu_mse"],
            "candidate_evaluations": record["probes"],
            "failed_evaluations": record["invalid_probes"],
            "source_sha256": digest(selection_file),
        }
        rows.append(row)
        for file in scored:
            rec = json.loads(file.read_text())
            assert len(rec["folds"]) == 4
            assert rec["uses_ground_truth"] is False and rec["test_set_access"] is False
            for fold in rec["folds"]:
                fold_times.append(float(fold["wall_seconds_with_operators"]))
    assert len(rows) == 25
    assert len(fold_times) == 420 * 4
    assert sum(x["candidate_evaluations"] for x in rows) == 420
    assert sum(x["failed_evaluations"] for x in rows) == 0
    assert (
        abs(
            statistics.mean(x["ssdu_mse"] for x in rows)
            - saved["mean_ssdu_measurement_mse"]
        )
        < 1e-15
    )
    return rows, fold_times


def generate_figures(rows):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained")
    for i, cat in enumerate(CATEGORIES):
        sample = [r for r in rows if r["category"] == cat]
        axs[0].scatter(
            [r["rho"] for r in sample], [r["sigma"] for r in sample], label=cat, s=40
        )
        axs[1].scatter([i] * 5, [r["ssdu_mse"] for r in sample], s=35)
        axs[2].scatter([i] * 5, [r["candidate_evaluations"] for r in sample], s=35)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Selected ADMM rho (log)")
    axs[0].set_ylabel("Selected denoiser sigma (log)")
    axs[0].set_title("Five independent calibrations per category")
    axs[0].grid(alpha=0.18)
    axs[0].legend(fontsize=7, frameon=False)
    for ax, label in (
        (axs[1], "Selected held-out MSE"),
        (axs[2], "Candidates evaluated"),
    ):
        ax.set_xticks(range(5), CATEGORIES, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.20)
    axs[1].set_title("Measured validation SSDU risk")
    axs[2].set_title("Search effort per calibration")
    fig.savefig(FIGURES / "admm_tune1_25_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.7, 3.7), layout="constrained")
    categories = list(CATEGORIES)
    means = [
        statistics.mean(x["ssdu_mse"] for x in rows if x["category"] == c)
        for c in categories
    ]
    sds = [
        statistics.stdev(x["ssdu_mse"] for x in rows if x["category"] == c)
        for c in categories
    ]
    ax.bar(categories, means, yerr=sds, capsize=4)
    ax.set_ylabel("SSDU held-out measurement MSE")
    ax.set_title("Five independently tuned validation acquisitions per category")
    ax.grid(axis="y", alpha=0.20)
    fig.savefig(FIGURES / "admm_tune1_category_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(x, n=7):
    return f"{float(x):.{n}f}"


def build_tex(rows, fold_times):
    result_rows = []
    category_rows = []
    for cat in CATEGORIES:
        items = sorted(
            [x for x in rows if x["category"] == cat], key=lambda x: x["calibration_id"]
        )
        assert len(items) == 5
        for x in items:
            result_rows.append(
                f'{cat} & {x["calibration_id"]:04d} & '
                f'{x["rho"]:.5f} & {x["sigma"]:.5f} & '
                f'{x["ssdu_mse"]:.8f} & {x["candidate_evaluations"]}' + r" \\"
            )
        category_rows.append(
            cat
            + " & "
            + fmt(statistics.mean(x["ssdu_mse"] for x in items), 8)
            + " & "
            + fmt(statistics.stdev(x["ssdu_mse"] for x in items), 8)
            + " & "
            + fmt(statistics.median(x["rho"] for x in items), 5)
            + " & "
            + fmt(statistics.median(x["sigma"] for x in items), 5)
            + " & "
            + fmt(statistics.mean(x["candidate_evaluations"] for x in items), 1)
            + r" \\"
        )
    total_candidate = len(list(RESULTS.glob("*/*/evaluations/*.json")))
    assert total_candidate == 420
    risks = [r["ssdu_mse"] for r in rows]
    source_admm = HERE / "run_tune1.py"
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=21mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,tabularx,graphicx,hyperref,fancyhdr,enumitem,xcolor}
\definecolor{ink}{HTML}{21364D}
\hypersetup{colorlinks=true,urlcolor=ink,linkcolor=ink}
\setlength{\parindent}{0pt}\setlength{\parskip}{.50em}
\setlength{\emergencystretch}{3em}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small PCM / ADMM tune-1 validation}
\fancyhead[R]{\small 9 November 2026}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\begin{document}
{\LARGE\bfseries\color{ink} PnP-ADMM Tune-1: completed parameter search}\\[5pt]
{\large PCM at 20\% sampling with fixed NPL-default noise}\\[4pt]
{\color{gray} Experiment results report, version 1.0 | 9 November 2026}\\[4pt]
\textbf{Status: ADMM calibration complete (25/25); GS-PnP tuning and test set 45 pending.}
\par\medskip\hrule

\section{Executive summary}

We completed \textbf{25 independent Tune-1 calibration searches} for
PnP-ADMM-DRUNet, with five randomly selected validation acquisitions
from each of five PCM categories. All acquisitions used the original
20\% Hadamard sampling masks and NPL-default noise observations.
The hyperparameters were selected using held-out acquired measurements
without access to ground-truth images or test image metrics.

The search evaluated \textbf{420 distinct parameter configurations},
each using four fixed SSDU folds, for \textbf{1,680 fold reconstructions}.
All 25 selections are frozen. There were \textbf{zero failed candidates}.
The mean of the 25 individually selected SSDU prediction errors was
\textbf{<<MEAN>>} (sample SD <<SD>>).

\textbf{Scope:} This report concerns \emph{validation-only tuning}.
The new configurations have not yet been evaluated on test set 45,
because the agreed test runner requires the 25 GS-PnP selections to be
frozen too. We therefore do not claim test PSNR, SSIM, HaarPSI or
reconstruction superiority from these calibration losses.

\section{Tuning algorithm}

For one calibration acquisition $y=Ax+\varepsilon$ at
$256\times256$ resolution and 20\% multilevel Hadamard sampling
($J=8$, coarse order $J_c=6$), the 13,107 acquired measurements
are split into four fixed SSDU folds. All 4,096 coarse rows remain
in the fit set; the remaining 9,011 acquired fine rows provide
held-out measurement subsets $\Lambda_k$.

For parameter vector $\theta=(\rho,\sigma)$, we minimise
\[
L(\theta)=\frac14\sum_{k=1}^{4}
\frac{\|A_{\Lambda_k}\widehat x_\theta(A_{\Theta_k},y_{\Theta_k})
-y_{\Lambda_k}\|_2^2}{|\Lambda_k|},
\]
where the reconstruction only sees the complementary rows $\Theta_k$.
The denoiser weights and ADMM solver remain fixed.

Search in logarithmic coordinates with positive parameter bounds
\[
\rho\in[0.03,4.0],\qquad \sigma\in[0.015,0.18].
\]
Start at the geometric midpoint. For each coordinate, evaluate two
interior points that divide its log interval into thirds, keeping the
two-thirds interval favoured by the lower SSDU loss.
Run \textbf{three contractions per coordinate and two coordinate
passes with reversed order}, caching duplicate evaluations.
Select the lowest \emph{actually evaluated} finite SSDU risk.

This approach assumes \emph{one-dimensional unimodality} of the
objective while other parameters are fixed. That assumption has
not been proved. It does not establish global optimality of
the nonlinear joint PnP objective.

\section{Step-by-step plan and execution}

\begin{enumerate}[leftmargin=1.5em]
\item Freeze the validation calibration sample IDs, test45 sample IDs,
RNG seed, archived acquisition SHA256 hashes, and parameter bounds.
\textbf{Complete}.
\item Verify the old normalised PCM data-fidelity convention,
four-fold measurement scoring, deterministic GPU behaviour, and
numerical search implementation. \textbf{Complete}.
\item Run five independent ADMM parameter searches per category.
Save candidate scores, original source checksums, fold diagnostics
and numerical errors. \textbf{Complete, 25 of 25}.
\item Freeze each winner as an independent validation-only
configuration. \textbf{Complete, 25 of 25}.
\item Run the separate GS-PnP calibration searches. \textbf{Pending,
deliberately not started}.
\item When all 50 selections are frozen, reconstruct the fixed
test set 45, nine images per category, using all acquired rows.
\textbf{Pending}.
\item Compare with existing PCM methods using image
PSNR, SSIM, HaarPSI, MSE, runtimes, and stored reconstruction
arrays. \textbf{Pending}.
\end{enumerate}

\section{Experiment running and numerical checks}

The ADMM search used \textbf{tuni's NVIDIA RTX 4070}, with
deterministic CUDA/CuDNN settings and the original DRUNet weights.
Each candidate requires four complete ADMM reconstructions, with
the original maximum of 100 iterations and residual-based stopping.
Candidate caching reduced the realised search from the maximum of
25 evaluations per calibration to a mean of
\textbf{<<MEANPROBES>>}, with range <<MINPROBES>>--<<MAXPROBES>>.
Each complete SSDU fold reconstruction took a mean of
<<MEANFOLDTIME>> seconds in the archived execution records,
but timing varies with the chosen parameter and stopping behaviour.

Completed: 25 calibrations, 420 candidates, 1,680 fold
reconstructions. No invalid candidates were recorded.
The frozen selections and every scored fold remain on tuni.
\emph{The RTX 4070 was not used to evaluate test45 reconstruction
quality in this phase.}

\section{Results and deliverables}

\subsection*{Five PCM category summaries}

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Category & Mean SSDU & SD SSDU & Median $\rho$ & Median $\sigma$ & Mean probes\\
\midrule
<<CATEGORY_ROWS>>
\bottomrule
\end{tabular}
\caption{Each category contains five independently tuned validation
measurements. Mean and sample SD describe selected held-out
\emph{measurement} MSE, not image reconstruction error.}
\end{table}

\subsection*{All 25 independently selected ADMM configurations}

\small
\begin{longtable}{@{}lrrrrr@{}}
\toprule
Category & Validation ID & $\rho$ & $\sigma$ &
SSDU MSE & Candidate evaluations\\
\midrule
\endfirsthead
\toprule
Category & Validation ID & $\rho$ & $\sigma$ &
SSDU MSE & Candidate evaluations\\
\midrule
\endhead
<<RESULT_ROWS>>
\bottomrule
\end{longtable}
\normalsize

\begin{figure}[htbp]\centering
\includegraphics[width=.82\linewidth]{figures/admm_tune1_25_summary.pdf}
\caption{Selected regularisation/denoiser parameters,
per-calibration SSDU prediction loss and candidate evaluation counts.}
\end{figure}

\clearpage
\begin{figure}[htbp]\centering
\includegraphics[width=.64\linewidth]{figures/admm_tune1_category_loss.pdf}
\caption{Within-category mean and sample SD of selected SSDU loss.}
\end{figure}

\subsection*{Reproducibility deliverables}

The ZIP contains the full 25-selection JSON/CSV summary, all 420
candidate JSON score files with four SSDU-fold records each,
all frozen parameter files and search traces, the fixed validation
and test45 manifests, source scripts, operator/scoring provenance,
and this editable \LaTeX{} report with its figures and PDF.
No test45 raw images or test quality metrics are included,
because no test45 run has been performed.

\section{Conclusion}

The category-wise Tune-1 parameter search for PnP-ADMM is
\textbf{complete and numerically successful}: all 25 independent
validation-only selections were frozen, with no recorded candidate
failure. The selected $\rho$ and denoiser $\sigma$ values vary
across calibration acquisitions, so a single universal parameter
pair is not established by this experiment.

A lower held-out measurement error does not necessarily imply
higher image PSNR or preservation of thin lines.
The next required study is transfer of each frozen configuration
to its nine category-matched test45 images, followed by
comparison with existing methods. GS-PnP calibration has been
left untouched as requested.

\paragraph{Primary references.}
Yaman et al. (2020), SSDU measurement partitioning,
\url{https://doi.org/10.1002/mrm.28378}.

Hurault, Leclaire and Papadakis (2022), gradient-step
plug-and-play with backtracking,
\url{https://openreview.net/pdf?id=fPhKeld3Okz}.
\end{document}
"""
    replacements = {
        "<<MEAN>>": fmt(statistics.mean(risks), 9),
        "<<SD>>": fmt(statistics.stdev(risks), 9),
        "<<CATEGORY_ROWS>>": "\n".join(category_rows),
        "<<RESULT_ROWS>>": "\n".join(result_rows),
        "<<MEANPROBES>>": fmt(
            statistics.mean(x["candidate_evaluations"] for x in rows), 1
        ),
        "<<MINPROBES>>": str(min(x["candidate_evaluations"] for x in rows)),
        "<<MAXPROBES>>": str(max(x["candidate_evaluations"] for x in rows)),
        "<<MEANFOLDTIME>>": fmt(statistics.mean(fold_times), 3),
    }
    for key, val in replacements.items():
        assert key in tex, key
        tex = tex.replace(key, val)
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT / "ADMM_Tune1_Validation_Report.tex"
    target.write_text(tex)
    csvpath = REPORT / "ADMM_25_Frozen_Selections.csv"
    with csvpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (REPORT / "ADMM_25_Category_Summary.json").write_text(
        json.dumps(
            {
                "total_candidates": 420,
                "total_fold_runs": 1680,
                "status": "completed_validation_tuning_only",
                "test45_status": "not executed",
                "gs_status": "not started",
                "mean_selected_ssdu_mse": statistics.mean(risks),
                "sd_selected_ssdu_mse": statistics.stdev(risks),
                "mean_fold_wall_seconds": statistics.mean(fold_times),
                "source_admm_sha256": digest(source_admm),
                "all_25": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print("REPORT_TEX_READY", target, "BYTES", target.stat().st_size)
    print(
        "SUMMARY",
        {
            "mean_ssdu": statistics.mean(risks),
            "sd_ssdu": statistics.stdev(risks),
            "mean_fold_seconds": statistics.mean(fold_times),
            "total_candidates": 420,
            "total_folds": 1680,
        },
    )


def main():
    rows, times = load()
    generate_figures(rows)
    build_tex(rows, times)


if __name__ == "__main__":
    main()
