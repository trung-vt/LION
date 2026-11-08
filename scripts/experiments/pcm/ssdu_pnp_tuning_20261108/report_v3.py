"""Create v3 PCM SSDU report: tuned-5 only in main evaluation and five visual pages.

Existing v1 tuning-only and v2 full-11-method reports are preserved unchanged.
A preregistered GS higher-bound candidate grid is documented but NOT run.
"""
from __future__ import annotations

import csv
import json
import hashlib
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("/store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1")
REPORT = ROOT / "report"
TEST = ROOT / "test450/v1"
VIS = TEST / "visual5"
OUT = VIS / "figures"
GIT = Path("/home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108")
GSPLAN = (
    GIT
    / "scripts/experiments/pcm/ssdu_pnp_tuning_20261108/GS_PNP_EXPANDED_SSDU_SWEEP_PLAN.md"
)
GRID = (
    GIT
    / "scripts/experiments/pcm/ssdu_pnp_tuning_20261108/gs_expanded_5_candidate_grid.json"
)
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
METRICS = ("psnr", "ssim", "haarpsi", "mse")
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
)
assert len(METHODS) == 9


def histdata():
    p = TEST / "merged/all_450_11_method_metrics.csv"
    with p.open(newline="") as f:
        data = {
            (row["category"], int(row["sample_id"])): row for row in csv.DictReader(f)
        }
    assert len(data) == 450
    return data


def summarize(data):
    result = {}
    for cat in ("ALL",) + CATEGORIES:
        cases = [v for (c, s), v in data.items() if cat == "ALL" or c == cat]
        assert len(cases) == (450 if cat == "ALL" else 90)
        result[cat] = {}
        for key, label in METHODS:
            result[cat][key] = {}
            for metric in METRICS:
                values = np.asarray([float(v[f"{key}_{metric}"]) for v in cases])
                assert np.isfinite(values).all()
                result[cat][key][metric] = {
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "n": len(values),
                }
    return result


def render_plots(data, stats):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    labels = [m for k, m in METHODS]
    fig, ax = plt.subplots(2, 2, figsize=(16.0, 10.0), layout="constrained")
    for a, (metric, nice) in zip(
        ax.flat,
        (
            ("psnr", "PSNR (dB)"),
            ("ssim", "SSIM"),
            ("haarpsi", "HaarPSI"),
            ("mse", "MSE"),
        ),
    ):
        vals = [
            np.array([float(v[f"{key}_{metric}"]) for v in data.values()])
            for key, label in METHODS
        ]
        a.boxplot(vals, tick_labels=labels, showfliers=False)
        a.set_title(nice, weight="bold")
        a.tick_params(axis="x", labelrotation=35, labelsize=8.5)
        for tick in a.get_xticklabels():
            tick.set_ha("right")
        a.grid(axis="y", alpha=0.22)
        if metric == "mse":
            a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "450 images | 9 selected method labels | NPL-noisy 20% PCM", fontsize=16
    )
    fig.savefig(OUT / "9_methods_450_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)

    def means_grid(cat, pdf):
        fig, ax = plt.subplots(2, 2, figsize=(16.0, 10.0), layout="constrained")
        for a, (metric, nice) in zip(
            ax.flat,
            (
                ("psnr", "PSNR (dB)"),
                ("ssim", "SSIM"),
                ("haarpsi", "HaarPSI"),
                ("mse", "MSE"),
            ),
        ):
            y = [stats[cat][key][metric]["mean"] for key, label in METHODS]
            e = [stats[cat][key][metric]["sample_sd"] for key, label in METHODS]
            a.bar(range(9), y, yerr=e, capsize=3, alpha=0.82)
            a.set_xticks(range(9), labels, rotation=35, ha="right", fontsize=8.5)
            a.set_title(nice, weight="bold")
            a.grid(axis="y", alpha=0.22)
            if metric == "mse":
                a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        fig.suptitle(
            ("Overall 450 images" if cat == "ALL" else cat + ": 90 test images")
            + " | Mean +/- sample SD",
            fontsize=16,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(OUT / "9_methods_450_mean_sd.pdf") as pdf:
        means_grid("ALL", pdf)
    with PdfPages(OUT / "9_methods_by_category_mean_sd.pdf") as pdf:
        for cat in CATEGORIES:
            means_grid(cat, pdf)
    print("FIGURES_READY", *[p.name for p in OUT.glob("*.pdf")], flush=True)


def score_table(stats):
    r = []
    for key, name in METHODS:
        d = stats["ALL"][key]
        num = (
            lambda m, dig: f'{d[m]["mean"]:.{dig}f} $\\pm$ {d[m]["sample_sd"]:.{dig}f}'
        )
        val = f'{d["mse"]["mean"]:.3e} $\\pm$ {d["mse"]["sample_sd"]:.3e}'
        r.append(
            name
            + " & "
            + " & ".join((num("psnr", 2), num("ssim", 4), num("haarpsi", 4), val))
            + r" \\"
        )
    return "\n".join(r)


def scores_by_cat(stats):
    lines = []
    for cat in CATEGORIES:
        scores = [stats[cat][key]["psnr"]["mean"] for key, label in METHODS]
        lines.append(cat + " & " + " & ".join(f"{s:.2f}" for s in scores) + r" \\")
    return "\n".join(lines)


def make_report(data, stats):
    if not GSPLAN.exists() or not GRID.exists():
        raise RuntimeError("Expanded sweep must be predeclared before report")
    grid = json.loads(GRID.read_text())
    assert (
        grid["status"] == "frozen_proposed_grid_not_run"
        and len(grid["candidates"]) == 27
    )
    selection = json.loads((VIS / "selection.json").read_text())
    names = [(x["category"], int(x["sample_id"])) for x in selection["chosen_cases"]]
    assert names == [
        ("CIGS", 279),
        ("Organic", 331),
        ("Perovskite", 786),
        ("Reference", 279),
        ("Silicone", 370),
    ]
    assert all((VIS / "audits" / f"{c}_{s:04d}.json").is_file() for c, s in names)
    pdf = VIS / "PCM_5_Cases_9_Methods_with_Metrics.pdf"
    assert pdf.exists()

    archived = REPORT / "archive_test450_11methods_v2"
    archived.mkdir(parents=True, exist_ok=True)
    for ext in (".tex", ".pdf"):
        src = REPORT / ("SSDU_PnP_Tuning_Report" + ext)
        dest = archived / ("SSDU_PnP_Tuning_Report_v2_11_methods" + ext)
        if not dest.exists():
            shutil.copy2(src, dest)
        else:
            assert src.read_bytes() == dest.read_bytes()
    src = (archived / "SSDU_PnP_Tuning_Report_v2_11_methods.tex").read_text()
    start = src.index(r"\subsection*{Matched 450-image test: all eleven methods}")
    stop = src.index(r"\section{Conclusion}", start)
    assert start < stop
    new_content = r"""
\subsection*{Main 450-image comparison: tuned-5 only}

The 5-image and 50-image SSDU selections chose the \emph{same} parameter
configuration for each PnP algorithm. Their 450-image test metrics are therefore
identical. To avoid duplicated results in the primary comparison, we retain only
\emph{PnP-ADMM-tuned-5} and \emph{GS-PnP-tuned-5} alongside the seven original
baselines. The separately frozen tuned-50 selections and the previous full
11-label comparison are preserved in the version-2 archive.

All methods use the exact same fixed 20\% NPL-noisy test acquisition per
image, with all 450 measurement and mask SHA256 hashes checked. The supervised
validation-tuned Step-6 GS-PnP oracle uses its \emph{paper-final $z$} output.
No test-based parameter selection was performed.

\begin{table}[H]
\centering\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Method}&\textbf{PSNR (dB)}&\textbf{SSIM}&\textbf{HaarPSI}&\textbf{MSE}\\
\midrule
@@NINESTATS@@
\bottomrule
\end{tabular}}
\caption{450-image mean $\pm$ sample standard deviation on raw
unclipped reconstructions, for nine retained method labels.}
\label{tab:nine_methods}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=.93\linewidth]{../test450/v1/visual5/figures/9_methods_450_boxplots.pdf}
\caption{Box plots of the paired 450-image PSNR, SSIM, HaarPSI and MSE
results for the seven baselines and two tuned-5 methods.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.93\linewidth]{../test450/v1/visual5/figures/9_methods_450_mean_sd.pdf}
\caption{Overall 450-image mean and sample standard deviation for the nine
retained method labels.}
\end{figure}

\subsection*{What does agreement with the oracle mean?}

\textbf{PnP-ADMM:} SSDU-tuned-5 chooses exactly the same parameters
($\rho=0.5,\sigma=0.075$) and stopping rule as the earlier supervised
validation-tuned ADMM benchmark. Therefore both methods produce the
same 450-image scores: mean PSNR 35.754 dB, SSIM 0.8970 and HaarPSI 0.8814.
This shows that SSDU reaches \emph{that} oracle-selected benchmark. It
does \emph{not} prove that these hyperparameters are globally optimal
over untested configurations.

\textbf{GS-PnP:} SSDU-tuned-5 gives mean PSNR 32.590 dB, compared with
35.110 dB for the frozen supervised-validation Step-6 GS-PnP oracle.
The paired PSNR difference is -2.519 dB, with sample SD 1.890 dB.
Thus the current measurement-only SSDU criterion did not recover
the same image-space quality for GS-PnP. This is an observed mismatch,
not proof of a specific cause.

\subsection*{Device-category scores}

\begin{table}[H]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
\textbf{Category}&\textbf{ZF}&\textbf{SPGL1}&\textbf{BPDN}&
\textbf{ADMM oracle}&\textbf{GS oracle}&\textbf{TV}&\textbf{C60}&
\textbf{ADMM-5}&\textbf{GS-5}\\
\midrule
@@CAT9@@
\bottomrule
\end{tabular}}
\caption{Per-category mean PSNR (dB) over 90 test images per category.
The complete 450-image row-level metrics are retained separately as CSV.}
\end{table}

Five-page per-category mean/SD plots are available in the companion output:
\texttt{test450/v1/visual5/figures/9\_methods\_by\_category\_mean\_sd.pdf}.

\subsection*{Five visual comparisons with per-panel metrics}

We fixed one case per device category before viewing reconstruction scores.
CIGS image 0279 was requested explicitly; the other four were drawn with
a recorded fixed random seed from the ten already preselected 450-test
visualisation cases per category:

\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Category}&\textbf{Image ID}\\
\midrule
CIGS&0279 (specified)\\
Organic&0331\\
Perovskite&0786\\
Reference&0279\\
Silicone&0370\\
\bottomrule
\end{tabular}
\end{center}

Each page shows ground truth and all nine retained methods, in a
two-by-five panel layout. Every reconstruction uses the same stored
20\% noisy PCM acquisition. The display uses AFM-hot capped at 0.8,
with fixed image intensity $[0,1]$. Below each method label we give
PSNR, SSIM, HaarPSI and MSE, computed on \emph{raw} image values.
The historical BP, BPDN, TV, C60 and ADMM examples were replayed
against their stored metrics; all numerical checks passed.

The five landscape comparison pages are embedded immediately below.
They also remain in a standalone PDF for detailed inspection.

\clearpage
\includepdf[pages=-,pagecommand={}]{../test450/v1/visual5/PCM_5_Cases_9_Methods_with_Metrics.pdf}

\section*{Next GS-PnP study: expanded blind SSDU search (planned)}

The original SSDU GS-PnP winner was at the \emph{maximum} of the
candidate grid along \emph{all three} parameter dimensions:
$\lambda=0.1$, denoiser $\sigma=0.075$ and $\tau_0=100/3$.
Therefore the original parameter search did not demonstrate an
interior minimum. We have predeclared a wider Cartesian grid:

\begin{center}
\begin{tabular}{@{}lp{100mm}@{}}
\toprule
\textbf{Parameter}&\textbf{Candidate values}\\
\midrule
$\lambda$&0.10, 0.20, 0.40\\
Denoiser $\sigma$&0.075, 0.10, 0.15\\
Initial $\tau_0$&33.3333, 50, 75\\
\bottomrule
\end{tabular}
\end{center}

The $3\times3\times3=27$ triples include the old winner and 26
previously untested combinations. Apply the exact original five
validation images and four held-out folds each: 540 GS-PnP
reconstruction calls. Keep the DRUNet model, loss function, operator
normalisation and GS stopping/backtracking rules unchanged.
Independently vary $\lambda$, $\sigma$ and $\tau_0$.

Before running, record all candidates and numerical instability
policies. Inspect accepted step sizes, $\lambda\tau$, line-search
rejections, stopping reason and held-out MSE. Select the winner from
\emph{validation measurement loss only}, freeze it, and report whether
the new minimum again lies on an upper boundary.

The extended sweep is a \textbf{future experiment and has NOT been
run yet}. Increasing parameter magnitudes does not necessarily improve
PSNR or preserve details, particularly when PCM measurement noise
is correlated. Further evaluation on the previously inspected 450
test cases must be labelled retrospective and cannot be used to
repeatedly tune parameters.

"""
    new_content = new_content.replace("@@NINESTATS@@", score_table(stats))
    new_content = new_content.replace("@@CAT9@@", scores_by_cat(stats))
    assert "@@" not in new_content
    src = src[:start] + new_content + "\n" + src[stop:]
    # Remove the old conclusion, which discussed four labels and requested
    # a future test, and replace it with a concise, data-grounded conclusion.
    start = src.index(r"\section{Conclusion}")
    end = src.index(r"\renewcommand{\refname}{Primary references}", start)
    conclusion = r"""\section{Conclusion}

The SSDU-style measurement-only tuning study is complete.
Both 5-image and 50-image validation sets selected the same
parameter configuration within each PnP algorithm's \emph{tested grid}.
Thus five images were sufficient for selection stability in this
particular protocol. We retain the tuned-5 configurations for subsequent
comparisons, while preserving all tuned-50 results and selections in
the original study archive.

For PnP-ADMM, ground-truth-free tuned-5 matched the old
supervised-validation-tuned benchmark \emph{exactly}: both use
$\rho=0.5$, $\sigma=0.075$ and the same stopping criterion, yielding
mean PSNR 35.754 dB over 450 matched test acquisitions.
This is agreement with the previous benchmark, not proof of globally
optimal reconstruction parameters.

For GS-PnP, tuned-5 gives mean PSNR 32.590 dB, compared with
35.110 dB for the Step-6 supervised-validation oracle (paired
difference -2.519 dB). The selected $\lambda$, denoiser $\sigma$ and
initial $\tau_0$ are all at the top of the original candidate grid,
so the grid should be expanded using \emph{validation-only} SSDU
scoring. The proposed 27-configuration higher-value sweep is written
and frozen but has \emph{not} been run.

The five case comparisons with raw-image metrics give qualitative
context; they cannot establish the preservation of thin lines without
further targeted evaluation. Temporally correlated noise and the
undersampled Hadamard nullspace may limit how well SSDU prediction
loss tracks image quality. Since these 450 test images have already
been inspected in related research, the evaluation is retrospective.

"""
    src = src[:start] + conclusion + src[end:]
    src = src.replace("Version 2.0", "Version 3.0")
    src = src.replace(
        "The agreed \\textbf{450-image test-set evaluation is complete}; Section~5 compares the frozen choices",
        "The agreed \\textbf{450-image test-set evaluation is complete}; Section~5 compares the tuned-5 choices",
    )
    # Make the primary emphasis unambiguous in the executive summary.
    needle = "The five-image and 50-image SSDU selections chose"
    # Not all v2 variants contain this sentence, so no assertion here.
    src = src.replace(
        "we retain four distinct selections",
        "we documented four separately frozen selections",
    )
    # Preserve stage history but correct future tense in step table if present.
    src = src.replace(
        "Run all four SSDU methods and seven existing baselines",
        "Compare two tuned-5 SSDU methods and seven existing baselines",
    )
    src = src.replace(
        "report the resulting image quality",
        "report image quality",
    )
    preamble = r"\usepackage{float}"
    assert preamble in src
    src = src.replace(preamble, preamble + "\n" + r"\usepackage{pdfpages}")
    assert src.count(r"\includepdf[pages=-,pagecommand={}]") == 1
    target = REPORT / "SSDU_PnP_Tuning_Report_v3.tex"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(src)
    (VIS / "summary_9_methods.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("REPORT_V3_TEX_READY", target, target.stat().st_size)
    print(
        "METHOD9_PSNR",
        [(k, round(stats["ALL"][k]["psnr"]["mean"], 3)) for k, n in METHODS],
    )


def main():
    data = histdata()
    stats = summarize(data)
    render_plots(data, stats)
    make_report(data, stats)


if __name__ == "__main__":
    main()
