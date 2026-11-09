"""Generate the completed random-one-per-category PnP test45 LaTeX report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from scripts.experiments.pcm.tune1_ternary_20261109.random_one_analysis import (
    ROOT,
    OUTPUT,
    METHODS,
    CATEGORIES,
    METRICS,
)

REPORT = ROOT / "report"
NAME = "PCM_Test45_Random_Tune1_Comparison"


def metric_cell(s, met, digits):
    d = s[met]
    return f'{d["mean"]:.{digits}f} $\\pm$ {d["std"]:.{digits}f}'


def make():
    summary = json.loads((ROOT / "analysis/test45_10_methods_summary.json").read_text())
    selections = json.loads((ROOT / "selection.json").read_text())
    paired = json.loads((ROOT / "analysis/paired_method_differences.json").read_text())
    with (ROOT / "analysis/test45_10_methods_per_image.csv").open(newline="") as f:
        allcases = list(csv.DictReader(f))
    assert len(allcases) == 45 and len(METHODS) == 10
    assert selections["methods_for_comparison"] == [m for m, _ in METHODS]
    assert selections["excluded_methods"] == ["pnp_admm_oracle", "gs_pnp_expanded5"]
    for method in ("admm", "gs"):
        assert len(selections["selected_calibrations"][method]) == 5

    overall = []
    for m, label in METHODS:
        values = summary["ALL"][m]
        overall.append(
            label
            + " & "
            + metric_cell(values, "psnr", 2)
            + " & "
            + metric_cell(values, "ssim", 4)
            + " & "
            + metric_cell(values, "haarpsi", 4)
            + " & "
            + f'{values["mse"]["mean"]:.3e} $\\pm$ {values["mse"]["std"]:.3e}'
            + r" \\"
        )
    selectionrows = []
    for c in CATEGORIES:
        a = selections["selected_calibrations"]["admm"][c]
        g = selections["selected_calibrations"]["gs"][c]
        ap = a["frozen_config"]
        gp = g["frozen_config"]
        selectionrows.append(
            f"{c} & {a['calibration_sample_id']:04d} & "
            f"{ap['rho']:.5f} & {ap['sigma']:.5f} & "
            f"{a['winner_measurement_ssdu_mse']:.7f} & "
            f"{g['calibration_sample_id']:04d} & "
            f"{gp['lambda']:.5f} & {gp['sigma']:.5f} & {gp['tau0']:.3f} & "
            f"{g['winner_measurement_ssdu_mse']:.7f}" + r" \\"
        )
    categories = []
    pairedcat = []
    for c in CATEGORIES:
        categories.append(
            c
            + " & "
            + " & ".join(f'{summary[c][m]["psnr"]["mean"]:.2f}' for m, _ in METHODS)
            + r" \\"
        )
        aold = summary[c]["pnp_admm_ssdu5"]["psnr"]["mean"]
        anew = summary[c]["pnp_admm_tune1"]["psnr"]["mean"]
        gold = summary[c]["gs_pnp_ssdu5"]["psnr"]["mean"]
        gnew = summary[c]["gs_pnp_tune1"]["psnr"]["mean"]
        pairedcat.append(
            f"{c} & {aold:.2f} & {anew:.2f} & {anew-aold:+.2f} & "
            f"{gold:.2f} & {gnew:.2f} & {gnew-gold:+.2f}" + r" \\"
        )
    runtimes = {}
    cases = {}
    for method in ("admm", "gs"):
        times = []
        for c in CATEGORIES:
            id_ = selections["selected_calibrations"][method][c][
                "calibration_sample_id"
            ]
            for x in allcases:
                if x["category"] != c:
                    continue
                source = (
                    ROOT
                    / "recon"
                    / method
                    / c
                    / "records"
                    / f"{id_:04d}"
                    / f'{int(x["sample_id"]):04d}.json'
                )
                rec = json.loads(source.read_text())
                assert (
                    rec["frozen_selection_sha256"]
                    == selections["selected_calibrations"][method][c][
                        "selection_file_sha256"
                    ]
                )
                times.append(float(rec["wall_seconds"]))
        assert len(times) == 45
        runtimes[method] = {
            "mean": statistics.mean(times),
            "sd": statistics.stdev(times),
            "total": sum(times),
            "n": len(times),
        }
    a = paired["pnp_admm_tune1_minus_pnp_admm_ssdu5"]["psnr"]
    g = paired["gs_pnp_tune1_minus_gs_pnp_ssdu5"]["psnr"]
    results = (
        f"On 45 matched test images, the randomly chosen PnP-ADMM Tune-1 "
        f'configuration averaged {summary["ALL"]["pnp_admm_tune1"]["psnr"]["mean"]:.3f} dB, '
        f'compared with {summary["ALL"]["pnp_admm_ssdu5"]["psnr"]["mean"]:.3f} dB '
        f'for PnP-ADMM-tuned-5 (paired difference {a["mean"]:+.3f} dB). '
        f'GS-PnP Tune-1 averaged {summary["ALL"]["gs_pnp_tune1"]["psnr"]["mean"]:.3f} dB, '
        f'compared with {summary["ALL"]["gs_pnp_ssdu5"]["psnr"]["mean"]:.3f} dB '
        f'for GS-PnP-tuned-5 (paired difference {g["mean"]:+.3f} dB).'
    )
    galleries = [ROOT / "analysis/galleries" / f"{c}_9_Pages.pdf" for c in CATEGORIES]
    gallery_complete = all(p.exists() for p in galleries)
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,xcolor,hyperref,enumitem,fancyhdr,float,pdfpages}
\hypersetup{colorlinks=true,urlcolor=blue!60!black,linkcolor=blue!60!black}
\setlength{\parindent}{0pt}\setlength{\parskip}{.45em}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=1.5em,itemsep=.1em}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small PCM / Random Tune-1 / test set 45}
\fancyhead[R]{\small 9 November 2026}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\begin{document}
{\LARGE\bfseries Random category-specific Tune-1 PnP for PCM}\\[4pt]
{\large Matched test set 45, 20\% Hadamard sampling, NPL-default noise}\\[5pt]
{\color{gray} Completed reconstruction experiment, version 1.0, 9 November 2026}\\[3pt]
{\color{gray} Ten requested methods; five random calibration choices per PnP family}
\par\medskip\hrule

\section{Executive summary}
We evaluate practical category-specific Tune-1 parameters
selected \emph{without ground truth} from noisy validation data.
For each of PnP-ADMM and GS-PnP, we draw one of five frozen
calibration configurations \emph{independently} for each device
category. These choices are frozen before test metrics are examined.
The same selected configuration is applied to all nine test images
from that category. Test set 45 is a fixed stratified subset
(9 images per category) of the original test set 450
(90 images per category).

All ten requested methods use the \emph{same frozen noisy
measurement and Hadamard mask for each test image},
with 20\% multilevel sampling, $J=8$, coarse $J_c=6$ and
the original NPL-default PCM noise realisation.
The two new methods contribute 45 reconstructions each,
with no additional parameter search on test data.

\textbf{Main result:} <<RESULTS>>

We report PSNR, SSIM, HaarPSI and MSE on raw (unclipped)
reconstructions. These are \emph{retrospective} measurements:
the full test set has been examined in related earlier studies,
and one random draw alone cannot establish the variability of
Tune-1 performance across calibration choices.

\section{Random selection and experimental protocol}
All five validation configurations per category were already
selected separately for both PnP families using four-fold
SSDU measurement-prediction loss and log-space ternary search.
The 25 ADMM configurations and 25 GS configurations were
\emph{frozen before this new comparison}.

We made ten independent uniform draws, one per
(category, PnP family), with deterministic random seed
\texttt{PCM-test45-one-calibration-2026-11-09-v1}.
The selection file stores the five eligible calibration IDs
in sorted order, selected index, exact frozen parameter values
and each source file's SHA256. No image PSNR or category-level
reconstruction metric informed the random choice.

\begin{table}[H]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
& \multicolumn{4}{c}{\textbf{ADMM Tune-1}} &
\multicolumn{5}{c}{\textbf{GS-PnP Tune-1}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-10}
Category & Calibration ID & $\rho$ & $\sigma$ & SSDU risk &
Calibration ID & $\lambda$ & $\sigma$ & $\tau_0$ & SSDU risk\\
\midrule
<<SELECTION_ROWS>>
\bottomrule
\end{tabular}}
\caption{One calibration configuration randomly drawn from five
validation-tuned choices in each category. Each SSDU risk is the
held-out measurement-prediction error for that individual
validation acquisition, not an image-space test MSE.}
\end{table}

New reconstructions: PnP-ADMM on RTX 4070 and GS-PnP
on RTX 4090, each with unchanged pretrained denoiser weights
and original solver stopping rules. Model and measurement
normalisation were not altered between tuning and testing.
The test45 measurement and pattern SHA256s were checked against
the historical full test set 450 archive.

\section{Methods compared}
We report exactly ten method labels:
zero-filled, SPGL1-BP, SPGL1-BPDN, PnP-ADMM-tuned-5,
GS-PnP-tuned-oracle, scalar TV, UTV-C60,
GS-PnP-tuned-5, PnP-ADMM Tune-1 and GS-PnP Tune-1.
The supervised-validation-selected \emph{PnP-ADMM oracle label}
is intentionally omitted. PnP-ADMM-tuned-5 is shown once,
even though the original SSDU-tuned-5 settings coincided with
that reference. The GS-PnP expanded SSDU-5 method is also
intentionally omitted.

The eight historical reference scores are taken from the
same fixed 450-image benchmark after checking each
category, image ID, noisy measurement hash and sampling
mask hash. Their established hyperparameters are unchanged.
Missing historical raw-image arrays were rerun solely for
visualisation under those original numerical settings.
SPGL1-BP can show small differences in a later numerical
replay: the archived benchmark metrics remain authoritative
for the main comparison, and visual captions for BP
use metrics calculated on the displayed replayed array.
No baseline parameter was adjusted to fit test results.

\section{Quantitative results}
\subsection*{Overall metrics, 45 unique test images}
\begin{table}[H]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
Method & PSNR (dB) & SSIM & HaarPSI & MSE\\
\midrule
<<OVERALL_ROWS>>
\bottomrule
\end{tabular}}
\caption{Means $\pm$ sample standard deviations over 45
\emph{distinct} images. This does not count the ten random
calibration draws as additional test images.}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=.85\linewidth]{../analysis/figures/All_45_10_Method_Mean_SD.pdf}
\caption{Overall PSNR, SSIM, HaarPSI and MSE means
with sample standard-deviation whiskers.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=.85\linewidth]{../analysis/figures/All_45_10_Method_Boxplots.pdf}
\caption{Distribution of 45 paired test-image scores;
outliers are hidden in the plot, but used in all means.}
\end{figure}

\subsection*{Per-category PSNR}
\begin{table}[H]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
Category & ZF & SPGL1-BP & BPDN & ADMM-5 & GS oracle &
TV & C60 & GS-5 & ADMM Tune-1 & GS Tune-1\\
\midrule
<<CATEGORY_ROWS>>
\bottomrule
\end{tabular}}
\caption{Mean PSNR in dB for nine fixed test images per
category. Companion figures report SSIM, HaarPSI and MSE,
as well as sample SD.}
\end{table}

\subsection*{Tune-1 versus the previous tuned-5 baseline}
\begin{table}[H]
\centering\scriptsize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{ADMM PSNR (dB)} &
\multicolumn{3}{c}{GS-PnP PSNR (dB)}\\
Category & Tuned-5 & Tune-1 & Difference &
Tuned-5 & Tune-1 & Difference\\
\midrule
<<PAIRED_CATEGORY>>
\bottomrule
\end{tabular}
\caption{Paired category mean PSNR differences,
Tune-1 minus the existing SSDU tuned-5 configuration.
Positive means Tune-1 obtained higher PSNR on average
for that category's nine test images.}
\end{table}

\subsection*{Reconstruction runtime}
For the 45 newly reconstructed test images,
the observed mean $\pm$ SD was
<<TIME_ADMM>> seconds/image for ADMM Tune-1 and
<<TIME_GS>> seconds/image for GS-PnP Tune-1.
These exclude the earlier calibration-search cost
and do not constitute a controlled speed comparison
because the algorithms ran on different GPUs.

\section{Visual comparisons and reproducibility}
The companion 45-page PDF shows \emph{every} image in test set 45,
with ground truth and exactly the ten methods specified above.
Each method panel displays PSNR, SSIM, HaarPSI and MSE.
The display uses the AFM-hot colormap capped at 0.8,
with the same image intensity range $[0,1]$ for every panel.

Files retained: a frozen random-selection JSON;
the original test45 image-ID and acquisition-hash manifests;
45 individual new ADMM reconstructions and 45 new GS-PnP
reconstructions with metrics and solver diagnostics;
archived benchmark scores and replayed visual arrays;
a 45-row ten-method metrics CSV;
per-category and global JSON summaries;
bar/boxplot PDFs and five nine-page category visual PDFs.
All experiment files are versioned separately from prior
tune-1 calibration studies and test set 450.

\section{Conclusion}
<<RESULTS>>

A single category-specific calibration can yield reasonable
reconstruction quality, but our results represent
\emph{one uniform random choice per category and method}.
They do not establish that one calibration image is enough
in general, that the joint PnP parameter objective is
globally unimodal, or that SSDU loss predicts the best
image-space reconstruction. The test45 comparisons must not
be used to repeatedly retune parameters.

\paragraph{Primary references}
Yaman et al. (2020), acquired-measurement splitting for SSDU:
\url{https://doi.org/10.1002/mrm.28378}.
Hurault, Leclaire and Papadakis (2022),
GS-PnP algorithm and backtracking:
\url{https://openreview.net/pdf?id=fPhKeld3Okz}.

\IfFileExists{../analysis/galleries/Test45_All_10_Methods_45_Pages.pdf}{
\clearpage
\includepdf[pages={1,10,19,28,37},pagecommand={\thispagestyle{empty}},fitpaper=true]{../analysis/galleries/Test45_All_10_Methods_45_Pages.pdf}
}{}

\end{document}
"""
    for k, v in {
        "<<RESULTS>>": results,
        "<<OVERALL_ROWS>>": "\n".join(overall),
        "<<SELECTION_ROWS>>": "\n".join(selectionrows),
        "<<CATEGORY_ROWS>>": "\n".join(categories),
        "<<PAIRED_CATEGORY>>": "\n".join(pairedcat),
        "<<TIME_ADMM>>": f'{runtimes["admm"]["mean"]:.3f} $\\pm$ {runtimes["admm"]["sd"]:.3f}',
        "<<TIME_GS>>": f'{runtimes["gs"]["mean"]:.3f} $\\pm$ {runtimes["gs"]["sd"]:.3f}',
    }.items():
        assert k in tex, k
        tex = tex.replace(k, v)
    assert "<<" not in tex
    REPORT.mkdir(parents=True, exist_ok=True)
    dest = REPORT / (NAME + ".tex")
    dest.write_text(tex)
    (REPORT / "report_data.json").write_text(
        json.dumps(
            {
                "status": "test45_reconstruction_and_quantitative_comparison_complete",
                "selection_sha256": hashlib.sha256(
                    (ROOT / "selection.json").read_bytes()
                ).hexdigest(),
                "baseline_method_labels": [m for m, _ in METHODS],
                "excluded": ["pnp_admm_oracle", "gs_pnp_expanded5"],
                "new_test_reconstruction_count": 90,
                "unique_test_images": 45,
                "runtimes_seconds": runtimes,
                "paired": paired,
                "galleries_complete": gallery_complete,
                "metrics_sha256": hashlib.sha256(
                    (ROOT / "analysis/test45_10_methods_per_image.csv").read_bytes()
                ).hexdigest(),
                "no_test_parameter_search": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        "REPORT_READY",
        dest,
        dest.stat().st_size,
        "GALLERIES_COMPLETE",
        gallery_complete,
    )
    print("RESULTS", results)


if __name__ == "__main__":
    make()
