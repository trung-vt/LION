"""Update dedicated GS-PnP SSDU report after freezing and retrospective 450 test.

Preserves the initial 3-page pre-run report in report/pretest_prepared_v1/.
No winner can be selected from test metrics. Source and PDF are versioned.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import numpy as np

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import (
    ROOT,
    RESULTS,
    REPORT,
    GRID_FILE,
    PREPARED,
    sha_file,
    get_grid,
    status,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs_analysis import (
    OUT,
    METHODS,
    CATEGORIES,
    METRICS,
    TEST,
)

NAME = "GS_PnP_Expanded_SSDU_Tuning_Full_Report"


def get_data():
    selected = json.loads((ROOT / "frozen_selection.json").read_text())
    assert selected["status"] == "frozen_before_any_additional_test_evaluation"
    assert selected["no_ground_truth_used"] and selected["no_test_metrics_used"]
    assert (OUT / "analysis_complete.json").exists()
    summary = json.loads((OUT / "metric_summary.json").read_text())
    paired = json.loads((OUT / "paired_differences.json").read_text())
    times = json.loads((OUT / "runtime_summary.json").read_text())
    grid, sha = get_grid()
    assert selected["grid_sha256"] == sha
    files = list((RESULTS / "candidates").glob("*/summary.json"))
    assert (
        len(files) >= 18
    )  # numerical invalid candidates allowed, subject to selection policy
    cases = list((TEST / "records").glob("*/*.json"))
    assert len(cases) == 450
    assert len(METHODS) == 10
    return selected, summary, paired, times, grid


def latex_stats(summary):
    rows = []
    for name, label in METHODS:
        s = summary["ALL"][name]
        vals = [
            f'{s["psnr"]["mean"]:.2f} $\\pm$ {s["psnr"]["sd"]:.2f}',
            f'{s["ssim"]["mean"]:.4f} $\\pm$ {s["ssim"]["sd"]:.4f}',
            f'{s["haarpsi"]["mean"]:.4f} $\\pm$ {s["haarpsi"]["sd"]:.4f}',
            f'{s["mse"]["mean"]:.2e} $\\pm$ {s["mse"]["sd"]:.2e}',
        ]
        rows.append(label + " & " + " & ".join(vals) + r" \\")
    return "\n".join(rows)


def by_category(summary):
    rows = []
    for cat in CATEGORIES:
        keys = (
            "zero_filled",
            "spgl1_bp",
            "spgl1_bpdn",
            "pnp_admm_oracle",
            "gs_pnp_oracle",
            "tv",
            "utv_c60",
            "gs_pnp_ssdu5",
            "gs_pnp_expanded5",
        )
        rows.append(
            cat
            + " & "
            + " & ".join(f'{summary[cat][m]["psnr"]["mean"]:.2f}' for m in keys)
            + r" \\"
        )
    return "\n".join(rows)


def top_candidates(selected):
    rows = []
    for obj in selected["ranked_valid_candidates"][:10]:
        p = obj["parameters"]
        rows.append(
            rf'\texttt{{{obj["candidate_id"].replace("_",r"\_")}}} & '
            f'{p["lambda"]:.3f} & {p["sigma"]:.3f} & {p["tau0"]:.3f} & '
            f'{obj["SSDU_MSE"]:.8f}' + r" \\"
        )
    return "\n".join(rows)


def numerical_report(selected):
    grid, _ = get_grid()
    diagnostics = {}
    for cand in grid:
        path = RESULTS / "candidates" / cand["candidate_id"]
        detail = []
        for rec_path in path.glob("*.json"):
            if rec_path.name in (
                "summary.json",
                "invalid_candidate.json",
            ) or rec_path.name.endswith(".failure.json"):
                continue
            record = json.loads(rec_path.read_text())
            if "folds" not in record:
                continue
            for fold in record["folds"]:
                d = fold["solver"]
                detail.append(
                    {
                        "iterations": float(d["accepted_iterations"]),
                        "rejections": float(d["rejections"]),
                        "lambda_tau": float(d["lambda_tau"]),
                        "time": float(d["wall_seconds_with_final_gradient"]),
                        "backtracking_exhausted": bool(d["backtracking_exhausted"]),
                    }
                )
        if detail:
            diagnostics[cand["candidate_id"]] = {
                "folds": len(detail),
                "mean_accepted_iterations": float(
                    np.mean([d["iterations"] for d in detail])
                ),
                "mean_rejections": float(np.mean([d["rejections"] for d in detail])),
                "mean_lambda_tau": float(np.mean([d["lambda_tau"] for d in detail])),
                "mean_seconds": float(np.mean([d["time"] for d in detail])),
                "frac_backtracking_exhausted": float(
                    np.mean([d["backtracking_exhausted"] for d in detail])
                ),
            }
    (OUT / "sweep_numerical_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    ids = [selected["winner"]["candidate_id"], "gs_exp_001"]
    rows = []
    for key in ids:
        d = diagnostics[key]
        rows.append(
            rf'\texttt{{{key.replace("_",r"\_")}}} & {d["folds"]} & '
            f'{d["mean_accepted_iterations"]:.1f} & '
            f'{d["mean_rejections"]:.1f} & '
            f'{d["mean_lambda_tau"]:.3f} & '
            f'{d["mean_seconds"]:.3f}' + r" \\"
        )
    return "\n".join(rows), diagnostics


def make_report():
    selected, summary, paired, times, grid = get_data()
    old = REPORT / "GS_PnP_Expanded_SSDU_Tuning_Report.tex"
    oldpdf = REPORT / "GS_PnP_Expanded_SSDU_Tuning_Report.pdf"
    assert old.is_file() and oldpdf.is_file()
    archive = REPORT / "pretest_prepared_v1"
    archive.mkdir(exist_ok=True)
    for oldname, newname in (
        (old, "GS_PnP_Expanded_SSDU_Tuning_Report_pretest.tex"),
        (oldpdf, "GS_PnP_Expanded_SSDU_Tuning_Report_pretest.pdf"),
    ):
        target = archive / newname
        if not target.exists():
            shutil.copy2(oldname, target)
    # Use the pre-test source as a stable base for every regeneration.
    source = (archive / "GS_PnP_Expanded_SSDU_Tuning_Report_pretest.tex").read_text()
    assert source.count(r"\section{Conclusion}") == 1
    initial_category = json.loads(
        (RESULTS / "candidates/gs_exp_001/summary.json").read_text()
    )["category_means"]
    expanded_category = json.loads(
        (
            RESULTS / "candidates" / selected["winner"]["candidate_id"] / "summary.json"
        ).read_text()
    )["category_means"]
    categories = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
    category_validation_rows = "\n".join(
        f"{cat} & {initial_category[cat]:.8f} & {expanded_category[cat]:.8f} "
        f"& {expanded_category[cat]-initial_category[cat]:+.8f} " + r"\\"
        for cat in categories
    )
    diag_rows, diagnostics = numerical_report(selected)
    winner = selected["winner"]
    p = winner["parameters"]
    base = selected["ranked_valid_candidates"]
    original = next(x for x in base if x["candidate_id"] == "gs_exp_001")
    risk_delta = winner["SSDU_MSE"] - original["SSDU_MSE"]
    improvement = -risk_delta / original["SSDU_MSE"] * 100
    diffold = paired["expanded_minus_gs_pnp_ssdu5"]
    difforacle = paired["expanded_minus_gs_pnp_oracle"]
    oldpsnr = summary["ALL"]["gs_pnp_ssdu5"]["psnr"]["mean"]
    oraclepsnr = summary["ALL"]["gs_pnp_oracle"]["psnr"]["mean"]
    newpsnr = summary["ALL"]["gs_pnp_expanded5"]["psnr"]["mean"]
    boundaries = selected["boundary_flag"]
    invalid = selected["invalid_candidates"]
    invalid_rows = []
    for candidate_id in invalid:
        root = RESULTS / "candidates" / candidate_id
        info = json.loads((root / "invalid_candidate.json").read_text())
        for name in info["failed_case_files"]:
            err = json.loads(Path(name).read_text())
            category_case = err["case"].replace("_", r"\_")
            label = candidate_id.replace("_", r"\_")
            errtext = err["error"].replace("_", r"\_")
            invalid_rows.append(
                r"\texttt{"
                + label
                + "} & "
                + r"\texttt{"
                + category_case
                + "} & "
                + errtext
                + r" \\"
            )
    invalid_table = "\n".join(invalid_rows)
    d = diagnostics[winner["candidate_id"]]
    pairedtext = (
        f"The expanded configuration gives mean PSNR {newpsnr:.3f} dB, "
        f"compared with {oldpsnr:.3f} dB for the original SSDU-5 GS-PnP "
        f"and {oraclepsnr:.3f} dB for the supervised-validation oracle. "
        f"The paired expanded-minus-original PSNR difference is "
        f'{diffold["psnr"]["mean"]:+.3f} dB '
        f'(sample SD {diffold["psnr"]["sd"]:.3f} dB), while '
        f"the difference against the oracle is "
        f'{difforacle["psnr"]["mean"]:+.3f} dB. '
        "All test parameters were frozen before scoring these images."
    )

    # A post-freezing, test-only diagnostic. No hyperparameter adjustment.
    with (OUT / "all_450_10_methods.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 450
    deltas = np.array(
        [
            float(x["gs_pnp_expanded5_psnr"]) - float(x["gs_pnp_ssdu5_psnr"])
            for x in records
        ],
        dtype=np.float64,
    )
    better = int(np.count_nonzero(deltas > 0))
    large_loss = int(np.count_nonzero(deltas < -1))
    median = float(np.median(deltas))
    parity_path = TEST / "same_gpu_probe.json"
    assert parity_path.exists()
    parity = json.loads(parity_path.read_text())
    assert parity["no_test_driven_tuning"] is True
    parity_delta = max(abs(float(x["difference_db"])) for x in parity["cases"])
    distribution = (
        f"The expanded configuration gave higher PSNR on {better} of 450 "
        f"matched images, with a median paired change of {median:+.3f} dB. "
        f"However, {large_loss} images lost over one dB, producing the "
        "negative mean paired change despite most cases improving. "
        f"An eight-case hardware parity audit on the RTX 4090, including "
        "large gain and loss cases, reproduced the earlier RTX 4070 original "
        f"configuration PSNR to within {parity_delta:.5f} dB. "
        "Thus the hardware difference does not explain those selected "
        "outliers; full 450-image hardware parity has not been tested."
    )
    latex = r"""
\subsection*{Expanded sweep: measured validation outcome}
The 27 candidates were predeclared before the GPU evaluation.
The original control risk was <<OLDLOSS>>, while the
minimum validation-only held-out risk was
<<BESTLOSS>> for \texttt{<<WINNER>>}.
This is a <<IMPROVEMENT>> percent reduction in the SSDU loss
relative to the original control. It is a difference in
\emph{measurement prediction error}, not necessarily image
reconstruction quality.

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Category & Original SSDU MSE & Expanded SSDU MSE & Change\\
\midrule
<<CATEGORY_VALIDATION>>
\bottomrule
\end{tabular}
\caption{Per-category validation measurement prediction errors: one
image and four held-out folds per category.}
\end{table}

The frozen winning parameters are
\[
\lambda=<<LAM>>,\qquad
\sigma=<<SIG>>,\qquad
\tau_0=<<TAU>>.
\]
The selection used only the original five validation images
and their fixed holdout masks. No test-image ground truth or
previous test metrics influenced the selection.
There were <<INVALID>> numerically invalid configurations
out of 27. The winner's upper-bound flags are:
lambda: <<FLAGLAM>>, sigma: <<FLAGSIG>>, tau0: <<FLAGTAU>>.

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lll@{}}
\toprule
Invalid candidate & Validation case & Failure reason\\
\midrule
<<INVALIDDETAILS>>
\bottomrule
\end{tabular}
\caption{Numerically invalid higher-value candidates; they were
excluded rather than silently assigned a favourable SSDU loss.}
\end{table}

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Candidate}&$\lambda$&$\sigma$&$\tau_0$&\textbf{SSDU risk}\\
\midrule
<<TOPRANK>>
\bottomrule
\end{tabular}
\caption{Top ten valid candidates ranked by frozen five-image SSDU loss.}
\end{table}

\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Config}&\textbf{Folds}&\textbf{Accepted iters}&
\textbf{Rejections}&$\boldsymbol{\lambda\tau}$&\textbf{s/solver}\\
\midrule
<<DIAGNOSTICS>>
\bottomrule
\end{tabular}
\caption{Selected versus original GS control. Means over 20 validation folds.
A faster method may require fewer accepted iterations or less work
per iteration/backtracking.}
\end{table}

\clearpage
\subsection*{Retrospective 450-image PCM evaluation}
After freezing the expanded SSDU winner, we applied it to the
exact same stored fixed 20 percent NPL-noisy acquisitions used
for the previous 450-image comparisons.
For each test image, the sampled Hadamard-mask and noisy
measurement SHA256 hashes were verified against the archived
baseline. We did not select or modify hyperparameters after
examining reconstruction metrics.

<<PAIREDTEXT>>

<<DISTRIBUTION>>

\begin{table}[htbp]
\centering\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Method}&\textbf{PSNR, dB}&\textbf{SSIM}&
\textbf{HaarPSI}&\textbf{MSE}\\
\midrule
<<SCORETABLE>>
\bottomrule
\end{tabular}}
\caption{Mean $\pm$ sample SD over 450 paired images, calculated on
raw (unclipped) reconstructions. Ten method labels are shown,
including the original SSDU-5 and newly expanded-5 GS-PnP.}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\linewidth]{../test450/analysis/figures/10_methods_450_boxplots.pdf}
\caption{Paired 450-image score distributions for ten methods.
Outliers are not drawn in the boxplot, but all 450 values contribute
to the descriptive statistics.}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\linewidth]{../test450/analysis/figures/10_methods_450_mean_sd.pdf}
\caption{Mean and standard deviation for all ten methods across 450 images.}
\end{figure}

\subsection*{Results by device category}
\begin{table}[htbp]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
\textbf{Category}&\textbf{ZF}&\textbf{SPGL1}&\textbf{BPDN}&
\textbf{ADMM}&\textbf{GS oracle}&\textbf{TV}&\textbf{C60}&
\textbf{GS SSDU-5}&\textbf{GS expanded}\\
\midrule
<<CATTABLE>>
\bottomrule
\end{tabular}}
\caption{Category mean PSNR in dB; 90 matched test cases per category.
The full ten-method category metrics and their sample standard
deviations are in the companion JSON and figure files.}
\end{table}

\subsection*{Five illustrative PCM reconstructions}
Five example images are shown using exactly the previously fixed case
selection: CIGS 0279, Organic 0331, Perovskite 0786, Reference 0279
and Silicone 0370. Each comparison includes one target and all ten
methods, displaying PSNR, SSIM, HaarPSI and MSE beside the image.
Colour is AFM-hot capped at 0.8, with identical image intensity
range $[0,1]$ for every panel. These examples are illustrative and
were not used to adjust hyperparameters.

\clearpage
\includepdf[pages=-,pagecommand={}]{../test450/analysis/figures/Five_Examples_10_Methods_with_Metrics.pdf}

"""
    replacements = {
        "<<OLDLOSS>>": f"{original['SSDU_MSE']:.9f}",
        "<<BESTLOSS>>": f"{winner['SSDU_MSE']:.9f}",
        "<<WINNER>>": winner["candidate_id"].replace("_", r"\_"),
        "<<IMPROVEMENT>>": f"{improvement:.2f}",
        "<<LAM>>": f"{p['lambda']:.3f}",
        "<<SIG>>": f"{p['sigma']:.3f}",
        "<<TAU>>": f"{p['tau0']:.4f}",
        "<<INVALID>>": str(len(invalid)),
        "<<FLAGLAM>>": "yes" if boundaries["lambda_upper"] else "no",
        "<<FLAGSIG>>": "yes" if boundaries["sigma_upper"] else "no",
        "<<FLAGTAU>>": "yes" if boundaries["tau0_upper"] else "no",
        "<<TOPRANK>>": top_candidates(selected),
        "<<DIAGNOSTICS>>": diag_rows,
        "<<PAIREDTEXT>>": pairedtext,
        "<<DISTRIBUTION>>": distribution,
        "<<SCORETABLE>>": latex_stats(summary),
        "<<CATTABLE>>": by_category(summary),
        "<<CATEGORY_VALIDATION>>": category_validation_rows,
        "<<INVALIDDETAILS>>": invalid_table,
    }
    for key, value in replacements.items():
        assert key in latex
        latex = latex.replace(key, value)
    assert "<<" not in latex
    # Keep original six sections. Insert all findings into section 5 and
    # replace the previous preliminary conclusion.
    source = source.replace(
        r"\usepackage{amsmath,amssymb,booktabs,longtable,hyperref,fancyhdr,enumitem,graphicx}",
        r"\usepackage{amsmath,amssymb,booktabs,longtable,hyperref,fancyhdr,enumitem,graphicx,pdfpages}",
    )
    assert source.count(r"\section{Conclusion}") == 1
    start = source.index(r"\section{Conclusion}")
    refs = source.index(r"\paragraph{Primary papers.", start)
    newconclusion = r"""\section{Conclusion}

The expanded five-image validation SSDU sweep is complete,
and the winning configuration was frozen before the additional
450-image reconstruction evaluation.
The measurement-only SSDU loss fell from <<OLDLOSS>> to
<<BESTLOSS>> (<<IMPROVEMENT>> percent reduction).
The validated candidate is \texttt{<<WINNER>>}, with
$\lambda=<<LAM>>$, $\sigma=<<SIG>>$ and
initial $\tau_0=<<TAU>>$.
The selected candidate lies at new upper search bounds
as follows: lambda <<FLAGLAM>>, sigma <<FLAGSIG>>,
initial step <<FLAGTAU>>.

<<PAIREDTEXT>>

These observations show the behaviour of this finite,
preregistered SSDU grid. They do not prove a global optimum.
The old test set had already been inspected in the broader
research project, so this is a retrospective controlled test.
Its results must not be used to make further unrecorded
hyperparameter changes. PCM noise correlation and the
undersampled Hadamard nullspace remain important limitations.

"""
    for key, value in replacements.items():
        newconclusion = newconclusion.replace(key, value)
    source = source[:start] + latex + "\n" + newconclusion + source[refs:]
    source = source.replace(
        "Current status: Prepared, not run.",
        "Current status: Completed validation sweep, frozen parameters and retrospective 450-image test.",
    )
    source = source.replace(
        "We prepare 27 new-grid configurations",
        "We evaluated 27 preregistered candidate configurations",
    )
    source = source.replace(
        "The expanded sweep has not begun. The intended GPU is",
        "The expanded sweep completed on 9 November 2026. The computing GPU was",
    )
    source = source.replace("8 November 2026", "9 November 2026")
    source = source.replace("version 1.0", "version 2.0")
    (REPORT / "GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex").write_text(source)
    (REPORT / "full_report_data.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "old_ssdu_mse": original["SSDU_MSE"],
                "relative_reduction_pct": improvement,
                "score_summary": summary["ALL"],
                "paired": paired,
                "diagnostics": diagnostics,
                "selection_sha256": sha_file(ROOT / "frozen_selection.json"),
                "test_report": str(
                    REPORT / "GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex"
                ),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        "FULL_REPORT_TEX_COMPLETE",
        REPORT / "GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex",
        "WINNER",
        winner["candidate_id"],
        flush=True,
    )


if __name__ == "__main__":
    make_report()
