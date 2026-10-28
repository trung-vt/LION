from pathlib import Path
import json

O = Path("/store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028")
s = json.loads((O / "summary_50.json").read_text())
p = json.loads((O / "protocol_manifest_50.json").read_text())
K = ("mse", "psnr", "ssim", "haarpsi")
C = p["categories"]
R = p["balanced_noise_conditions"]
B = "balanced_50_50"
parts = []


def A(t):
    parts.append(t)


def er(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def line(*cells):
    return " & ".join(map(str, cells)) + " " + chr(92) * 2


def mean(m, k, reg=B):
    return s["means"][m][k][reg]["mean"]


def effect(c, k, reg=B):
    return s["comparisons"][c][k][reg]


def n(v, k="mse", sign=False):
    if k == "mse":
        return f"{v*1000:+.3f}" if sign else f"{v*1000:.3f}"
    if k == "psnr":
        return f"{v:+.3f}" if sign else f"{v:.3f}"
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def CI(e, k):
    return "[" + ", ".join(n(a, k, True) for a in e["ci95"]) + "]"


def fig(name, width=r".92\linewidth"):
    A(r"\begin{figure}[p]\centering")
    A(r"\includegraphics[width=" + width + "]{figures/" + name + ".pdf}")
    A(r"\caption{" + er(name.replace("_", " ").capitalize()) + r".}")
    A(r"\end{figure}\clearpage")


A(
    r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=21mm,headheight=15pt]{geometry}
\usepackage{graphicx,booktabs,longtable,tabularx,amsmath,amssymb,fancyhdr,hyperref,microtype}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{enumitem}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\pagestyle{fancy}\fancyhf{}\lhead{LION PCM | paired comparison}\rhead{28 October 2026}\cfoot{\thepage}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\newcommand{\mes}{\ensuremath{10^{3}\!\times\!\mathrm{MSE}}}
\begin{document}
\begin{center}
{\LARGE\bfseries Full comparison of 4070 and 4090 trainings}\\[7pt]
{\Large 50-image repeat: previous and current checkpoints}\\[7pt]
LION / PCM supervised U-TV reconstruction\\[4pt]
28 October 2026 | RTX 4070 evaluation on tuni
\end{center}
\hrule\vspace{6pt}
\begin{abstract}
A fully paired comparison on 50 test images (10 sampled without replacement per category), under two predeclared PCM noise regimes. The previous frozen checkpoints (4070 epoch 7, 4090 epoch 22) and current validation-selected checkpoints (4070 epoch 17, 4090 epoch 23) are compared on the exact same 100 acquisitions. The original measurement hashes are reproduced exactly. Two hundred current-checkpoint reconstructions were run on the RTX 4070; the previous checkpoint metrics were reused from the complete 27 October test. Reported confidence intervals use 20,000 paired, category-stratified bootstrap resamples. The images were held out from training and validation but were already used in the earlier full test, so this repeat must not be represented as a new untouched test-set confirmation.
\end{abstract}
\tableofcontents\clearpage
\section{Executive summary}
"""
)
old7 = mean("4070_old", "mse")
new7 = mean("4070_new", "mse")
old9 = mean("4090_old", "mse")
new9 = mean("4090_new", "mse")
A(r"\begin{itemize}[leftmargin=*]")
A(
    r"\item \textbf{4070:} balanced MSE decreases by "
    + f"{100*(1-new7/old7):.1f}"
    + r"\% ("
    + n(old7)
    + " to "
    + n(new7)
    + r" in units of $10^{-3}$); PSNR increases by "
    + n(mean("4070_new", "psnr") - mean("4070_old", "psnr"), "psnr")
    + r" dB. MSE paired 95\% CI: "
    + CI(effect("4070_new_minus_old", "mse"), "mse")
    + "."
)
A(
    r"\item \textbf{4090:} balanced MSE decreases by "
    + f"{100*(1-new9/old9):.1f}"
    + r"\% ("
    + n(old9)
    + " to "
    + n(new9)
    + r" in units of $10^{-3}$); PSNR and SSIM decline on average. The paired MSE interval "
    + CI(effect("4090_new_minus_old", "mse"), "mse")
    + " contains zero."
)
A(
    r"\item \textbf{Current head-to-head:} 4070 is ahead on all four average metrics. The 4090-minus-4070 MSE interval "
    + CI(effect("new_4090_minus_new_4070", "mse"), "mse")
    + r" ($\times 10^{-3}$) includes zero; the MSE mean ranking is not resolved convincingly by this 50-image sample."
)
A(
    r"\item \textbf{Important qualification:} the same 450-image test partition was already evaluated on 27 October. This is a predeclared random subset to track checkpoint progress on matched measurements, not a new external validation of model-selection choices."
)
A(r"\end{itemize}")
A(
    r"""\section{Protocol and reproducibility}
\textbf{Test split.} The original configuration uses permuted positions 1--900 for training, 901--910 for validation, and 911--1000 for test. Exactly 10 of the 90 test IDs per category were drawn with no replacement, independently within CIGS, Organic, Perovskite, Reference and Silicone. A single fixed random seed namespace was recorded before evaluation, and the same 50 images were used for all checkpoints. There was no selection based on reconstruction results.

\textbf{Forward model.} PCM normalized operator, 256-by-256 reference images, multilevel Hadamard patterns at $J=8$ and coarse level $J=6$, with each image's sampling ratio drawn uniformly from $[0.05,0.50]$. One acquisition per image per noise condition, for a total of 100 acquisitions. Both U-TV models reconstruct the exact same measurement object for each acquisition. All 100 measurement and pattern hashes matched the earlier evaluation.

\textbf{Noise regimes.} \emph{Zero-inclusive} draws raw dark noise on $[0,10^{-5}]$, bias on $[0,4\cdot 10^{-5}]$, bias gradient on $[0,2\cdot10^{-7}]$ and beam on $[0,6\cdot10^{-4}]$, each via shifted log-uniform. The lower bound is zero but the continuous sampler need not yield exactly zero. \emph{Positive} uses log-uniform raw dark noise $[10^{-8},10^{-5}]$ and uniform bias $[2\cdot10^{-5},4\cdot10^{-5}]$, bias gradient $[10^{-7},2\cdot10^{-7}]$, beam $[10^{-4},6\cdot10^{-4}]$. The original physical model-to-current conversion is unchanged.

\textbf{Checkpoints.} All four are 4-block U-Nets with 128 initial filters and 512 unrolled PDHG iterations. The current checkpoint was selected before this subset's evaluation from each run's existing validation procedure; no change was made to ongoing training. The older 4090 job uses stochastic validation acquisitions and different optimizer/noise settings, which complicates causal attribution.

\textbf{Uncertainty.} Each bootstrap draw resamples 10 full image units with replacement within each of the five categories (20,000 draws). Its unit is an image with both noise acquisitions and all checkpoints paired. Reported 95\% intervals are percentile intervals for paired mean differences; they reflect subset sampling variation, not train-seed or synthetic-generator uncertainty. This is not a new, unseen-test confirmation because the full test partition was examined previously.
"""
)
A(
    r"\begin{table}[ht]\centering\caption{Frozen checkpoints; old snapshots from 27 October and new snapshots frozen before this repeat.}\small"
)
A(
    r"\begin{tabular}{lrrl}\toprule Checkpoint & Epoch & Optimizer steps & Selection \\ \midrule"
)
for model, kind in [("4070", "old"), ("4070", "new"), ("4090", "old"), ("4090", "new")]:
    z = (
        p["checkpoints_old_frozen"][model]
        if kind == "old"
        else p["checkpoints_current_frozen"][model]
    )
    A(
        line(
            model + " " + kind,
            z["epoch"],
            f"{int(z.get('step') or z.get('global_step')):,}",
            "Training validation",
        )
    )
A(r"\bottomrule\end{tabular}\end{table}")
A(
    r"\textbf{Source commit:} \texttt{9188b0735843c96526537b6b9bfd834fd089f7cd}; the full checkpoint hashes, exact IDs, copied noise settings and acquisition seeds are stored in \texttt{protocol\_manifest\_50.json}."
)
A(r"\section{Overall reconstruction metrics}")
A(
    r"\begin{table}[ht]\centering\caption{Mean of the 100 acquisitions, evenly balancing the two noise regimes. MSE columns are scaled by $10^{3}$.}\small"
)
A(
    r"\begin{tabular}{lrrrr}\toprule Checkpoint & \mes & PSNR (dB) & SSIM & HaarPSI \\ \midrule"
)
for label, pretty in [
    ("4070_old", "4070 epoch 7"),
    ("4070_new", "4070 epoch 17"),
    ("4090_old", "4090 epoch 22"),
    ("4090_new", "4090 epoch 23"),
]:
    A(line(pretty, *[n(mean(label, k), k) for k in K]))
A(r"\bottomrule\end{tabular}\end{table}")
A(
    r"\textbf{Reading the metrics.} The 4070 checkpoint improves simultaneously in MSE, PSNR, SSIM and HaarPSI, while the 4090 changes in different directions. An improvement in mean MSE can coexist with a loss of mean per-image PSNR because of the nonlinearity of the logarithm, especially if a few high-error acquisitions dominate the mean MSE."
)
fig("four_checkpoint_means", r".92\linewidth")
A(r"\section{Noise-specific reconstruction results}")
A(
    r"\begin{table}[ht]\centering\caption{Mean metrics by acquisition noise regime (50 acquisitions per regime). MSE uses $10^{-3}$ units.}\small"
)
A(
    r"\begin{tabular}{llrrrr}\toprule Regime & Checkpoint & \mes & PSNR & SSIM & HaarPSI \\ \midrule"
)
for reg, tag in [
    ("zero_inclusive_4070", "Zero-inclusive"),
    ("positive_4090", "Positive"),
]:
    for label, pretty in [
        ("4070_old", "4070 old"),
        ("4070_new", "4070 new"),
        ("4090_old", "4090 old"),
        ("4090_new", "4090 new"),
    ]:
        A(line(tag, pretty, *[n(mean(label, k, reg), k) for k in K]))
    A(r"\midrule" if reg == "zero_inclusive_4070" else r"")
A(r"\bottomrule\end{tabular}\end{table}")
A(
    r"\textbf{Robustness across noise distributions.} The 4070 gains in all four metrics under both regimes. The latest 4090 gains modestly in mean MSE for both regimes, but mean PSNR and SSIM decline in both. Thus the direction of these changes is not caused solely by averaging the two noise regimes."
)
A(r"\section{Paired changes and uncertainty}")
A(
    r"""For a difference, \emph{positive} is favorable to the 4090 in the current head-to-head, except for MSE where \emph{negative} is favorable. A negative ``new minus old'' MSE denotes learning progress. Confidence intervals below are calculated on paired image-level units while preserving both acquisitions per image. All figures use 20,000 category-stratified bootstrap samples."""
)
contrast_names = [
    ("4070_new_minus_old", "4070: new - old"),
    ("4090_new_minus_old", "4090: new - old"),
    ("old_4090_minus_old_4070", "Old: 4090 - 4070"),
    ("new_4090_minus_new_4070", "New: 4090 - 4070"),
    ("head_to_head_delta_change", "Shift in model gap"),
]
for k, pretty in [
    ("mse", r"\mes"),
    ("psnr", "PSNR (dB)"),
    ("ssim", "SSIM"),
    ("haarpsi", "HaarPSI"),
]:
    A(r"\begin{table}[ht]\centering\small")
    A(
        r"\caption{Paired "
        + er(k.upper())
        + r" differences and 95\% confidence intervals (balanced regimes). "
        + ("Scaled by $10^{3}$." if k == "mse" else "")
        + "}"
    )
    A(r"\begin{tabular}{lrr}\toprule Comparison & Difference & 95\% CI \\ \midrule")
    for key, label in contrast_names:
        a = effect(key, k)
        A(line(label, n(a["mean"], k, True), CI(a, k)))
    A(r"\bottomrule\end{tabular}\end{table}")
    if k == "psnr":
        A(r"\clearpage")
A(
    r"""\textbf{Reading the paired results.} The 4070 gains are reasonably consistent across acquisitions in PSNR, SSIM and HaarPSI. The 4090's MSE change interval overlaps zero, while its paired PSNR and SSIM changes favor the older checkpoint. The difference-in-differences indicates that the relative gap between 4070 and 4090 shifted toward the 4070, although the current mean MSE head-to-head comparison itself has an interval including zero. The two statements are compatible: change in relative performance is measured more precisely than the remaining gap."""
)
fig("paired_effect_confidence_intervals", r".91\linewidth")
A(r"\section{Category and sampling heterogeneity}")
A(
    r"\begin{table}[ht]\centering\caption{Current-checkpoint mean errors by category and noise regime. Negative gap is favorable to 4090. MSE numbers are $\times10^{-3}$.}\footnotesize"
)
A(
    r"\begin{tabular}{llrrr}\toprule Category & Noise & 4070 MSE & 4090 MSE & Gap (4090 - 4070) \\ \midrule"
)
for ci, cat in enumerate(C):
    for reg, tag in [("zero_inclusive_4070", "Zero"), ("positive_4090", "Positive")]:
        a = s["by_category"][cat][reg]
        A(
            line(
                cat,
                tag,
                n(a["4070_new"]["mse"]),
                n(a["4090_new"]["mse"]),
                n(a["delta_4090_minus_4070_new"]["mse"], sign=True),
            )
        )
    if ci < 4:
        A(r"\addlinespace")
A(r"\bottomrule\end{tabular}\end{table}")
A(
    r"""\textbf{Heterogeneity.} A positive aggregated MSE gap does not imply that one checkpoint wins on every type of image. Some categories and sampling ratios favor the 4090. The paired distribution figure reveals whether the average is sensitive to a small number of large errors; do not interpret the 50-image mean as a category-independent effect."""
)
fig("category_by_noise", r".91\linewidth")
A(
    r"\begin{table}[ht]\centering\caption{MSE changes stratified by sampling ratio (scaled by $10^3$). Small bin counts mean exploratory results.}\small"
)
A(
    r"\begin{tabular}{lrrrr}\toprule Sampling ratio & Images & New gap & 4070 new - old & 4090 new - old \\ \midrule"
)
for band, z in s["sampling"]["bins"].items():
    A(
        line(
            band,
            z["images"],
            n(z["new_4090_minus_4070_mse"], sign=True)
            if z["new_4090_minus_4070_mse"] is not None
            else "--",
            n(z["4070_gain_mse"], sign=True)
            if z["4070_gain_mse"] is not None
            else "--",
            n(z["4090_gain_mse"], sign=True)
            if z["4090_gain_mse"] is not None
            else "--",
        )
    )
A(r"\bottomrule\end{tabular}\end{table}")
fig("mse_difference_vs_sampling", r".90\linewidth")
A(r"\section{How often each model wins}")
A(
    r"\begin{table}[ht]\centering\caption{Acquisition-level wins for the current 4090 versus current 4070 (50 acquisitions per regime).}\small"
)
A(
    r"\begin{tabular}{llrrr}\toprule Noise regime & Metric & 4070 wins & 4090 wins & Ties \\ \midrule"
)
for reg, tag in [
    ("zero_inclusive_4070", "Zero-inclusive"),
    ("positive_4090", "Positive"),
]:
    for k in K:
        z = s["wins"][reg][k]
        A(line(tag, k.upper(), z["4070"], z["4090"], z["tie"]))
    A(r"\midrule" if tag == "Zero-inclusive" else "")
A(r"\bottomrule\end{tabular}\end{table}")
A(
    r"""The win counts and mean differences answer different questions. A model may win on most cases while losing the \emph{mean MSE} because its losses on a few difficult images are much larger. PSNR, SSIM and HaarPSI are therefore essential alongside mean MSE. Each acquisition yields one deterministic paired comparison; these counts are not independent replications of training."""
)
fig("paired_old_new_scatter", r".90\linewidth")
A(r"\section{Qualitative reconstruction assessment}")
A(
    r"""For each category, the illustrative image is the \emph{first of the ten preselected random IDs}, not an outcome-dependent best or worst case. Each panel displays the same normalized intensity scale, $[0,1.2]$, using the afmhot colormap. The zero-filled image is included as a reconstruction reference. The newly trained 4070 and 4090 checkpoints both receive the identical acquisition. Individual image panels are embedded in a PDF figure without first flattening them into a giant low-resolution bitmap. Numerical PSNR is displayed adjacent to reconstructions. The qualitative comparisons serve as examples; the tables reflect all 50 images, not merely these examples."""
)
fig("reconstructions_zero_inclusive_4070", r".80\linewidth")
fig("reconstructions_positive_4090", r".80\linewidth")
A(r"\subsection{Absolute error on common scales}")
A(
    r"""The following figures use absolute errors with the same linear color scale $[0,0.5]$ for \emph{all} displayed categories, noise regimes and models. This prevents a model with larger errors from appearing artificially similar through independent auto-rescaling. The display can clip larger errors, so numeric MSE and the detailed per-image CSV remain the quantitative reference."""
)
fig("absolute_error_zero_inclusive_4070", r".56\linewidth")
fig("absolute_error_positive_4090", r".56\linewidth")
A(r"\section{Performance impact and training safety}")
A(
    r"""The experiment used physical RTX 4070 (CUDA device 0), with approximately 10.6~GiB free VRAM before launch. Only the frozen current checkpoints were evaluated; old checkpoint results were reused. The 200 reconstructions (100 acquisitions times two current models) finished between 14:11:53 and 14:14:42 UTC without GPU OOM or errors, and all three ongoing training processes were observed alive afterwards. The evaluation paused 0.5 seconds per image and used a low CPU scheduling priority.

The 4070 training monitor recorded median 2.30 seconds/step during the 15 minutes just before evaluation, 2.33 during the three minute evaluation window and 2.30 in the three minutes after. This represents a short-term increase of approximately 1.3\% while evaluating, returning to baseline afterwards; it is not a complete controlled throughput experiment. The original 4090 and 3-by-56 training were not touched."""
)
A(r"\section{Interpretation, limitations, and next steps}")
A(
    r"""\begin{enumerate}[leftmargin=*]
\item \textbf{Evidence of continued improvement.} The 4070 epoch-17 checkpoint delivers better reconstruction metrics than its epoch-7 checkpoint on exactly matched measurements. The paired change has a negative MSE interval that excludes zero and positive PSNR, SSIM and HaarPSI changes.
\item \textbf{4090 is not uniformly improving.} Its epoch-23 checkpoint slightly improves the mean MSE over epoch 22, but loses mean PSNR and SSIM. These discrepancies are compatible with image-dependent changes in loss distribution and merit investigation with fixed validation acquisitions and visual examples. They do not establish that the hardware itself is worse.
\item \textbf{Cross-model comparisons.} The current 4070 leads the sample averages, particularly PSNR and HaarPSI. However, some image categories and low sampling-ratio cases favor 4090 in MSE, and the overall head-to-head MSE interval contains zero. Avoid saying the 4070 is universally better.
\item \textbf{Selection and reuse limitations.} The 50 images were held out from model fitting, but all came from a test set already examined on 27 October. Repeated inspection may bias research decisions. The 4090 uses a differently distributed, stochastic validation procedure and an unequal number of training updates; neither model was trained under a controlled single-factor intervention. All images originate from the same synthetic PCM generator, not unseen physical acquisitions.
\item \textbf{Recommended next analysis.} Compare future checkpoints at predeclared epoch milestones on a \emph{fixed, separate validation acquisition set}; reserve untouched data for the final claim. To test causal hypotheses about optimizer versus noise schedule, run controlled retraining with common settings. Keep observing both running jobs without altering this experiment's frozen data.
\end{enumerate}"""
)
A(r"\appendix\section{Selected test image IDs and reproduction}")
A(
    r"\begin{longtable}{lp{.74\linewidth}}\toprule Category & Fixed random sample IDs \\ \midrule\endhead"
)
for cat in C:
    A(line(cat, ", ".join(map(str, p["subset_per_category"][cat]))))
A(r"\bottomrule\end{longtable}")
A(
    r"""The bundled \texttt{protocol\_manifest\_50.json} contains checkpoint epochs, steps, hashes and selection provenance. \texttt{evaluate\_current\_50.py} regenerates paired PCM acquisitions using the earlier test's seed namespace and checks pattern and measurement hashes. \texttt{summarize\_50.py} reconstructs the tables and paired bootstrap intervals. \texttt{make\_figures.py} generates both vector PDF and PNG charts. \texttt{all\_paired\_results\_old\_new.csv} contains all 100 acquisitions and all four checkpoint metric sets. The old full-test data have not been overwritten. The LaTeX source refers to the accompanying \texttt{figures/} PDF files and can be compiled with \texttt{latexmk -pdf}."""
)
A(r"\section{Machine-readable source files}")
A(
    r"""\begin{itemize}[leftmargin=*]
\item \texttt{summary\_50.json}: overall results, per-category means, paired uncertainty, win counts, ratio stratification.
\item \texttt{all\_paired\_results\_old\_new.csv}: complete measured per-acquisition data for four frozen model checkpoints.
\item \texttt{protocol\_manifest\_50.json}: original and current checkpoint hashes, random sampling and test-reuse caveats.
\item \texttt{figures/}: nine PDF plots and matching PNG previews; the reconstruction/absolute-error panels use common display scales.
\item The five \texttt{current\_CATEGORY.csv} and ten \texttt{example\_CATEGORY\_REGIME.npz} files are also preserved in the run directory on tuni.
\end{itemize}"""
)
A(r"\end{document}")
tex = O / "Full_comparison_4070_4090_50_images.tex"
tex.write_text("\n".join(parts) + "\n")
print(
    "TEX_READY",
    tex,
    "BYTES",
    tex.stat().st_size,
    "FIGURES",
    len(list((O / "figures").glob("*.pdf"))),
)
