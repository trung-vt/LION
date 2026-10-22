"""Create LaTeX reports from the completed randomized-noise pilot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from LION.pcm.randomized.figures import LABELS


def tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(headers: list[str], rows: list[list[str]], specification: str) -> str:
    return "\n".join(
        [
            r"\begin{tabular}{" + specification + "}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            *[" & ".join(row) + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def generate(root: Path) -> None:
    bundle = root / "report_bundle"
    data = json.loads((bundle / "summary.json").read_text())
    coverage = pd.read_csv(root / "training_coverage.csv")
    den = pd.read_csv(bundle / "denoising_test.csv")
    reg = pd.read_csv(bundle / "regularizer_test.csv")
    pnp = pd.read_csv(bundle / "pnp_test.csv")
    choices = json.loads((root / "pnp_sweep/selected.json").read_text())
    names = ["dwtdnet", "gs_dwtdnet", "drunet", "gs_drunet"]
    den_rows = []
    for name in names:
        rows = den[(den.model == name) & (den.noise_kind == "pcm_like")]
        g = rows[rows.training_noise == "gaussian"].psnr.mean()
        m = rows[rows.training_noise == "mixture"].psnr.mean()
        den_rows.append([LABELS[name], f"{g:.2f}", f"{m:.2f}", f"{m-g:+.2f}"])
    den_table = table(["Architecture", "Gaussian", "Mixed", "Change"], den_rows, "lrrr")
    regs = ["utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu"]
    reg_rows = []
    for name in regs:
        rows = reg[
            (reg.model == name)
            & (reg.severity == "medium")
            & np.isclose(reg.sampling_ratio, 0.2)
        ]
        run = next(r for r in data["run_summary"] if r["model"] == name)
        reg_rows.append(
            [
                LABELS[name],
                str(run["best_step"]),
                f"{rows.psnr.mean():.2f}",
                f"{rows.baseline_psnr.mean():.2f}",
            ]
        )
    reg_table = table(
        ["Method", "Selected update", "PSNR", "Zero-filled"], reg_rows, "lrrr"
    )
    pnp_rows = []
    for name in names:
        config = next(
            c["config"]
            for c in choices
            if c["model"] == name and c["ratio"] == 0.2 and c["dark_sigma"] == 0.2
        )
        setting = (
            f"$\\rho={config['rho']:g}$"
            if "rho" in config
            else f"$\\lambda={config['lambda']:g},\\tau_0={config['tau']:g}$"
        )
        conditioning = "ignored" if "dwtdnet" in name else f"{config['sigma']:g}"
        rows = pnp[
            (pnp.model == name)
            & np.isclose(pnp.ratio, 0.2)
            & np.isclose(pnp.dark_sigma, 0.2)
        ]
        pnp_rows.append(
            [
                LABELS[name],
                setting,
                conditioning,
                f"{rows.psnr.mean():.2f}",
                f"{rows.seconds.mean():.3f}",
            ]
        )
    pnp_table = table(
        ["Denoiser", "Selected setting", "$\\sigma$", "PSNR", "Seconds"],
        pnp_rows,
        "llrrr",
    )
    all_rows = []
    for row in data["run_summary"]:
        name = LABELS[row["model"]] + (
            " / G"
            if row["noise"] == "gaussian"
            else " / M"
            if row["noise"] == "mixture"
            else ""
        )
        all_rows.append(
            [
                name,
                str(row["updates"]),
                str(row["best_step"]),
                f"{row['test_psnr']:.2f}",
                f"{row['test_ssim']:.3f}",
                f"{row['seconds']:.1f}",
            ]
        )
    all_table = table(
        ["Run", "Updates", "Selected", "PSNR", "SSIM", "Run sec."], all_rows, "lrrrrr"
    )
    dcoverage = coverage[coverage.run == "dwtdnet_mixture"].iloc[0]
    rcoverage = coverage[coverage.run == "utv_supervised_physical_pcm"].iloc[0]
    stops = data["pnp_stops"]
    improved = sum(float(row[3]) > 0 for row in den_rows)
    protocol = data["pnp_protocol"]
    values = {
        "@@DENOISER_TABLE@@": den_table,
        "@@REGULARIZER_TABLE@@": reg_table,
        "@@PNP_TABLE@@": pnp_table,
        "@@ALL_TABLE@@": all_table,
        "@@DENOISER_DISTINCT@@": str(int(dcoverage.distinct_sources)),
        "@@REGULARIZER_DISTINCT@@": str(int(rcoverage.distinct_sources)),
        "@@RATIO_MIN@@": f"{100*rcoverage.requested_ratio_min:.2f}",
        "@@RATIO_MAX@@": f"{100*rcoverage.requested_ratio_max:.2f}",
        "@@BETTER_COUNT@@": str(improved),
        "@@VALIDATION_RECONS@@": str(protocol["validation_reconstructions"]),
        "@@TEST_RECONS@@": str(protocol["test_reconstructions"]),
        "@@STAGNATION_STOPS@@": str(stops.get("numerical_stagnation", 0)),
        "@@RESULT_ROOT@@": str(root),
    }

    article = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,hyperref,xcolor,float,listings}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\graphicspath{{figures/}}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible}
\newcommand{\pilotfigure}[3]{\begin{figure}[H]\centering\includegraphics[width=#2\linewidth]{#1}\caption{#3}\end{figure}}
\title{Randomized-noise PCM experiments\\\large Bounded pilot study and reproducibility report}
\author{A3SmartML / LION working report}
\date{14 October 2026}
\begin{document}
\maketitle
\textbf{Status: preliminary engineering experiments, not final paper evidence.}
All reported numbers come from completed local runs. The experiments use one
training seed, $64\times64$ grayscale crops and four distinct source images per
validation or test panel (two natural and two synthetic). They do not establish
full-resolution performance or statistical superiority.

\section{Questions and completed experiments}
We tested three questions: whether mixed Gaussian/PCM-like image corruptions
improve denoiser robustness, whether learned TV/TGV maps tolerate random
physical PCM noise and sampling, and which PnP parameters work on fixed
validation cases. Twelve training runs finished: eight denoiser pilots and
four adaptive-regularizer pilots. The corrected PnP study evaluated
@@VALIDATION_RECONS@@ validation reconstructions, selected 24 configurations,
and evaluated @@TEST_RECONS@@ held-out reconstructions.

Mixed-noise training improved mean test PSNR under the raster PCM-like
surrogate for @@BETTER_COUNT@@ of four architectures, not all of them.
Supervised U-TV and U-TGV outperformed the corresponding SSDU pilots in the
fixed medium-noise panel. These are observations for this small test set,
not general claims about the methods. All non-finite checks passed in the
completed runs. The first GS search failure was retained separately and is
explained in Section~\ref{sec:validation}.

\section{Data and separation}
The natural-image pool is the locally available DIV2K high-resolution data
\cite{div2k}, with CBSD68 used as a separate natural test pool. The nominal
DF2K directory duplicated 800 DIV2K files; SHA-256 deduplication removed these
copies. No additional dataset was downloaded.
\begin{center}
\begin{tabular}{lrrr}\toprule
Source & Train & Validation & Test\\\midrule
Natural images & 800 & 100 & 68\\
Synthetic PCM images & 4000 & 500 & 500\\\bottomrule
\end{tabular}
\end{center}
Synthetic image identifiers were split 80/10/10 with seed 20261014, keeping
the same identifier in one split across all five material categories.
Source-file hashes are disjoint across splits. The synthetic generator's
underlying template ancestry was not available; template-disjoint testing
is therefore \emph{not} claimed.

Each training draw selects the natural or synthetic pool with probability
one half, then an image uniformly from that pool. We convert to grayscale,
normalize by 255, extract a $64\times64$ crop, and apply random flips/rotations.
Validation and test crops use deterministic center crops of hash-sorted
source files. A denoiser run used 256 crop draws from
@@DENOISER_DISTINCT@@ distinct source images, not all 4800 training images.
A regularizer run used 1024 draws from @@REGULARIZER_DISTINCT@@ sources.
The exact identities are stored in the manifest and per-run draw logs.

\section{Noise distribution and interpretation}
The coefficient intervals are arithmetic floors and ceilings at two decimal
places, including negative coefficients:
\begin{center}
\begin{tabular}{ll}\toprule
Parameter & Distribution\\\midrule
$\sigma_g$ & $\mathcal U(0,3\cdot10^{-7})$\\
$\sigma_{\rm bias}$ & $\mathcal U(0,4\cdot10^{-5})$\\
$\sigma_{\rm beam}$ & $\mathcal U(0,10^{-3})$\\
$\sigma_{\rm dark}$ & $\mathcal U(0,1)$\\
AR coefficients & $[-.93,-.92],[-.18,-.17],[-.08,-.07],[-.04,-.03]$\\
MA coefficients & $[.03,.04],[-.37,-.36]$\\
Integration order & $d=1$ fixed\\\bottomrule
\end{tabular}
\end{center}
Draws are rejected unless the ARMA increments are stationary and invertible.
This does not make the integrated beam process stationary. The existing
LION simulator uses $\sigma_{\rm beam}$ as the innovation SD, not the marginal
SD of the cumulative beam trace. We preserve that convention. The fixed
signal-to-bias ratio is 0.3. Sigma values are scenario amplitudes on normalized
image/measurement units, not calibrated ampere ranges.

\subsection{Denoisers: no measurement operator}
For denoiser training we use a clearly labelled \emph{raster-domain surrogate}.
Let $t$ be the C-order pixel index, $\overline I=\operatorname{mean}(x)/0.3$,
and let $b_t$ be the cumulative ARIMA beam trace. Then
\begin{equation}
 z_t=x_t(1+b_t)+\overline I(tg+\eta_t)+\delta_t,
 \quad g\sim\mathcal N(0,\sigma_g^2),\quad
 \eta_t\sim\mathcal N(0,\sigma_{\rm bias}^2),\quad
 \delta_t\sim\mathcal N(0,\sigma_{\rm dark}^2).
\end{equation}
There is no Hadamard transform, differential pair, or subsampling here. The
constant illumination offset is subtracted. The mixed arm uses this
surrogate for half of the draws and additive Gaussian corruption for the
other half. The control arm uses Gaussian corruption for every draw, with
the same image sequence, initialization seed and update budget. Noisy inputs
and model outputs are not clipped for losses or metrics. Display images
alone use the common range $[0,1]$.

\subsection{Adaptive regularizers and PnP: physical measurements}
For reconstruction experiments we call LION's existing split-pattern PCM
simulator. It forms positive/negative measurements, applies the raw noise
mechanisms and pair-order randomization, and then takes their difference.
The normalized sensing operator is $A=SPW$ and $y=Ax+e$. The illumination level
uses the image sum and the operator normalization. Independent dark noises
on a pair give a differential SD of $\sqrt{2}\sigma_{\rm dark}$, not
$\sigma_{\rm dark}$. Thus image corruption and measurement corruption are
not interchangeable training targets.

We draw $r\sim\mathcal U(.01,1)$ and retain
$m=\max(2,\lceil rN\rceil)$ unique coefficients, with DC always retained.
For an image side $2^J$, eligible coarse levels are
$\{0\}\cup\{c\in\{1,\ldots,J\}:4^{c-1}<m\}$. This allows a partially
sampled last coarse level without invalid calls. We use the existing
coarse-block plus uniform-tail sampler. Boundary tests found that the older
multilevel allocation can underfill high sampling rates; it was not changed
or used by these new pilots. Actual training rates ranged from
@@RATIO_MIN@@\% to @@RATIO_MAX@@\%; endpoint 1\% and 100\% are covered by tests
and evaluation, even though continuous random draws need not hit them.
\pilotfigure{sampling_coverage.png}{0.78}{Actual operator draws for one 512-update supervised U-TV run.}

\section{Training procedures}
All models start from random initialization. DWTDNet-tiny has 124,784
parameters; DRUNet has 32,638,656. Gradient-step versions use LION's wrapper
\begin{equation}
 g_\theta(x)=\tfrac12\|x-N_\theta(x)\|^2,
 \qquad D_\theta(x)=x-\nabla g_\theta(x),
\end{equation}
with second-order differentiation during training, following the structural
GS-PnP construction~\cite{gs}. ReLU activations are replaced by Softplus in
GS-DRUNet. We did not certify a global Lipschitz constant, and do not infer a
convergence guarantee merely from this construction. DWTDNet's current
implementation ignores its sigma argument; DRUNet receives the sampled
Gaussian/dark SD, not the true residual RMS.

Each denoiser arm uses Adam, learning rate $2\cdot10^{-4}$, effective batch
size two (gradient accumulation), gradient-norm clipping at one, 128 updates,
and validation every 32 updates. Only trained checkpoints compete; the
untrained network is retained as a baseline. Selection minimizes common
validation image MSE. Validation/test corruption panels use dark SDs
$.05,.2,.8$, beam innovation SDs $.00012,.0006,.001$, and both image-noise
families. Each split contains four distinct crops and 24 corruption cases.

U-TV and U-TGV use the existing adaptive parameter-map architectures with
6993 and 7001 parameters. The shared scale key is zero, so one network accepts
variable operators. Both use adjoint inputs and initializations, 64 unrolled
PDHG iterations, primal/dual steps 0.2, and 512 optimizer updates with batch
two. Positive maps are parameterized with Softplus, starting from raw scale
$-4$. The optimizer and clipping match the denoiser pilots.

Supervised loss is $\|\hat x_\theta-x\|^2/N$. SSDU~\cite{ssdu} partitions
one already-generated acquisition into input rows $\Theta$ and 20\% held-out
loss rows $\Lambda$, with disjoint sets and DC on the input side:
\begin{equation}
 \hat x_\theta=R_\theta(A_\Theta,y_\Theta),\qquad
 L_{\rm SSDU}=|\Lambda|^{-1}\|A_\Lambda\hat x_\theta-y_\Lambda\|^2.
\end{equation}
Neither the SSDU training loss nor its checkpoint selection uses clean-image
targets. SSDU selects checkpoints by held-out measurement validation loss;
clean validation MSE is logged only as an audit metric. At inference, all
available rows are supplied. Because ARIMA/bias noise may correlate across
these partitions, this loss is not asserted unbiased or equivalent to
supervised training. Reconstruction panels use rates 1\%, 20\%, and 100\%
and three noise severities, giving 36 cases on four source crops.

\section{Denoiser results}
\begin{center}@@DENOISER_TABLE@@\end{center}
Values are mean PSNR in dB on PCM-like \emph{image} corruption across three
severities and four held-out source crops. Positive change means mixed-noise
training exceeded Gaussian-only training for that architecture. We do not
rank architectures as fully trained models; 128 updates is an engineering
pilot and initialization/optimization effects remain large.
\pilotfigure{denoiser_learning.png}{0.88}{Validation curves from stored checkpoints. Dashed lines: Gaussian-only training; solid lines: mixed noise.}
\pilotfigure{denoising_examples.png}{1.0}{Fixed natural and synthetic held-out crops, medium raster-noise severity. All image panels share display range $[0,1]$; quantitative metrics are unclipped.}

\section{Adaptive regularizer results}
\begin{center}@@REGULARIZER_TABLE@@\end{center}
This table is the 20\%-sampling, medium physical-noise panel. Supervised and
SSDU runs used identical training image/acquisition draws before the SSDU
partition. The SSDU checkpoint ranking remained measurement-only. The SSDU
pilots do not support a positive performance claim yet; their deficits may
involve the loss, corruption correlations, limited optimization or finite
unrolling, rather than a single demonstrated cause.
\pilotfigure{regularizer_learning.png}{0.86}{Image-domain validation error. For SSDU this curve is diagnostic and did not select the checkpoint.}
\pilotfigure{regularizer_examples.png}{1.0}{Medium physical noise at 20\% sampling on fixed test crops. No figures were selected by reconstruction quality.}

\section{PnP search and held-out results}
The ordinary denoisers use the existing LION PnP-ADMM implementation with
zero initialization, exact normalized PCM proximal updates, and
$\rho\in\{.01,.1,1,10\}$. For GS denoisers, we use proximal gradient on
$F(x)=\tfrac12\|Ax-y\|^2+\lambda g_\theta(x)$ with
$\lambda\in\{.01,.1,1\}$ and initial step $\tau_0\in\{.5,2\}$.
The data proximal is the existing LION closed form with $\rho=1/\tau$.
For DRUNet-based priors we test denoiser sigma $.05$ and $.2$. We do not
sweep sigma for DWTDNet because that argument is ignored.

Each candidate uses the same validation acquisitions in six conditions:
three sampling rates and dark SD $.05$ or $.2$, at beam innovation SD $.0006$.
The search budget is at most 20 iterations. The best parameter setting per
method/condition is chosen only from validation MSE and is then applied to
four separate test crops. We do not tune the number of iterations or choose
best test iterates. Ordinary ADMM and GS-PG have different specified
initializations; this is not a controlled comparison of only the denoiser.

\begin{center}\small @@PNP_TABLE@@\end{center}
The table shows the medium-noise 20\% condition. Seconds are synchronized
inference times per crop on the RTX 4090, not the number of iterations.
GS uses additional gradient evaluations and backtracking, and may finish
before its budget. The complete 24 selected settings are in
\texttt{pnp\_selected.csv}; all candidate scores are retained.
\pilotfigure{pnp_test_psnr.png}{0.86}{Held-out results after validation-only selection; at most 20 iterations.}
\pilotfigure{pnp_examples.png}{1.0}{Fixed medium-noise 20\%-sampling test cases for the validation-selected PnP configurations.}

\section{Validation, numerical issues and limitations}\label{sec:validation}
Twenty-eight new CPU tests cover coefficient ranges, sampling endpoints,
invalid coarse levels, exact zero-noise identities in float64, disjoint
SSDU splits, reused noisy measurements, ADMM identity-prior recovery and
GS objective descent. All eight model types passed finite, nonzero-gradient
forward/backward smoke checks on the local GPU before training.

The first PnP sweep stopped at a full-sampling GS-DWTDNet case: after several
accepted updates, float32 proximal/denoiser rounding prevented further
Armijo acceptance. That partial run is preserved under
\texttt{pnp\_sweep\_initial\_line\_search\_failure}. The final sweep adds a
relative-step stagnation threshold $10^{-6}$, keeps the last accepted iterate,
and logs the stopping reason. It completed all candidate cases with zero
exceptions; @@STAGNATION_STOPS@@ of the 96 selected test runs stopped on that
numerical criterion. This is not a proof of stationarity or convergence.

\pilotfigure{noise_scaling.png}{0.84}{Measured image-surrogate component RMS for eight matched parameter draws on a constant image of value 0.5. Integration and drift make the same parameter bounds resolution dependent.}
The broad dark-noise range dominates the raster surrogate at $64\times64$.
Beam and especially drift components increase with trace length. Therefore
these crop-trained denoisers are not established as robust on full-size
$256\times256$ or $1024\times1024$ PCM images. A larger test cohort, multiple
training seeds, longer training, and a noise-calibration/scale check are
required before making paper claims. WLS/GLS reconstruction was not evaluated
in this run. The known multilevel-allocation issue is isolated by the new
sampling wrapper; existing benchmark implementations were left unchanged.

\section{Run accounting and reproducibility}
\begin{center}\small @@ALL_TABLE@@\end{center}
G denotes Gaussian-only training and M mixed-noise training. The aggregate
test metrics cover all three severities; denoising includes both corruption
families, reconstruction includes all three rates. Run seconds include
validation, checkpoint I/O and test evaluation. DWT/TV training used the RTX
4090 and DRUNet/TGV training the RTX 5090; those run times are not a fair
same-hardware architecture-speed comparison. Training used PyTorch 2.13.0
and the local \texttt{lion\_2} environment. Nothing was executed on a remote
training service.

All experiment files are rooted at:
\begin{lstlisting}
@@RESULT_ROOT@@
\end{lstlisting}
The root contains the immutable data manifest, run configurations,
\texttt{draws.jsonl}, validation/test CSVs, \texttt{best.pt}/\texttt{last.pt},
training curves, PnP search records and noise audits. Optimizer/RNG states
are included in checkpoints; the current CLI starts new output directories
rather than automatically resuming or overwriting previous runs.
\begin{lstlisting}
export PYTHONPATH=.
export LION_DATA_PATH=/mnt/store_ext4/GIT/LION/data
export LION_TTV22_PATH=/mnt/store_ext4/store/LION/ttv22
python scripts/experiments/pcm/randomized_noise.py --help
python -m LION.pcm.randomized.sweep --help
python -m LION.pcm.randomized.audit --root <result_root>
python -m LION.pcm.randomized.figures --root <result_root>
python -m LION.pcm.randomized.report --root <result_root>
\end{lstlisting}
\begin{thebibliography}{9}
\bibitem{gs} S. Hurault, A. Leclaire and N. Papadakis, \emph{Gradient Step Denoiser for convergent Plug-and-Play}, ICLR 2022. Section 3 describes the gradient-step construction. \url{https://arxiv.org/abs/2110.03220}.
\bibitem{ssdu} B. Yaman et al., \emph{Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data}, Magnetic Resonance in Medicine, 2020. See the data-partition training strategy. \url{https://arxiv.org/abs/1912.07669}.
\bibitem{div2k} E. Agustsson and R. Timofte, \emph{NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study}, CVPR Workshops, 2017. Dataset and academic-use conditions: \url{https://data.vision.ee.ethz.ch/cvl/DIV2K/}.
\end{thebibliography}
\end{document}
"""

    frames = r"""\begin{frame}{What ran: randomized-noise pilot}
\textbf{12 completed training runs; 24 validation-selected PnP settings.}
\begin{itemize}
\item Four denoisers, each with Gaussian-only and mixed-noise training: 128 updates.
\item Supervised and SSDU U-TV/U-TGV: 512 updates, 64 unrolled PDHG iterations.
\item @@VALIDATION_RECONS@@ validation and @@TEST_RECONS@@ held-out PnP reconstructions.
\end{itemize}
\begin{alertblock}{Scope}
One seed, $64\times64$ crops, four source images per evaluation split. Engineering evidence, not final paper results.
\end{alertblock}
\note{All jobs ran in the Linux Mint lion\_2 environment. No pretrained downloads or remote training. Best and last checkpoints and all noise draws are saved.}
\end{frame}

\begin{frame}{Two different corruption problems}
\textbf{Denoising: no PCM operator and no subsampling.}
\[z_t=x_t(1+b_t)+\overline I(tg+\eta_t)+\delta_t,\qquad \overline I=\operatorname{mean}(x)/0.3.\]
Mixed training draws Gaussian noise or this raster surrogate with equal probability.
\medskip
\textbf{Reconstruction: existing physical differential PCM simulator.}
\[ y=SPW x+e,\qquad r\sim\mathcal U(.01,1),\qquad m=\lceil rN\rceil.\]
DC is retained, coarseJ is valid, and noisy inputs are not clipped.
\note{The raster surrogate is not the distribution of image-space backprojected PCM noise. The physical simulator applies noise before differencing. The dark sigma range zero to one is a stress test, not an ampere calibration.}
\end{frame}

\begin{frame}{Randomized noise and data}
\small
\begin{columns}[T]
\column{.49\textwidth}
\begin{tabular}{lr}\toprule
Parameter & Uniform interval\\\midrule
$\sigma_g$ & $[0,3\cdot10^{-7}]$\\
$\sigma_{\rm bias}$ & $[0,4\cdot10^{-5}]$\\
$\sigma_{\rm beam}$ & $[0,10^{-3}]$\\
$\sigma_{\rm dark}$ & $[0,1]$\\\bottomrule
\end{tabular}
\medskip
AR/MA coefficients lie between two-decimal floor/ceiling. $d=1$ stays fixed.
\column{.49\textwidth}
\begin{tabular}{lrrr}\toprule
Pool & Train & Val & Test\\\midrule
Natural & 800 & 100 & 68\\
PCM & 4000 & 500 & 500\\\bottomrule
\end{tabular}
\medskip
800 duplicate files removed. Synthetic IDs grouped across categories.
\end{columns}
\medskip
Only @@DENOISER_DISTINCT@@ distinct sources were drawn per denoiser pilot;
@@REGULARIZER_DISTINCT@@ per regularizer pilot. Indexed data count is not training coverage.
\note{Train and evaluation source hashes are disjoint. Template ancestry is not known. Both domains receive equal sampling probability. The fixed test panel uses two natural and two synthetic images.}
\end{frame}

\begin{frame}{Denoiser result: mixed noise is not uniformly better}
\centering\includegraphics[height=.64\textheight]{figures/denoiser_test_psnr.png}
\par\small Fixed PCM-like test corruption. Improvement for @@BETTER_COUNT@@ of four architectures in this small pilot.
\note{The comparison uses matched seeds, crops, 128 updates, Gaussian sigma distribution and clean validation selection. GS-DWTDNet and GS-DRUNet use a gradient-of-potential architecture; no global Lipschitz certificate was established.}
\end{frame}

\begin{frame}{Denoising examples: fixed held-out crops}
\centering\includegraphics[width=.99\linewidth]{figures/denoising_examples.png}
\par\small Medium raster-noise severity. Common display range $[0,1]$; metrics remain unclipped.
\note{The first hash-sorted crop in each domain is displayed. These examples were not chosen because a particular method looked better. Checkpoint selection used validation images, not this test panel.}
\end{frame}

\begin{frame}{Adaptive TV/TGV: supervised beats this SSDU pilot}
\centering\includegraphics[height=.65\textheight]{figures/regularizer_test_psnr.png}
\par\small 512 optimizer updates, 64 PDHG steps. SSDU selects checkpoints using held-out measurement loss only.
\note{The graph uses dark sigma .2 and beam innovation sigma .0006. One acquisition is split, and correlated noise can cross the partition. We do not attribute the deficit to a single established cause.}
\end{frame}

\begin{frame}{Adaptive regularizer examples}
\centering\includegraphics[width=.99\linewidth]{figures/regularizer_examples.png}
\par\small 20\% sampling; medium physical noise; fixed natural and synthetic test crops.
\note{This is reconstruction from noisy Hadamard measurements, unlike the previous image-denoising examples. The initial image is the zero-filled adjoint.}
\end{frame}

\begin{frame}{PnP tuning: select on validation, then test}
\small
\begin{itemize}
\item ADMM: $\rho\in\{.01,.1,1,10\}$; GS-PG: $\lambda\in\{.01,.1,1\}$ and $\tau_0\in\{.5,2\}$.
\item DRUNet conditioning sigma $.05$ or $.2$; DWTDNet ignores sigma, so it is not swept.
\item Rates 1\%, 20\%, 100\%; two noise levels; same acquisitions for all candidates.
\end{itemize}
\begin{center}\scriptsize @@PNP_TABLE@@\end{center}
\small Table: held-out 20\% sampling, dark sigma .2. At most 20 iterations.
\note{Seconds are inference time per crop on RTX4090. They do not mean iteration count. GS can stop on numerical stagnation and requires extra gradient/backtracking evaluations. Exact chosen settings for all conditions are saved to CSV/JSON.}
\end{frame}

\begin{frame}{PnP: held-out reconstruction curves}
\centering\includegraphics[height=.66\textheight]{figures/pnp_test_psnr.png}
\par\small No best-test-iterate selection. @@STAGNATION_STOPS@@/96 selected runs reached the numerical step threshold.
\note{A first sweep failed near float32 stagnation and was preserved. The corrected protocol retains the last accepted iterate when relative change reaches 1e-6 and logs the stopping reason. This is not a convergence proof.}
\end{frame}

\begin{frame}{Noise strength depends on image size}
\centering\includegraphics[height=.66\textheight]{figures/noise_scaling.png}
\par\small Fixed parameter bounds do not imply fixed marginal beam or drift noise.
\note{The audit holds mean intensity at .5 and reuses eight parameter seeds for sides64,256,1024. Dark noise dominates small crops, while integration and linear drift grow with raster length. This must be addressed before claiming full-resolution robustness.}
\end{frame}

\begin{frame}{What this does and does not establish}
\begin{block}{Completed}
Randomized data/noise pipeline, twelve finite-gradient training runs,
validation-only PnP selection, saved checkpoints, figures and raw metrics.
\end{block}
\begin{alertblock}{Not yet established}
Converged training, broad held-out accuracy, template-independent synthetic
generalization, unbiased SSDU under correlated noise, or GS convergence certificates.
\end{alertblock}
\small Next evidence threshold: larger held-out cohort, multiple seeds and full-resolution noise calibration before a paper claim.
\note{The code and report clearly distinguish tested pilot configurations from a production training campaign. No existing paper results were overwritten.}
\end{frame}

\begin{frame}{References and reproducibility}
\small
Hurault, Leclaire and Papadakis, \emph{Gradient Step Denoiser for convergent Plug-and-Play}, ICLR 2022.\par
Yaman et al., \emph{Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data}, MRM 2020.\par
Agustsson and Timofte, \emph{NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study}, CVPR Workshops 2017.\par
\medskip
\texttt{LION/pcm/randomized/}: simulation, training, PnP search, audits and report generation.\par
\texttt{randomized\_noise\_20261014\_pilot/}: manifests, checkpoints, per-case CSVs and figures.\par
\medskip
28 targeted integrity tests plus the existing regression suite; all reported results are from local completed jobs.
\end{frame}
"""
    slides = r"""\documentclass[9pt,aspectratio=169]{beamer}
\IfFileExists{beamerthemecambridge.sty}{\usetheme{cambridge}}{\usetheme{Boadilla}}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\usepackage[T1]{fontenc}
\usepackage{lmodern,graphicx,booktabs,amsmath}
\title{Randomized-noise PCM experiments}
\subtitle{A3SmartML working update: bounded pilot study}
\author{PCM / LION}
\date{14 October 2026}
\begin{document}
\begin{frame}\titlepage\end{frame}
\input{randomized_noise_frames.tex}
\end{document}
"""
    manuscript = r"""% Include-ready subsection. Compile report.tex for the full protocol and limitations.
\subsection{Randomized-noise pilot study}
We conducted a bounded pilot of mixed Gaussian/PCM-like denoiser training,
adaptive TV/TGV learning under physical PCM measurements, and validation-only
PnP tuning. All clean inputs were $64\times64$ grayscale crops normalized to
$[0,1]$. The indexed pool contains 800 unique natural and 4000 synthetic PCM
training images; separate natural/synthetic validation and test pools were
used. The fixed evaluation panel contains only four source images per split.

For denoising, the PCM-like corruption is a raster-domain surrogate without
Hadamard sensing or subsampling. For reconstruction, the existing physical
split-pattern simulator is used, with randomized rates from 1\% to 100\%
and valid coarse levels. Noise standard deviations are sampled over
$[0,3\cdot10^{-7}]$, $[0,4\cdot10^{-5}]$, $[0,10^{-3}]$ and $[0,1]$
for bias drift, white bias, beam innovations and dark noise, respectively.
AR/MA coefficients vary within their two-decimal floor/ceiling intervals and
integration order remains one.

All four denoiser architectures were trained from scratch for 128 updates
under Gaussian-only and mixed-noise controls. Supervised and SSDU U-TV/U-TGV
were trained for 512 updates with 64 PDHG iterations. SSDU training and
checkpoint selection use held-out measurements; correlated-noise unbiasedness
is not assumed. The PnP search evaluated @@VALIDATION_RECONS@@ validation
reconstructions and @@TEST_RECONS@@ separate test reconstructions.

Mixed-noise training improved PCM-like test PSNR for @@BETTER_COUNT@@ of the
four denoiser architectures, not uniformly. Supervised regularizers exceeded
these SSDU pilots. These preliminary results do not establish converged or
full-resolution performance; larger test cohorts, multiple seeds and
resolution-dependent noise calibration remain necessary. The full protocol,
per-case metrics and image comparisons are in the accompanying working report.
"""
    for key, value in values.items():
        article = article.replace(key, value)
        frames = frames.replace(key, value)
        manuscript = manuscript.replace(key, value)
    for filename, text in (
        ("report.tex", article),
        ("randomized_noise_frames.tex", frames),
        ("slides.tex", slides),
        ("manuscript_section.tex", manuscript),
    ):
        if "@@" in text:
            raise ValueError(f"Unsubstituted report placeholder in {filename}")
        (bundle / filename).write_text(text)
    print(
        f"Wrote report.tex, slides.tex, manuscript_section.tex and include-ready frames to {bundle}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)
