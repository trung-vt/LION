"""Evaluate adaptive regularizers on exactly the frozen PnP test acquisitions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from LION.pcm.randomized.data import MixedImageCorpus, write_json
from LION.pcm.randomized.figures import LABELS, finish
from LION.pcm.randomized.models import build_regularizer, regularized_reconstruction
from LION.pcm.randomized.report import table
from LION.pcm.randomized.runner import metrics, synchronize, write_csv
from LION.pcm.randomized.sweep import make_cases


def matched_evaluation(root: Path, device: str = "cuda:0") -> None:
    corpus = MixedImageCorpus(root / "data_manifest.json", 64)
    pnp = pd.read_csv(root / "pnp_sweep/test.csv")
    rows = []
    torch.set_num_threads(2)
    dev = torch.device(device)
    for name in ("utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu"):
        path = root / f"runs/{name}_physical_pcm/best.pt"
        checkpoint = torch.load(path, map_location=dev, weights_only=False)
        model = build_regularizer(
            name, 64, checkpoint["config"]["unroll_iterations"], dev
        ).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        for ratio in (0.01, 0.2, 1.0):
            for dark in (0.05, 0.2):
                for case, (clean, record, acquisition, info) in enumerate(
                    make_cases(corpus, "test", ratio, dark, 2, dev)
                ):
                    baseline = acquisition.op.adjoint(acquisition.measurement)
                    baseline_stats = metrics(baseline, clean)
                    previous = pnp[
                        (pnp.model == "dwtdnet")
                        & np.isclose(pnp.ratio, ratio)
                        & np.isclose(pnp.dark_sigma, dark)
                        & (pnp.case == case)
                    ].iloc[0]
                    assert previous.source_hash == record["sha256"]
                    assert abs(previous.baseline_mse - baseline_stats["mse"]) < 1e-7
                    synchronize(dev)
                    start = time.perf_counter()
                    with torch.no_grad():
                        reconstruction = regularized_reconstruction(
                            model, acquisition.op, acquisition.measurement
                        )
                    synchronize(dev)
                    elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "model": name,
                            "ratio": ratio,
                            "dark_sigma": dark,
                            "coarse_j": info["coarse_j"],
                            "domain": record["domain"],
                            "case": case,
                            "source_hash": record["sha256"],
                            **metrics(reconstruction, clean),
                            **{"baseline_" + k: v for k, v in baseline_stats.items()},
                            "seconds": elapsed,
                            "iterations_completed": checkpoint["config"][
                                "unroll_iterations"
                            ],
                        }
                    )
        del model, checkpoint
        torch.cuda.empty_cache()
    regularizers = pd.DataFrame(rows)
    write_csv(root / "matched_regularizers.csv", rows)
    combined = pd.concat([regularizers, pnp], ignore_index=True)
    bundle = root / "report_bundle"
    combined.to_csv(bundle / "matched_test.csv", index=False)
    medium = combined[np.isclose(combined.dark_sigma, 0.2)]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for name in LABELS:
        selected = medium[medium.model == name].groupby("ratio").psnr.mean()
        label = LABELS[name]
        if not name.startswith("ut"):
            label = ("GS-PG / " if name.startswith("gs_") else "ADMM / ") + label
        ax.plot(
            selected.index.to_numpy() * 100,
            selected.to_numpy(),
            marker="o",
            linestyle="-" if name.startswith("ut") else "--",
            label=label,
        )
    baseline = medium[medium.model == "dwtdnet"].groupby("ratio").baseline_psnr.mean()
    ax.plot(
        baseline.index.to_numpy() * 100,
        baseline.to_numpy(),
        linestyle=":",
        marker="x",
        label="Zero-filled",
    )
    ax.set(
        xscale="log",
        xlabel="Sampled coefficients (%)",
        ylabel="Mean held-out PSNR (dB)",
        title="Matched test measurements: all methods, dark sigma 0.2",
    )
    ax.set_xticks([1, 20, 100], ["1", "20", "100"])
    ax.legend(fontsize=8, ncol=3)
    finish(fig, bundle / "figures", "matched_test_psnr")
    selected = medium[np.isclose(medium.ratio, 0.2)]
    result_rows = []
    for name in (
        "utv_supervised",
        "utgv_supervised",
        "utv_ssdu",
        "utgv_ssdu",
        "dwtdnet",
        "gs_dwtdnet",
        "drunet",
        "gs_drunet",
    ):
        values = selected[selected.model == name]
        label = (
            LABELS[name]
            if name.startswith("ut")
            else ("GS-PG / " if name.startswith("gs_") else "ADMM / ") + LABELS[name]
        )
        result_rows.append(
            [
                label,
                f"{values.psnr.mean():.2f}",
                f"{values.ssim.mean():.3f}",
                f"{values.seconds.mean():.3f}",
            ]
        )
    result_rows.append(["Zero-filled", f"{baseline.loc[0.2]:.2f}", "--", "--"])
    tbl = table(["Method", "PSNR", "SSIM", "Seconds"], result_rows, "lrrr")
    section = (
        r"""\section{Matched-acquisition comparison}
The earlier regularizer and PnP panels use different noise seed families and
coarse-level choices, so their values should not be directly ranked across
panels. We therefore evaluated the four fixed adaptive checkpoints on the
exact PnP test acquisitions. This adds 96 adaptive reconstructions without
retraining or changing checkpoint selection. Source hashes and baseline
measurement statistics were checked against the PnP records.
\begin{center}\small
"""
        + tbl
        + r"""
\end{center}
This table uses 20\% sampling and dark SD 0.2, averaged across the same four
held-out crops. Hardware is RTX4090. Adaptive methods use 64 PDHG steps;
PnP uses at most 20 iterations, so iteration budgets and training budgets
remain different. This is a matched-data pilot, not a resource-matched
benchmark. Timing includes the reconstruction itself, not model loading.
\pilotfigure{matched_test_psnr.png}{0.90}{Matched test measurements for all methods. No test-driven retuning or choice of best test iterate was performed.}
"""
    )
    (bundle / "matched_benchmark.tex").write_text(section)
    frame = r"""\begin{frame}{Matched measurements: a fairer data comparison}
\centering\includegraphics[height=.60\textheight]{figures/matched_test_psnr.png}
\par\small Same four test crops, masks and noise. Different training/iteration budgets; no test-driven retuning.
\note{This additional evaluation checks identical source hashes and zero-filled residual statistics. Adaptive methods use64 PDHG steps; PnP uses up to20. Comparing these curves is valid as a matched-data pilot, not a resource-matched final benchmark.}
\end{frame}
"""
    (bundle / "matched_frame.tex").write_text(frame)
    report_path = bundle / "report.tex"
    report = report_path.read_text()
    marker = r"\section{Run accounting and reproducibility}"
    include = r"\input{matched_benchmark.tex}"
    if include not in report:
        assert marker in report
        report_path.write_text(report.replace(marker, include + "\n" + marker))
    frames_path = bundle / "randomized_noise_frames.tex"
    frames = frames_path.read_text()
    marker = r"\begin{frame}{What this does and does not establish}"
    include = r"\input{matched_frame.tex}"
    if include not in frames:
        assert marker in frames
        frames_path.write_text(frames.replace(marker, include + "\n" + marker))
    write_json(
        bundle / "matched_summary.json",
        {
            "additional_reconstructions": len(rows),
            "test_cases": regularizers.source_hash.nunique(),
            "same_source_hashes_and_baseline_metrics": True,
            "medium_20_percent_table": result_rows,
        },
    )
    print(
        f"Completed {len(rows)} matched adaptive reconstructions; report/slide sections added."
    )
    print(tbl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    matched_evaluation(args.root, args.device)
