"""Generate pilot-report tables and figures from stored experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LION.pcm.randomized.monitoring import image_grid

from LION.pcm.randomized.data import write_json


LABELS = {
    "dwtdnet": "DWTDNet",
    "gs_dwtdnet": "GS-DWTDNet",
    "drunet": "DRUNet",
    "gs_drunet": "GS-DRUNet",
    "utv_supervised": "U-TV supervised",
    "utgv_supervised": "U-TGV supervised",
    "utv_ssdu": "U-TV SSDU",
    "utgv_ssdu": "U-TGV SSDU",
}


def finish(fig, directory: Path, name: str):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(root: Path) -> dict:
    out = root / "report_bundle"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    runs, summary, denoising_rows, regularizer_rows = {}, [], [], []
    for directory in sorted((root / "runs").iterdir()):
        if not (directory / "run.json").exists():
            continue
        record = json.loads((directory / "run.json").read_text())
        if record["status"] != "completed":
            raise RuntimeError(f"Run is not complete: {directory}")
        config = record["config"]
        test = pd.read_csv(directory / "test.csv")
        runs[directory.name] = record
        row = {
            "run": directory.name,
            "model": config["model"],
            "noise": config["noise_mode"],
            "updates": record["completed_updates"],
            "best_step": record["best_step"],
            "parameters": record["parameters"],
            "seconds": record["elapsed_seconds"],
            "gpu": record["device_name"],
            "test_mse": test.mse.mean(),
            "test_psnr": test.psnr.mean(),
            "test_ssim": test.ssim.mean(),
            "test_cases": test.source_hash.nunique(),
            "test_conditions": len(test),
            "initial_validation_mse": record["initial_validation_mse"],
            "selected_validation_mse": record["best_validation_mse"],
            "selection": record.get("selection_criterion", "clean_validation_mse"),
        }
        summary.append(row)
        (
            regularizer_rows if config["model"].startswith("ut") else denoising_rows
        ).append(test)
    if len(runs) != 12:
        raise RuntimeError(f"Expected 12 completed training pilots, got {len(runs)}")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "run_summary.csv", index=False)
    den = pd.concat(denoising_rows, ignore_index=True)
    reg = pd.concat(regularizer_rows, ignore_index=True)
    den.to_csv(out / "denoising_test.csv", index=False)
    reg.to_csv(out / "regularizer_test.csv", index=False)
    pnp_root = root / "pnp_sweep"
    pnp_protocol = json.loads((pnp_root / "protocol.json").read_text())
    if pnp_protocol["status"] != "completed":
        raise RuntimeError("PnP sweep is not complete")
    pnp = pd.read_csv(pnp_root / "test.csv")
    pnp.to_csv(out / "pnp_test.csv", index=False)
    selected = json.loads((pnp_root / "selected.json").read_text())
    pd.DataFrame(
        [
            {k: v for k, v in item.items() if k != "config"} | item["config"]
            for item in selected
        ]
    ).to_csv(out / "pnp_selected.csv", index=False)

    # Each plot has one axis, with actual validation checkpoints and held-out metrics.
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    for name in ("dwtdnet", "gs_dwtdnet", "drunet", "gs_drunet"):
        for mode in ("gaussian", "mixture"):
            history = json.loads(
                (root / "runs" / f"{name}_{mode}" / "history.json").read_text()
            )
            points = [r for r in history if r["validation_mse"] is not None]
            ax.plot(
                [p["step"] for p in points],
                [p["validation_mse"] for p in points],
                marker="o",
                linestyle="--" if mode == "gaussian" else "-",
                label=LABELS[name]
                + (" / Gaussian" if mode == "gaussian" else " / mixed"),
            )
    ax.set(
        yscale="log",
        xlabel="Optimizer updates",
        ylabel="Validation MSE (unclipped)",
        title="Denoiser pilots: matched 128-update budgets",
    )
    ax.legend(fontsize=8, ncol=2)
    finish(fig, figs, "denoiser_learning")

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    names = ["dwtdnet", "gs_dwtdnet", "drunet", "gs_drunet"]
    x = np.arange(4)
    subset = den[den.noise_kind == "pcm_like"]
    for offset, mode in ((-0.18, "gaussian"), (0.18, "mixture")):
        means = [
            subset[(subset.model == n) & (subset.training_noise == mode)].psnr.mean()
            for n in names
        ]
        ax.bar(
            x + offset,
            means,
            width=0.35,
            label="Gaussian-only training"
            if mode == "gaussian"
            else "Mixed-noise training",
        )
    ax.set_xticks(x, [LABELS[n] for n in names])
    ax.set(
        ylabel="Mean held-out PSNR (dB)",
        title="PCM-like image noise: four held-out source images, three severities",
    )
    ax.legend(fontsize=9)
    finish(fig, figs, "denoiser_test_psnr")

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for name in ("utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu"):
        history = json.loads(
            (root / "runs" / f"{name}_physical_pcm" / "history.json").read_text()
        )
        points = [r for r in history if r["validation_mse"] is not None]
        ax.plot(
            [p["step"] for p in points],
            [p["validation_mse"] for p in points],
            marker="o",
            label=LABELS[name],
        )
    ax.set(
        yscale="log",
        xlabel="Optimizer updates",
        ylabel="Validation image MSE (audit only for SSDU)",
        title="Adaptive regularizers: 64 unrolled PDHG iterations",
    )
    ax.legend(fontsize=9)
    finish(fig, figs, "regularizer_learning")

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    medium = reg[reg.severity == "medium"]
    for name in ("utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu"):
        group = medium[medium.model == name].groupby("sampling_ratio").psnr.mean()
        ax.plot(
            100 * group.index.to_numpy(),
            group.to_numpy(),
            marker="o",
            label=LABELS[name],
        )
    baseline = (
        medium[medium.model == "utv_supervised"]
        .groupby("sampling_ratio")
        .baseline_psnr.mean()
    )
    ax.plot(
        100 * baseline.index.to_numpy(),
        baseline.to_numpy(),
        linestyle="--",
        marker="x",
        label="Zero-filled",
    )
    ax.set(
        xscale="log",
        xlabel="Sampled coefficients (%)",
        ylabel="Mean held-out PSNR (dB)",
        title="Physical PCM noise: dark sigma 0.2, beam innovation sigma 0.0006",
    )
    ax.set_xticks([1, 20, 100], ["1", "20", "100"])
    ax.legend(fontsize=9)
    finish(fig, figs, "regularizer_test_psnr")

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    medium_pnp = pnp[np.isclose(pnp.dark_sigma, 0.2)]
    for name in names:
        group = medium_pnp[medium_pnp.model == name].groupby("ratio").psnr.mean()
        ax.plot(
            100 * group.index.to_numpy(),
            group.to_numpy(),
            marker="o",
            label=("GS-PG / " if name.startswith("gs_") else "ADMM / ") + LABELS[name],
        )
    base = (
        medium_pnp[medium_pnp.model == "dwtdnet"].groupby("ratio").baseline_psnr.mean()
    )
    ax.plot(
        100 * base.index.to_numpy(),
        base.to_numpy(),
        linestyle="--",
        marker="x",
        label="Zero-filled",
    )
    ax.set(
        xscale="log",
        xlabel="Sampled coefficients (%)",
        ylabel="Mean held-out PSNR (dB)",
        title="PnP: validation-selected parameters, at most 20 iterations",
    )
    ax.set_xticks([1, 20, 100], ["1", "20", "100"])
    ax.legend(fontsize=8)
    finish(fig, figs, "pnp_test_psnr")

    scaling = pd.read_csv(root / "image_noise_scaling.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for field, label in (
        ("dark_image_rms", "Additive dark"),
        ("beam_image_rms", "Multiplicative ARIMA beam"),
        ("bias_image_rms", "Bias and drift"),
    ):
        group = scaling.groupby("side")[field].mean()
        ax.plot(group.index, group.to_numpy(), marker="o", label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Image side length",
        ylabel="Mean component RMS (8 fixed parameter draws)",
        title="Image-domain surrogate: resolution changes integrated-noise strength",
    )
    ax.set_xticks([64, 256, 1024], ["64", "256", "1024"])
    ax.legend(fontsize=9)
    finish(fig, figs, "noise_scaling")

    draws = pd.read_json(
        root / "runs/utv_supervised_physical_pcm/draws.jsonl", lines=True
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.scatter(100 * draws.requested_ratio, draws.coarse_j, s=9, alpha=0.3)
    ax.set(
        xlabel="Requested sampling (%)",
        ylabel="Sampled coarseJ",
        title="Actual training draws: randomized valid operator configurations",
    )
    finish(fig, figs, "sampling_coverage")

    rows = []
    for domain, index in (("natural", 0), ("pcm_synthetic", 2)):
        key = f"{domain}_{index}"
        arrays = np.load(root / "runs/dwtdnet_mixture/examples.npz")
        cells = [
            ("Ground truth", arrays[key + "_clean"]),
            ("Noisy image", arrays[key + "_input"]),
        ]
        for name in names:
            with np.load(root / f"runs/{name}_mixture/examples.npz") as sample:
                np.testing.assert_array_equal(
                    sample[key + "_clean"], arrays[key + "_clean"]
                )
                np.testing.assert_array_equal(
                    sample[key + "_input"], arrays[key + "_input"]
                )
                cells.append((LABELS[name], sample[key + "_prediction"]))
        rows.append((domain.replace("_", " "), cells))
    image_grid(figs / "denoising_examples.png", rows)

    rows = []
    for domain, index in (("natural", 0), ("pcm_synthetic", 2)):
        key = f"{domain}_{index}"
        base = np.load(root / "runs/utv_supervised_physical_pcm/examples.npz")
        cells = [
            ("Ground truth", base[key + "_clean"]),
            ("Zero-filled", base[key + "_input"]),
        ]
        for name in ("utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu"):
            with np.load(root / f"runs/{name}_physical_pcm/examples.npz") as sample:
                np.testing.assert_array_equal(
                    sample[key + "_input"], base[key + "_input"]
                )
                cells.append((LABELS[name], sample[key + "_prediction"]))
        rows.append((domain.replace("_", " "), cells))
    image_grid(figs / "regularizer_examples.png", rows)

    rows = []
    for domain in ("natural", "pcm_synthetic"):
        base = np.load(pnp_root / f"dwtdnet_{domain}_example.npz")
        cells = [("Ground truth", base["clean"]), ("Zero-filled", base["initial"])]
        for name in names:
            with np.load(pnp_root / f"{name}_{domain}_example.npz") as sample:
                np.testing.assert_array_equal(sample["initial"], base["initial"])
                cells.append((LABELS[name], sample["reconstruction"]))
        rows.append((domain.replace("_", " "), cells))
    image_grid(figs / "pnp_examples.png", rows)
    results = {
        "run_summary": summary,
        "manifest_counts": json.loads((root / "data_manifest.json").read_text())[
            "counts"
        ],
        "duplicate_files_excluded": 800,
        "pnp_protocol": pnp_protocol,
        "pnp_stops": pnp.stop_reason.value_counts().to_dict(),
        "denoising_pcm_like_psnr": subset.groupby(["model", "training_noise"])
        .psnr.mean()
        .reset_index()
        .to_dict("records"),
        "regularizer_medium": medium.groupby(["model", "sampling_ratio"])[
            ["psnr", "baseline_psnr"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
        "pnp_medium": medium_pnp.groupby(["model", "ratio"])[
            ["psnr", "baseline_psnr", "seconds", "iterations_completed"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
        "noise_scaling": scaling.groupby("side")[
            ["dark_image_rms", "beam_image_rms", "bias_image_rms"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
    }
    write_json(out / "summary.json", results)
    print(
        f"Generated {len(list(figs.iterdir()))} figure files and summary tables in {out}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    make_figures(args.root)
