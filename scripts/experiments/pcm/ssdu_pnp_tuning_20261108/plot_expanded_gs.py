"""Plot only validation SSDU losses and diagnostics after expanded GS scoring."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import (
    REPORT,
    RESULTS,
    get_grid,
    prepared_record,
)


def plot():
    prepared_record()
    grid, digest = get_grid()
    valid = {}
    for cfg in grid:
        path = RESULTS / "candidates" / cfg["candidate_id"] / "summary.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record["candidate"] == cfg and record["grid_sha256"] == digest
            valid[cfg["candidate_id"]] = record
    if len(valid) < 2:
        print("PLOTS_NOT_YET_AVAILABLE_ONLY_CONTROL", len(valid), flush=True)
        return
    figs = REPORT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}
    )
    x = np.arange(len(grid))
    y = np.array(
        [
            valid[c["candidate_id"]]["loss"] if c["candidate_id"] in valid else np.nan
            for c in grid
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    ax.plot(x[np.isfinite(y)], y[np.isfinite(y)], "o", markersize=5)
    ax.set_xticks(x, [str(i + 1) for i in x], fontsize=7)
    ax.set_xlabel("Frozen candidate index (01 through 27)")
    ax.set_ylabel("Held-out SSDU MSE")
    ax.set_title("Expanded GS-PnP five-image measurement-only score")
    ax.grid(axis="y", alpha=0.24)
    fig.savefig(figs / "candidate_losses.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, lambd in zip(axes, (0.1, 0.2, 0.4)):
        matrix = np.full((3, 3), np.nan)
        for i, sigma in enumerate((0.075, 0.1, 0.15)):
            for j, tau in enumerate((100 / 3, 50.0, 75.0)):
                matching = [
                    c
                    for c in grid
                    if (c["lambda"], c["sigma"], c["tau0"]) == (lambd, sigma, tau)
                ]
                assert len(matching) == 1
                id = matching[0]["candidate_id"]
                if id in valid:
                    matrix[i, j] = valid[id]["loss"]
        mask = np.ma.masked_invalid(matrix)
        image = ax.imshow(mask, aspect="auto", interpolation="nearest")
        ax.set_title(f"lambda={lambd:.2f}")
        ax.set_xticks((0, 1, 2), ("33.33", "50", "75"))
        ax.set_yticks((0, 1, 2), ("0.075", "0.10", "0.15"))
        ax.set_xlabel("initial tau0")
        ax.set_ylabel("denoiser sigma")
        for i in range(3):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j, i, f"{matrix[i,j]:.4g}", ha="center", va="center", fontsize=7
                    )
                else:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.043, pad=0.04, label="SSDU MSE")
    fig.suptitle("Measurement risk over expanded GS-PnP lambda, sigma, tau0")
    fig.savefig(figs / "parameter_slices.pdf", bbox_inches="tight")
    plt.close(fig)
    print("EXPANDED_GS_FIGURES_READY", len(valid), "candidates", flush=True)


if __name__ == "__main__":
    plot()
