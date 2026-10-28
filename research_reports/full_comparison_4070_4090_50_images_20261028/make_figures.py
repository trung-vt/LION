"""Report figures; vector charts and individual 256x256 image panels."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path("/store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028")
F = O / "figures"
F.mkdir(exist_ok=True)
s = json.loads((O / "summary_50.json").read_text())
p = json.loads((O / "protocol_manifest_50.json").read_text())
C = p["categories"]
R = p["balanced_noise_conditions"]
K = ("mse", "psnr", "ssim", "haarpsi")
rows = list(csv.DictReader((O / "all_paired_results_old_new.csv").open()))


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(F / f"{name}.png", bbox_inches="tight", pad_inches=0.15, dpi=135)
    plt.close(fig)
    print("FIGURE", name, flush=True)


# Metric evolution for four frozen checkpoints.
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")
models = ("4070_old", "4070_new", "4090_old", "4090_new")
for k, a in zip(K, axes.flat):
    values = [s["means"][m][k]["balanced_50_50"]["mean"] for m in models]
    a.bar(np.arange(4), values, width=0.7)
    a.set_xticks(
        range(4),
        ["4070 old", "4070 new", "4090 old", "4090 new"],
        rotation=16,
        fontsize=8,
    )
    a.set_title(k.upper() + (" (lower better)" if k == "mse" else " (higher better)"))
    a.grid(axis="y", alpha=0.2)
    for i, v in enumerate(values):
        a.annotate(
            f"{v:.5g}",
            (i, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    if k == "mse":
        a.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
fig.suptitle("Four checkpoints evaluated on the same 100 acquisitions", fontsize=14)
save(fig, "four_checkpoint_means")
# Category-stratified paired bootstrap, five comparisons.
fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
contrasts = (
    "4070_new_minus_old",
    "4090_new_minus_old",
    "old_4090_minus_old_4070",
    "new_4090_minus_new_4070",
    "head_to_head_delta_change",
)
labels = (
    "4070 new - old",
    "4090 new - old",
    "Old 4090 - 4070",
    "New 4090 - 4070",
    "Change in model gap",
)
for k, a in zip(K, axes.flat):
    for i, key in enumerate(contrasts):
        z = s["comparisons"][key][k]["balanced_50_50"]
        v = z["mean"]
        lo, hi = z["ci95"]
        a.errorbar(v, 4 - i, xerr=[[v - lo], [hi - v]], fmt="o", capsize=3)
    a.set_yticks(range(4, -1, -1), labels, fontsize=8)
    a.set_xlabel("Paired difference in " + k.upper())
    a.axvline(0, color="grey", lw=0.9)
    a.grid(axis="x", alpha=0.17)
    if k == "mse":
        a.ticklabel_format(axis="x", style="sci", scilimits=(-4, -4))
fig.suptitle("Paired effect estimates and 95% confidence intervals", fontsize=14)
save(fig, "paired_effect_confidence_intervals")
# Model gaps per category, separately for both noise regimes.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for k, a in zip(K, axes.flat):
    pos = np.arange(5)
    for ri, reg in enumerate(R):
        v = [s["by_category"][cat][reg]["delta_4090_minus_4070_new"][k] for cat in C]
        a.bar(
            pos + (0.2 if ri else -0.2),
            v,
            width=0.38,
            label=("Zero-inclusive" if ri == 0 else "Positive noise"),
        )
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(pos, C, rotation=18, fontsize=9)
    a.set_title(k.upper() + ": current 4090 minus 4070", fontsize=11)
    a.grid(axis="y", alpha=0.2)
    if k == "mse":
        a.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
axes[0, 0].legend(fontsize=8)
fig.suptitle("Category-level differences on paired test acquisitions", fontsize=14)
save(fig, "category_by_noise")
# Matched old/current scatter on 100 acquisitions per model.
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), layout="constrained")
for mi, model in enumerate(("4070", "4090")):
    for ki, k in enumerate(("mse", "haarpsi")):
        a = axes[mi, ki]
        old = np.array([float(r[f"old_{model}_{k}"]) for r in rows])
        new = np.array([float(r[f"new_{model}_{k}"]) for r in rows])
        for reg in R:
            mask = np.array([r["condition"] == reg for r in rows])
            a.scatter(old[mask], new[mask], s=16, alpha=0.68, label=reg)
        lo = min(old.min(), new.min())
        hi = max(old.max(), new.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_xlabel("Old " + k.upper())
        a.set_ylabel("New " + k.upper())
        a.set_title(model + " checkpoint")
        a.grid(alpha=0.17)
        if mi == ki == 0:
            a.legend(fontsize=7)
fig.suptitle(
    "Per-acquisition checkpoint progress; diagonal indicates no change", fontsize=13
)
save(fig, "paired_old_new_scatter")
# MSE relative to sampling ratio, colored by category.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
for a, reg in zip(axes, R):
    for cat in C:
        subset = [r for r in rows if r["category"] == cat and r["condition"] == reg]
        a.scatter(
            [float(r["sampling_ratio"]) for r in subset],
            [float(r["new_4090_mse"]) - float(r["new_4070_mse"]) for r in subset],
            s=27,
            alpha=0.75,
            label=cat,
        )
    a.set_title(reg)
    a.set_xlabel("Sampling ratio")
    a.axhline(0, color="grey")
    a.grid(alpha=0.17)
    a.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
axes[0].set_ylabel("4090 new MSE minus 4070 new MSE")
axes[1].legend(fontsize=7)
fig.suptitle("Current per-image error differences by sampling ratio", fontsize=13)
save(fig, "mse_difference_vs_sampling")
# 10 precommitted visual examples (first selected ID in each category).
# Images remain independent embedded raster objects in a vector PDF plate.
for reg in R:
    fig, ax = plt.subplots(5, 4, figsize=(13.5, 15), layout="constrained")
    for ci, cat in enumerate(C):
        sid = p["subset_per_category"][cat][0]
        example = np.load(O / f"example_{cat}_{reg}.npz")
        arrays = [
            example["ground_truth"],
            example["zero_filled"],
            example["4070_current"],
            example["4090_current"],
        ]
        titles = ("Ground truth", "Zero-filled", "4070 current", "4090 current")
        metric = next(
            r
            for r in rows
            if r["category"] == cat
            and int(r["sample_id"]) == sid
            and r["condition"] == reg
        )
        for j, x in enumerate(arrays):
            img = np.real(x.squeeze())
            assert img.shape == (256, 256)
            im = ax[ci, j].imshow(
                img, cmap="afmhot", vmin=0, vmax=1.2, interpolation="nearest"
            )
            ax[ci, j].set_xticks([])
            ax[ci, j].set_yticks([])
            if ci == 0:
                ax[ci, j].set_title(titles[j], fontsize=11)
            if j == 0:
                ax[ci, j].set_ylabel(
                    f'{cat}\nID {sid}, ratio {float(metric["sampling_ratio"]):.2f}',
                    fontsize=9,
                )
            if j >= 2:
                mod = ("4070", "4090")[j - 2]
                val = float(metric[f"new_{mod}_psnr"])
                ax[ci, j].text(
                    0.5,
                    -0.04,
                    f"PSNR {val:.2f} dB",
                    transform=ax[ci, j].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
    fig.colorbar(
        im,
        ax=ax.ravel().tolist(),
        shrink=0.47,
        pad=0.014,
        label="Normalized intensity (common scale 0-1.2)",
    )
    fig.suptitle(f"Reconstruction examples: {reg} noise", fontsize=13)
    save(fig, "reconstructions_" + reg)
# Error maps have a strictly shared color scale in all panels.
for reg in R:
    fig, ax = plt.subplots(5, 2, figsize=(8.5, 15), layout="constrained")
    for ci, cat in enumerate(C):
        sid = p["subset_per_category"][cat][0]
        e = np.load(O / f"example_{cat}_{reg}.npz")
        target = np.real(e["ground_truth"].squeeze())
        for j, model in enumerate(("4070", "4090")):
            error = np.abs(np.real(e[f"{model}_current"].squeeze()) - target)
            im = ax[ci, j].imshow(
                error, cmap="viridis", vmin=0, vmax=0.5, interpolation="nearest"
            )
            ax[ci, j].set_xticks([])
            ax[ci, j].set_yticks([])
            if ci == 0:
                ax[ci, j].set_title(f"{model} current - absolute error", fontsize=10)
            if j == 0:
                ax[ci, j].set_ylabel(f"{cat}\nID {sid}", fontsize=10)
    fig.colorbar(
        im,
        ax=ax.ravel().tolist(),
        shrink=0.48,
        pad=0.016,
        label="Absolute error, shared scale 0-0.5",
    )
    fig.suptitle(f"Absolute reconstruction error maps: {reg}", fontsize=13)
    save(fig, "absolute_error_" + reg)
print("ALL_FIGURES_COMPLETE", len(list(F.glob("*.pdf"))))
