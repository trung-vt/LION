"""Publication-oriented tables and figures from the frozen unseen PCM benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LION.pcm.randomized.figure_style import image_style


METHOD_LABELS = {
    "zero_fill": "Zero-fill",
    "tv_fixed_002": "Scalar TV",
    "spgl1": "SPGL1",
    "utv_supervised": "Supervised U-TV",
    "utv_ssdu": "SSDU U-TV",
    "utgv_supervised": "Supervised U-TGV",
    "utgv_ssdu": "SSDU U-TGV",
    "pnp_admm_dwtdnet": "PnP-ADMM + DWTDNet",
    "gs_pnp_drunet": "GS-PnP + DRUNet",
}
SCENARIO_LABELS = {
    "noiseless": "Noiseless",
    "gaussian_dark": "Independent Gaussian dark noise",
    "pcm_default_10ms": "NPL default (nominal 10 ms)",
    "pcm_short_100us": "NPL extrapolated 100 us",
}
SELECTED = (
    "zero_fill",
    "spgl1",
    "tv_fixed_002",
    "utv_supervised",
    "utv_ssdu",
    "utgv_supervised",
    "utgv_ssdu",
    "pnp_admm_dwtdnet",
    "gs_pnp_drunet",
)


def generate(root: Path) -> dict:
    data = pd.read_csv(root / "reconstruction_metrics.csv")
    if len(data) != 216:
        raise ValueError(f"Expected 216 reconstructions, got {len(data)}")
    out = root / "analysis"
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    group = data.groupby(["scenario", "sampling_ratio", "method"], as_index=False)[
        ["psnr", "ssim", "mse", "seconds", "measurement_noise_rms"]
    ].mean()
    group.to_csv(out / "mean_metrics.csv", index=False)
    cells = data[(data.sampling_ratio == 0.2)][
        ["cell", "crop", "scenario", "method", "psnr", "ssim", "mse", "seconds"]
    ]
    cells.to_csv(out / "by_cell_at_20pct.csv", index=False)
    ref = group[group.method == "zero_fill"][
        ["scenario", "sampling_ratio", "psnr"]
    ].rename(columns={"psnr": "zero_fill_psnr"})
    compare = group.merge(ref, on=["scenario", "sampling_ratio"])
    compare["gain_over_zero_fill_db"] = compare.psnr - compare.zero_fill_psnr
    compare.to_csv(out / "gains_over_zero_fill.csv", index=False)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    for scenario in SCENARIO_LABELS:
        subset = group[group.scenario == scenario]
        for field, unit in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
            fig, ax = plt.subplots(figsize=(9, 4.9))
            for method in SELECTED:
                curve = subset[subset.method == method].sort_values("sampling_ratio")
                ax.plot(
                    curve.sampling_ratio * 100,
                    curve[field],
                    marker="o",
                    linewidth=1.9,
                    label=METHOD_LABELS[method],
                )
            ax.set(
                title=f"{SCENARIO_LABELS[scenario]}: unseen PCM map reconstructions",
                xlabel="Sampling (%)",
                ylabel=unit,
                xticks=[10, 20, 50],
            )
            ax.grid(alpha=0.24)
            ax.legend(
                ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8
            )
            save(fig, f"{field}_vs_sampling_{scenario}")
    for scenario in ("pcm_default_10ms", "pcm_short_100us"):
        subset = group[
            (group.scenario == scenario) & (np.isclose(group.sampling_ratio, 0.2))
        ]
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        labels = [METHOD_LABELS[m] for m in SELECTED]
        values = [float(subset[subset.method == m].psnr.iloc[0]) for m in SELECTED]
        ax.barh(labels[::-1], values[::-1])
        ax.set(
            xlabel="Mean PSNR over two cells (dB)",
            title=f"20% sampling: {SCENARIO_LABELS[scenario]}",
        )
        ax.grid(axis="x", alpha=0.2)
        save(fig, f"psnr_20pct_{scenario}")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for scenario, sub in data.groupby("scenario"):
        v = sub.groupby("sampling_ratio").measurement_noise_rms.mean()
        ax.plot(v.index * 100, v.values, marker="o", label=SCENARIO_LABELS[scenario])
    ax.set(
        xlabel="Sampling (%)",
        ylabel="Measurement noise RMS (normalized units)",
        title="Fixed physical-noise scenarios (identical mask across conditions)",
        xticks=[10, 20, 50],
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    save(fig, "noise_rms_vs_sampling")

    # Render the two originally measured maps with the SAME AFM palette as
    # reconstruction images. Read their saved numeric arrays, not older grayscale
    # previews or user-provided screenshots.
    import json

    doc = json.loads((root / "protocol.json").read_text())
    manifest = json.loads(Path(doc["source_manifest"]).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    cmap, vmin, vmax = image_style("reconstruction")
    plotted = None
    for axis, (cell, crop, label) in zip(
        axes,
        (
            ("cigs", "active_full", "CIGS: full active area"),
            ("perovskite", "center_256", "Perovskite: fixed centre crop"),
        ),
        strict=True,
    ):
        record = next(
            x for x in manifest["cases"] if x["cell"] == cell and x["label"] == crop
        )
        image = np.load(record["path"], allow_pickle=False)
        plotted = axis.imshow(
            image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        axis.set_title(label, fontsize=13, pad=10)
        axis.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.84, bottom=0.19, wspace=0.10)
    bar_axis = fig.add_axes((0.23, 0.09, 0.54, 0.026))
    colorbar = fig.colorbar(plotted, cax=bar_axis, orientation="horizontal")
    colorbar.set_label("Transformed current (relative units)", fontsize=11)
    fig.savefig(figdir / "normalized_source_pair.png", dpi=150, facecolor="white")
    plt.close(fig)

    return {
        "rows": len(data),
        "figures_png": len(list(figdir.glob("*.png"))),
        "tv_gain_20pct_pcm_default_db": float(
            compare[
                (compare.scenario == "pcm_default_10ms")
                & np.isclose(compare.sampling_ratio, 0.2)
                & (compare.method == "tv_fixed_002")
            ].gain_over_zero_fill_db.iloc[0]
        ),
        "utgv_supervised_gain_20pct_short_db": float(
            compare[
                (compare.scenario == "pcm_short_100us")
                & np.isclose(compare.sampling_ratio, 0.2)
                & (compare.method == "utgv_supervised")
            ].gain_over_zero_fill_db.iloc[0]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.root))


if __name__ == "__main__":
    main()
