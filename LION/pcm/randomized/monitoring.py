"""Training and PnP diagnostics, without changing model optimisation."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from LION.pcm.randomized.figure_style import image_style


def append_event(path: Path, event: dict) -> None:
    """Publish one complete update for a dashboard polling during training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


def run_history(folder: Path) -> list[dict]:
    stream = folder / "history.jsonl"
    if stream.is_file():
        result = []
        for line in stream.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if result:
            return result
    file = folder / "history.json"
    try:
        return json.loads(file.read_text()) if file.is_file() else []
    except json.JSONDecodeError:  # Atomic replacement should prevent this.
        return []


def image_grid(
    target: Path, rows: list[tuple[str, list[tuple[str, np.ndarray]]]]
) -> None:
    """Use LION's shared AFM reconstruction render style, without manual RGB conversion."""
    if not rows:
        return
    max_cols = max(len(cells) for _, cells in rows)
    fig, axes = plt.subplots(
        len(rows),
        max_cols,
        figsize=(max(9.0, 2.8 * max_cols), 3.4 * len(rows)),
        squeeze=False,
    )
    cmap, vmin, vmax = image_style("reconstruction")
    for i, (domain, cells) in enumerate(rows):
        for j, (label, pixels) in enumerate(cells):
            axis = axes[i, j]
            values = np.asarray(pixels).squeeze()
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f"Expected finite 2D example, got {values.shape}")
            axis.imshow(
                values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            axis.set_title(label, fontsize=10, pad=8)
            axis.set_xlabel(domain, fontsize=9, labelpad=8)
            axis.set_xticks([])
            axis.set_yticks([])
        for j in range(len(cells), max_cols):
            axes[i, j].axis("off")
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.86,
        bottom=0.11,
        wspace=0.18,
        hspace=0.42,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=125, facecolor="white")
    plt.close(fig)


def training_preview(directory: Path, step: int, examples: dict) -> None:
    rows = []
    for domain in ("natural", "pcm_synthetic"):
        prefixes = sorted(
            key[:-6]
            for key in examples
            if key.startswith(domain) and key.endswith("_clean")
        )
        if prefixes:
            prefix = prefixes[0]
            rows.append(
                (
                    domain,
                    [
                        ("Ground truth", examples[prefix + "_clean"]),
                        ("Input", examples[prefix + "_input"]),
                        ("Prediction", examples[prefix + "_prediction"]),
                    ],
                )
            )
    if rows:
        image_grid(directory / "validation_steps" / f"step_{step:06d}.png", rows)


def _export(fig, directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return name + ".png"


def training_figures(root: Path) -> list[str]:
    """Works on 2026-10-14 logs and future append-only progress files."""
    out = root / "monitoring" / "figures"
    result = []
    for family, prefixes in (
        ("denoisers", ("dwtdnet_", "gs_dwtdnet_", "drunet_", "gs_drunet_")),
        ("regularizers", ("utv_", "utgv_")),
    ):
        for field in (
            "training_loss",
            "gradient_norm",
            "validation_mse",
            "selection_score",
            "seconds",
        ):
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            found = False
            for folder in sorted((root / "runs").glob("*")):
                if not folder.name.startswith(prefixes):
                    continue
                series = [
                    r
                    for r in run_history(folder)
                    if isinstance(r.get(field), (int, float)) and np.isfinite(r[field])
                ]
                if not series:
                    continue
                found = True
                ax.plot(
                    [v["step"] for v in series],
                    [v[field] for v in series],
                    label=folder.name,
                    lw=1.3,
                    marker="." if len(series) < 12 else None,
                )
            if found:
                ax.set(
                    xlabel="Optimizer update",
                    ylabel=field.replace("_", " "),
                    title=f"{family.title()}: {field.replace('_', ' ')}",
                )
                if field not in ("seconds",) and all(
                    line.get_ydata().min() > 0 for line in ax.lines
                ):
                    ax.set_yscale("log")
                ax.grid(alpha=0.25)
                ax.legend(ncol=2, fontsize=7)
                result.append(_export(fig, out, f"{family}_{field}"))
            else:
                plt.close(fig)
    return result


def convergence_figures(root: Path, folder: Path | None = None) -> list[str]:
    import pandas as pd

    folder = folder or root / "pnp_sweep" / "convergence"
    path = folder / "convergence_test.csv"
    if not path.is_file():
        return []
    data = pd.read_csv(path)
    data = data[np.isclose(data.ratio, 0.2) & np.isclose(data.dark_sigma, 0.2)]
    output = root / "monitoring" / "figures"
    result = []
    for field in (
        "psnr",
        "ssim",
        "data_residual_mse",
        "relative_update_norm",
        "objective",
    ):
        for xfield in ("iteration", "seconds"):
            if xfield == "seconds" and field in ("objective", "relative_update_norm"):
                continue
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            found = False
            for (model, domain, case), group in data.groupby(
                ["model", "domain", "case"]
            ):
                if case not in (0, 2):
                    continue
                series = group.dropna(subset=[field]).sort_values(xfield)
                if series.empty:
                    continue
                found = True
                ax.plot(
                    series[xfield], series[field], lw=1.3, label=f"{model} / {domain}"
                )
            if found:
                ax.set(
                    title=f"PnP convergence, 20% sampling, dark sigma 0.2: {field}",
                    xlabel=xfield,
                    ylabel=field,
                )
                if field in ("data_residual_mse", "relative_update_norm"):
                    ax.set_yscale("log")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, ncol=2)
                result.append(_export(fig, output, f"pnp_{field}_vs_{xfield}"))
            else:
                plt.close(fig)
    return result
