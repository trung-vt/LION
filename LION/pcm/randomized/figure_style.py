"""PCM image display styles loaded from the existing LION render configuration.

Reuses LION.visualize.plotting_functions rather than implementing another
image colormap and uint8 conversion. Display changes never touch model arrays,
measurements, scores, or the existing magma lambda-map rendering.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.colors
import numpy as np
import yaml

from LION.visualize.plotting_functions import make_colormap, save_image_formats


@lru_cache(maxsize=1)
def _styles() -> dict[str, dict]:
    """Same config used by supervised PCM reconstruction visualisation."""
    root = Path(__file__).resolve().parents[3]
    path = root / "config" / "pcm" / "pcm_tv_supervised.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    styles = raw["log"]["final_result"]
    for kind in ("reconstruction", "absolute_error"):
        if kind not in styles:
            raise ValueError(f"Missing PCM {kind} render spec in {path}")
    return styles


def image_style(kind: str) -> tuple[matplotlib.colors.Colormap, float, float]:
    """Get the shared PCM reconstruction/error palette and display limits."""
    if kind not in ("reconstruction", "absolute_error"):
        raise ValueError(f"Unsupported display kind {kind!r}")
    spec = _styles()[kind]
    return (
        make_colormap(spec["cmap"], float(spec.get("cmap_max", 1.0))),
        float(spec["vmin"]),
        float(spec["vmax"]),
    )


def save_pcm_image(path: Path, pixels: np.ndarray, *, kind: str) -> None:
    """Save image through the existing LION scalar-image renderer."""
    cmap, vmin, vmax = image_style(kind)
    values = np.asarray(pixels).squeeze()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 2-D image, got {values.shape}")
    save_image_formats(
        path,
        values,
        ("png",),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
