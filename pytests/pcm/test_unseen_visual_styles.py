"""Ensure new PCM figures respect existing LION colormap presets."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image
import pytest
import yaml

from LION.pcm.randomized.figure_style import image_style, save_pcm_image
from LION.pcm.randomized.archive_unseen_reconstructions import _montage, METHODS
from LION.pcm.randomized.monitoring import image_grid
from LION.pcm.randomized.unseen_analysis import METHOD_LABELS


def test_existing_render_config_is_single_authority():
    config = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "pcm"
        / "pcm_tv_supervised.yaml"
    )
    final = yaml.safe_load(config.read_text())["log"]["final_result"]
    assert final["reconstruction"]["cmap"] == "afmhot"
    assert final["absolute_error"]["cmap"] == "viridis"
    assert final["sapm"]["lambda_map_render_configs"][0]["linear"][2] == "magma"
    for kind in ("reconstruction", "absolute_error"):
        cmap, low, high = image_style(kind)
        assert len(cmap.colors) == 256
        assert low == final[kind]["vmin"]
        assert high == final[kind]["vmax"]


def test_palette_renders_are_not_grayscale_and_differ(tmp_path):
    values = np.tile(np.linspace(0, 0.5, 64), (64, 1))
    original = values.copy()
    recon = tmp_path / "recon.png"
    error = tmp_path / "error.png"
    save_pcm_image(recon, values, kind="reconstruction")
    save_pcm_image(error, values, kind="absolute_error")
    with Image.open(recon) as im:
        rgb = np.array(im.convert("RGB"))
    with Image.open(error) as im:
        viridis = np.array(im.convert("RGB"))
    assert rgb.shape == viridis.shape == (64, 64, 3)
    assert np.any(rgb[:, :, 0] != rgb[:, :, 1])
    assert np.any(viridis[:, :, 0] != viridis[:, :, 1])
    assert not np.array_equal(rgb, viridis)
    np.testing.assert_array_equal(values, original)


def test_montages_labels_and_clean_rendering(tmp_path):
    target = np.full((1, 1, 24, 24), 0.45, dtype=np.float32)
    predictions = {"reference": target}
    rows = []
    for i, name in enumerate(METHODS):
        predictions[name] = target + i * 0.02
        rows.append({"method": name, "psnr": 21.2 + i, "ssim": 0.61})
    _montage(tmp_path, "demo_20pct", predictions, rows)
    _montage(tmp_path, "demo_20pct", predictions, rows, error=True)
    for name in ("demo_20pct_recon.png", "demo_20pct_abs_error.png"):
        with Image.open(tmp_path / name) as im:
            assert im.width > im.height
            assert im.width > 1200
    assert len(METHODS) == len(METHOD_LABELS) == 9


def test_training_preview_uses_shared_afm_style(tmp_path):
    x = np.ones((1, 1, 32, 32), dtype=np.float32) * 0.33
    image_grid(
        tmp_path / "preview.png", [("natural", [("clean", x), ("prediction", x * 0.9)])]
    )
    with Image.open(tmp_path / "preview.png") as im:
        assert im.width > 600


def test_invalid_plot_kinds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        image_style("parameter_map")
    with pytest.raises(ValueError, match="finite"):
        save_pcm_image(
            tmp_path / "bad.png", np.full((3, 3), np.nan), kind="reconstruction"
        )
