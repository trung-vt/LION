"""Tests for loading padded NPL PCM text maps in physical current units."""

from copy import deepcopy

import numpy as np
import pytest

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.data import (
    _select_image_txt_map_active_region,
    load_ground_truth_current_map,
    prepare_data,
)


def test_padded_txt_map_preserves_sign_and_orientation(tmp_path):
    """Crop the raw current map before sign correction, without flips."""
    preset = deepcopy(PRESETS["unseen_npl_cigs_active_256"])
    preset.data.j_order = 2
    preset.data.root = tmp_path
    preset.this_sample.data_filename = "example.txt.map"
    preset.this_sample.ground_truth_filename = "example.txt.map"

    raw = np.zeros((9, 11), dtype=np.float64)
    active = np.array(
        [
            [-1.0e-9, +0.1e-9, -0.3e-9, -0.2e-9],
            [-0.4e-9, -0.5e-9, -0.6e-9, -0.7e-9],
            [-0.8e-9, -0.9e-9, -1.0e-9, -1.1e-9],
            [-1.2e-9, -1.3e-9, -1.4e-9, -1.5e-9],
        ]
    )
    raw[3:7, 5:9] = active
    np.savetxt(tmp_path / "example.txt.map", raw, fmt="%.8E", delimiter="\t")

    selected = _select_image_txt_map_active_region(raw, preset)
    np.testing.assert_array_equal(selected, active)
    loaded = load_ground_truth_current_map(preset)
    np.testing.assert_array_equal(loaded, -active)

    target, measurement = prepare_data(preset, device="cpu")
    assert measurement is None
    assert tuple(target.shape) == (4, 4)
    np.testing.assert_allclose(target.numpy(), -active, rtol=1e-7, atol=2e-16)
    normalized = preset.scaling.scale(target)
    np.testing.assert_allclose(
        preset.scaling.rescale(normalized).numpy(),
        target.numpy(),
        rtol=1e-7,
        atol=2e-16,
    )
    assert float(normalized.max()) > 0 and float(normalized.min()) < 0


def test_full_square_txt_map_remains_unchanged():
    """The existing square E27-style input is not cropped."""
    preset = deepcopy(PRESETS["unseen_npl_cigs_active_256"])
    preset.data.j_order = 2
    original = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_array_equal(
        _select_image_txt_map_active_region(original, preset), original
    )


def test_mismatched_active_shape_fails():
    """A padded text map cannot silently select the wrong active region."""
    preset = deepcopy(PRESETS["unseen_npl_cigs_active_256"])
    preset.data.j_order = 2
    wrong = np.zeros((10, 10), dtype=float)
    wrong[2:5, 3:7] = -1e-9
    with pytest.raises(ValueError, match="does not match"):
        _select_image_txt_map_active_region(wrong, preset)


def test_all_zero_txt_map_fails():
    """An empty padded export must not be treated as a valid PCM sample."""
    preset = deepcopy(PRESETS["unseen_npl_cigs_active_256"])
    preset.data.j_order = 2
    with pytest.raises(ValueError, match="no nonzero"):
        _select_image_txt_map_active_region(np.zeros((9, 11)), preset)
