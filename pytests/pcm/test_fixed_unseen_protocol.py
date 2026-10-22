"""Fixed PCM acquisition and persistence regression tests."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from LION.pcm.randomized.fixed_protocol import fixed_acquisition, fixed_noise_config
from LION.pcm.randomized.persist_fixed_cases import _save_npz_exact


def test_default_noise_preset_at_j8():
    p = fixed_noise_config("pcm_default_10ms", 8).noise_params
    assert p.beam_sigma == pytest.approx(0.00012)
    assert p.dark_sigma == pytest.approx(3.61e-7 / 3.5e-8 / 256)
    assert p.bias_sigma == pytest.approx(2.9e-5)
    assert p.bias_gradient_sigma == pytest.approx(1.6e-7)


def test_gaussian_noise_has_only_dark_noise():
    p = fixed_noise_config("gaussian_dark", 8).noise_params
    assert p.beam_sigma == p.bias_sigma == p.bias_gradient_sigma == 0
    assert p.dark_sigma > 0
    assert fixed_noise_config("noiseless", 8) is None


def test_npz_immutable_round_trip(tmp_path):
    dest = tmp_path / "fixed.npz"
    x = np.zeros((1, 1, 32, 32), dtype=np.float32)
    y = np.array([[3, 4, 5]], dtype=np.float32)
    inds = np.array([0, 2, 4], dtype=np.int64)
    _save_npz_exact(dest, image=x, measurements=y, indices=inds)
    size = dest.stat().st_size
    _save_npz_exact(dest, image=x, measurements=y, indices=inds)
    assert dest.stat().st_size == size
    with pytest.raises(RuntimeError, match="immutable"):
        _save_npz_exact(dest, image=x, measurements=y + 1, indices=inds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires available CUDA GPU")
def test_fixed_masks_and_noise_are_reproducible():
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    sample = torch.full((1, 1, 64, 64), 0.37, device=device)
    a, ma = fixed_acquisition(
        sample,
        split="validation",
        case_index=0,
        ratio=0.2,
        scenario="pcm_default_10ms",
    )
    b, mb = fixed_acquisition(
        sample,
        split="validation",
        case_index=0,
        ratio=0.2,
        scenario="pcm_default_10ms",
    )
    c, mc = fixed_acquisition(
        sample,
        split="validation",
        case_index=0,
        ratio=0.2,
        scenario="noiseless",
    )
    assert ma["mask_sha256"] == mb["mask_sha256"] == mc["mask_sha256"]
    np.testing.assert_array_equal(a.sampled_indices, b.sampled_indices)
    np.testing.assert_array_equal(a.sampled_indices, c.sampled_indices)
    torch.testing.assert_close(a.measurement, b.measurement, rtol=0, atol=0)
    torch.testing.assert_close(c.measurement, c.op(sample), atol=1e-5, rtol=1e-5)
