"""Integrity checks for the randomized-noise pilot, independent of image datasets."""

import numpy as np
import pytest
import torch
from statsmodels.tsa.arima_process import ArmaProcess

from LION.pcm.randomized.noise import (
    AR,
    MA,
    NoiseRange,
    decimal_interval,
    image_corruption,
    randomized_acquisition,
    sample_parameters,
    sample_pattern_indices,
    split_ssdu,
    valid_coarse_levels,
)


def test_decimal_bounds_negative_coefficients():
    assert decimal_interval(-0.9221) == (-0.93, -0.92)
    assert decimal_interval(-0.1709) == (-0.18, -0.17)
    assert decimal_interval(0.0377) == (0.03, 0.04)
    assert decimal_interval(-0.3669) == (-0.37, -0.36)


def test_randomized_parameters_stable_and_reproducible():
    rng = np.random.default_rng(11)
    p = sample_parameters(rng, NoiseRange())
    q = sample_parameters(np.random.default_rng(11), NoiseRange())
    assert p == q
    for _ in range(100):
        p = sample_parameters(rng, NoiseRange())
        assert 0 <= p.bias_gradient_sigma <= 3e-7
        assert 0 <= p.bias_sigma <= 4e-5
        assert 0 <= p.beam_sigma <= 1e-3
        assert 0 <= p.dark_sigma <= 1
        for value, original in zip(p.beam_ar, AR):
            assert (
                decimal_interval(original)[0] <= value <= decimal_interval(original)[1]
            )
        process = ArmaProcess(np.r_[1, -np.asarray(p.beam_ar)], np.r_[1, p.beam_ma])
        assert process.isstationary and process.isinvertible


@pytest.mark.parametrize("mode", ["gaussian", "pcm_like", "mixture"])
def test_image_zero_noise_and_no_physics(mode, monkeypatch):
    import LION.pcm.randomized.noise as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Image corruption must not invoke PCM acquisition")

    monkeypatch.setattr(module, "simulate_pcm_acquisition", forbidden)
    clean = torch.rand(1, 1, 16, 16)
    bounds = NoiseRange(0, 0, 0, 0)
    noisy, sigma, _ = image_corruption(
        clean, np.random.default_rng(2), bounds, mode=mode
    )
    torch.testing.assert_close(noisy, clean, atol=1e-7, rtol=0)
    assert sigma == 0


def test_no_clipping_heavy_noise():
    x = torch.full((1, 1, 16, 16), 0.5)
    p = sample_parameters(np.random.default_rng(2), NoiseRange())
    p.dark_sigma = 1.0
    z, _, _ = image_corruption(
        x, np.random.default_rng(5), NoiseRange(), mode="gaussian", parameters=p
    )
    assert z.min() < 0 and z.max() > 1


@pytest.mark.parametrize("side", [16, 64, 256])
@pytest.mark.parametrize("ratio", [0.01, 0.02, 0.2, 0.999, 1.0])
def test_sampling_all_valid_coarse_levels(side, ratio):
    count = max(2, int(np.ceil(ratio * side * side)))
    for coarse in valid_coarse_levels(side.bit_length() - 1, count):
        indices, info = sample_pattern_indices(
            side, np.random.default_rng(7), ratio=ratio, coarse_j=coarse
        )
        assert len(indices) == count
        assert len(set(indices)) == count
        assert indices[0] == 0
        assert indices[-1] < side * side
        assert info["realized_ratio"] >= ratio


def test_reject_impossible_sampling():
    with pytest.raises(ValueError):
        sample_pattern_indices(32, np.random.default_rng(0), ratio=0.01, coarse_j=5)
    with pytest.raises(ValueError):
        sample_pattern_indices(30, np.random.default_rng(0))


def test_full_pcm_zero_noise_identity():
    # Large bias-offset cancellation in float32 is not an algebraic identity test.
    x = torch.rand(
        1, 1, 16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(12)
    )
    a, info = randomized_acquisition(
        x, np.random.default_rng(1), NoiseRange(0, 0, 0, 0), ratio=1.0
    )
    torch.testing.assert_close(a.measurement, a.op(x), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(a.op.adjoint(a.measurement), x, atol=1e-10, rtol=1e-10)


def test_ssdu_partition_disjoint_and_uses_original_measurements():
    x = torch.rand(1, 1, 16, 16)
    a, _ = randomized_acquisition(x, np.random.default_rng(9), NoiseRange(), ratio=0.01)
    input_op, input_y, loss_op, loss_y, partition = split_ssdu(
        a, np.random.default_rng(7)
    )
    train_rows, loss_rows = partition["input_rows"], partition["loss_rows"]
    assert set(train_rows).isdisjoint(loss_rows)
    assert set(train_rows) | set(loss_rows) == set(a.sampled_indices)
    assert 0 in train_rows and 0 not in loss_rows
    lookup = {int(v): i for i, v in enumerate(a.sampled_indices)}
    torch.testing.assert_close(
        input_y, a.measurement[..., [lookup[v] for v in train_rows]]
    )
    torch.testing.assert_close(
        loss_y, a.measurement[..., [lookup[v] for v in loss_rows]]
    )
    assert input_op(x).shape == input_y.shape
    assert loss_op(x).shape == loss_y.shape
    assert input_y.numel() > 0 and loss_y.numel() > 0
