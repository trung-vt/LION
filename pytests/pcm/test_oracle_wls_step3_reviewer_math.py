"""Independent Step 3 mathematical tests supplied in the external review.

The stress tests use the REAL PCM simulator, not the cached-clean
Monte Carlo path from Step 4. All random seeds and test thresholds are fixed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import _sample_split_measurements, simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import sample_indices
from scripts.experiments.pcm.oracle_wls.oracle_variance import (
    beam_pair_variances,
    oracle_marginal_variance,
)

NPL = PCMNoiseSimulation.npl_default(current_to_model_scale=1 / 3.5e-8, j_order=8)
P = NPL.noise_params
SMALL_MASK = np.arange(3, 120)
LATE_COARSE_MASK = np.arange(119, -1, -1)


def _image(n):
    """Smooth bright map with a dark block."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = 0.55 + 0.3 * np.exp(-((xx - 0.35) ** 2 + (yy - 0.6) ** 2) / 0.05) - 0.2 * yy
    img[n // 6 : n // 3, n // 2 : n - n // 6] *= 0.3
    return torch.from_numpy(np.clip(img, 0, 1)).reshape(1, 1, n, n)


def _contrast(n):
    """High-contrast blocks to generate large non-DC differences."""
    img = np.zeros((n, n))
    img[:, : n // 2] = 1.0
    img[: n // 2, : n // 4] = 0.2
    img[n // 4 : n // 2, n // 2 :] = 0.7
    return torch.from_numpy(img).reshape(1, 1, n, n)


def _simulated_vs_oracle(x, mask, noise, j_order, draws, seed=2026):
    """Empirical E[(y-Ax)^2] versus analytic oracle, using the full simulator."""
    op = PhotocurrentMapOp(
        j_order=j_order, s_indices=mask, normalized=True, device="cpu"
    )
    plus, minus = _sample_split_measurements(x, op)
    i0 = float(x.sum() * op.normalization_factor / noise.signal_to_bias_ratio)
    n = noise.noise_params
    oracle = oracle_marginal_variance(
        plus.reshape(-1).numpy(),
        minus.reshape(-1).numpy(),
        i0,
        beam_ar=n.beam_ar,
        beam_ma=n.beam_ma,
        beam_sigma=n.beam_sigma,
        bias_sigma=n.bias_sigma,
        bias_gradient_sigma=n.bias_gradient_sigma,
        dark_sigma=n.dark_sigma,
        repeat_acquisitions=noise.n_repeat_acquisitions,
    )
    rng = np.random.default_rng(seed)
    second_moment = torch.zeros(len(mask), dtype=torch.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        acquired = simulate_pcm_acquisition(
            x.expand(count, -1, -1, -1).contiguous(),
            j_order=j_order,
            sampled_indices=mask,
            noise=noise,
            rng=rng,
            device="cpu",
        )
        residual = acquired.measurement.reshape(count, -1) - (plus - minus).reshape(-1)
        second_moment += (residual**2).sum(0)
    return oracle.variance, (second_moment / draws).numpy()


def _small(**changes):
    return replace(NPL, noise_params=replace(P, **changes))


def test_oracle_matches_simulator_for_d_squared_beam_term():
    """Beam x50, coarse rows last: a strong integrated-beam stress test."""
    variance, observed = _simulated_vs_oracle(
        _contrast(16),
        LATE_COARSE_MASK,
        _small(dark_sigma=0.01, beam_sigma=50 * P.beam_sigma),
        4,
        6000,
    )
    assert variance.max() / variance.min() > 20
    ratio = observed / variance
    assert abs(ratio.mean() - 1) < 0.02
    assert np.all(np.abs(ratio - 1) < 5 * np.sqrt(2 / 6000))
    assert abs(np.polyfit(variance, observed, 1)[0] - 1) < 0.05


def test_oracle_matches_simulator_for_s_squared_beam_term():
    """Small ascending mask emphasizes the sum-current beam contribution."""
    variance, observed = _simulated_vs_oracle(
        _contrast(16),
        SMALL_MASK,
        _small(dark_sigma=0.01, beam_sigma=50 * P.beam_sigma),
        4,
        6000,
    )
    ratio = observed / variance
    assert abs(ratio.mean() - 1) < 0.02
    assert np.all(np.abs(ratio - 1) < 5 * np.sqrt(2 / 6000))


def test_oracle_matches_simulator_for_shared_bias_gradient():
    """Gradient x5000 must appear once as (I0*sigma_g)^2."""
    variance, observed = _simulated_vs_oracle(
        _image(16),
        SMALL_MASK,
        _small(
            dark_sigma=0.01,
            beam_sigma=0.0,
            bias_gradient_sigma=5000 * P.bias_gradient_sigma,
        ),
        4,
        4000,
    )
    assert abs((observed / variance).mean() - 1) < 0.07


def test_oracle_matches_simulator_for_three_repeats():
    noise = replace(
        _small(dark_sigma=0.01, beam_sigma=50 * P.beam_sigma),
        n_repeat_acquisitions=3,
    )
    variance, observed = _simulated_vs_oracle(_image(16), SMALL_MASK, noise, 4, 3000)
    assert abs((observed / variance).mean() - 1) < 0.03


def test_oracle_matches_full_size_default_npl_simulator():
    """256x256, actual 20% multilevel sampling, 1000 independent draws."""
    mask = sample_indices(
        j_order=8,
        sampling_ratio=0.2,
        coarse_j=6,
        randomising_scheme="multilevel",
        rng=np.random.default_rng(10655734495037282845),
    )
    variance, observed = _simulated_vs_oracle(_image(256), mask, NPL, 8, 1000)
    ratio = observed / variance
    assert abs(ratio.mean() - 1) < 0.01
    deciles = [ratio[idx].mean() for idx in np.array_split(np.argsort(variance), 10)]
    assert np.all(np.abs(np.asarray(deciles) - 1) < 0.02)


def test_first_pair_beam_variances_match_hand_calculation():
    h1 = P.beam_ar[0] + P.beam_ma[0]
    plus, minus = beam_pair_variances(
        1, beam_ar=P.beam_ar, beam_ma=P.beam_ma, beam_sigma=P.beam_sigma
    )
    np.testing.assert_allclose(
        plus[0], P.beam_sigma**2 * (1 + (2 + h1) ** 2), rtol=1e-14
    )
    np.testing.assert_allclose(minus[0], P.beam_sigma**2 * (1 + h1**2), rtol=1e-14)


def test_full_length_beam_sums_match_independent_recursion():
    """All 13,107 pairs: explicit ARMA recursion plus covariance sums."""
    m, n, sigma = 13107, 2 * 13107, P.beam_sigma
    h = np.zeros(n)
    for t in range(n):
        acc = (1.0 if t == 0 else 0.0) + (
            P.beam_ma[t - 1] if 1 <= t <= len(P.beam_ma) else 0.0
        )
        acc += sum(a * h[t - i] for i, a in enumerate(P.beam_ar, start=1) if t - i >= 0)
        h[t] = acc
    H = np.cumsum(h)
    var_r = sigma**2 * np.cumsum(H**2)
    cov_next = sigma**2 * np.cumsum(H[:-1] * H[1:])
    a, b = np.arange(0, n, 2), np.arange(1, n, 2)
    plus, minus = beam_pair_variances(
        m, beam_ar=P.beam_ar, beam_ma=P.beam_ma, beam_sigma=sigma
    )
    np.testing.assert_allclose(plus, var_r[a] + var_r[b] + 2 * cov_next[a], rtol=1e-12)
    np.testing.assert_allclose(minus, var_r[a] + var_r[b] - 2 * cov_next[a], rtol=1e-9)


@pytest.mark.parametrize(
    "k, plus, minus",
    [
        (0, 3.2321712384000002e-08, 2.5663152384000006e-08),
        (1, 4.1318059692236494e-08, 2.7249767958380187e-08),
        (4095, 4.3677654434307407e-05, 2.7889845818837871e-08),
        (13106, 0.00013972297749928237, 2.7889845818837871e-08),
    ],
)
def test_npl_beam_pair_variances_are_pinned(k, plus, minus):
    """Version-specific numeric anchors supplied by the external reviewer."""
    got_plus, got_minus = beam_pair_variances(
        13107, beam_ar=P.beam_ar, beam_ma=P.beam_ma, beam_sigma=P.beam_sigma
    )
    np.testing.assert_allclose(got_plus[k], plus, rtol=1e-12)
    np.testing.assert_allclose(got_minus[k], minus, rtol=1e-12)
