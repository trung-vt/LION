"""Step 3 exact-variance algebra and immutable oracle-weight storage tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

from LION.pcm_tools.noise_generate import generate_beam_noise
from scripts.experiments.pcm.oracle_wls.oracle_variance import (
    beam_pair_variances,
    oracle_marginal_variance,
)


def test_dark_only_pair_difference():
    out = oracle_marginal_variance(
        np.array([3.0, 2.0, 7.0]),
        np.array([2.0, 4.0, 1.0]),
        i0=100.0,
        beam_ar=(),
        beam_ma=(),
        beam_sigma=0.0,
        bias_sigma=0.0,
        bias_gradient_sigma=0.0,
        dark_sigma=0.02,
    )
    np.testing.assert_array_equal(out.variance, np.full(3, 2 * 0.02**2))
    np.testing.assert_array_equal(out.dark, out.variance)
    np.testing.assert_array_equal(out.beam_integrated_arma, np.zeros(3))
    np.testing.assert_array_equal(out.weight, 1 / out.variance)
    np.testing.assert_array_equal(out.inverse_std, 1 / np.sqrt(out.variance))


def test_bias_only_white_and_shared_gradient():
    I0, bias, gradient = 365.0, 2.9e-5, 1.6e-7
    out = oracle_marginal_variance(
        np.array([300.0, 100.0]),
        np.array([10.0, 20.0]),
        i0=I0,
        beam_ar=(),
        beam_ma=(),
        beam_sigma=0,
        bias_sigma=bias,
        bias_gradient_sigma=gradient,
        dark_sigma=0,
    )
    expected = 2 * (I0 * bias) ** 2 + (I0 * gradient) ** 2
    np.testing.assert_allclose(out.variance, np.full(2, expected), rtol=2e-16)
    np.testing.assert_array_equal(out.dark, np.zeros(2))


def test_independent_averages_reduce_variance_by_repeat_count():
    params = dict(
        beam_ar=(-0.2,),
        beam_ma=(0.1,),
        beam_sigma=0.004,
        bias_sigma=0.001,
        bias_gradient_sigma=0.0001,
        dark_sigma=0.2,
    )
    p = np.array([0.8, 3.5, 17.1])
    q = np.array([0.4, 1.5, 17.1])
    one = oracle_marginal_variance(p, q, 123.0, repeat_acquisitions=1, **params)
    four = oracle_marginal_variance(p, q, 123.0, repeat_acquisitions=4, **params)
    np.testing.assert_allclose(one.variance / 4, four.variance, rtol=1e-15)
    np.testing.assert_allclose(four.weight, 4 * one.weight, rtol=1e-15)


def _brute_pair_covariance(num_pairs, ar, ma, sigma):
    """Compute covariance by explicitly multiplying the finite impulse matrix."""
    n = 2 * num_pairs
    pulse = np.zeros(n)
    pulse[0] = 1
    h = lfilter(np.r_[1.0, np.asarray(ma)], np.r_[1.0, -np.asarray(ar)], pulse)
    cumulative_h = np.cumsum(h)
    lower = np.zeros((n, n))
    for t in range(n):
        lower[t, : t + 1] = cumulative_h[t::-1]
    covariance = sigma**2 * lower @ lower.T
    return covariance


@pytest.mark.parametrize(
    "ar, ma",
    [
        ((), ()),
        ((-0.5,), (0.25,)),
        ((-0.9221, -0.1709, -0.0769, -0.0354), (0.0377, -0.3669)),
    ],
)
def test_integrated_arma_pair_variance_matches_direct_finite_covariance(ar, ma):
    sigma = 0.002
    n_pairs = 5
    covariance = _brute_pair_covariance(n_pairs, ar, ma, sigma)
    plus, minus = beam_pair_variances(n_pairs, beam_ar=ar, beam_ma=ma, beam_sigma=sigma)
    for k in range(n_pairs):
        a, b = 2 * k, 2 * k + 1
        expected_plus = covariance[a, a] + covariance[b, b] + 2 * covariance[a, b]
        expected_minus = covariance[a, a] + covariance[b, b] - 2 * covariance[a, b]
        np.testing.assert_allclose(plus[k], expected_plus, rtol=5e-12, atol=1e-20)
        np.testing.assert_allclose(minus[k], expected_minus, rtol=5e-12, atol=1e-20)


def test_swap_marginal_beam_variance_matches_two_exact_conditional_covariances():
    sigma = 0.008
    p = np.array([2.0, 10.0, 3.0, 5.0])
    q = np.array([1.0, 0.7, 2.0, 1.0])
    ar, ma = (-0.3, 0.04), (0.05,)
    covariance = _brute_pair_covariance(len(p), ar, ma, sigma)
    result = oracle_marginal_variance(
        p,
        q,
        i0=1.0,
        beam_ar=ar,
        beam_ma=ma,
        beam_sigma=sigma,
        bias_sigma=0,
        bias_gradient_sigma=0,
        dark_sigma=0,
    )
    for k in range(len(p)):
        a, b = 2 * k, 2 * k + 1
        cov = covariance[np.ix_([a, b], [a, b])]
        unswapped = np.array([p[k], -q[k]])
        swapped = np.array([-q[k], p[k]])
        expected = 0.5 * (unswapped @ cov @ unswapped + swapped @ cov @ swapped)
        np.testing.assert_allclose(result.beam_integrated_arma[k], expected, rtol=1e-12)


def test_beam_impulse_agrees_with_actual_statsmodels_generator():
    ar, ma, sigma = (-0.9221, -0.1709, -0.0769, -0.0354), (0.0377, -0.3669), 1.2e-4
    n = 22
    generated = generate_beam_noise(
        num_measurements=n,
        beam_ar=ar,
        beam_ma=ma,
        beam_sigma=sigma,
        rng=np.random.default_rng(43),
    )
    rng = np.random.default_rng(43)
    innovations = rng.standard_normal(n) * sigma
    expected = np.cumsum(
        lfilter(
            np.r_[1.0, np.asarray(ma)],
            np.r_[1.0, -np.asarray(ar)],
            innovations,
        )
    )
    np.testing.assert_allclose(generated, expected, rtol=5e-14, atol=1e-20)


def test_integrated_beam_with_equal_split_currents_depends_on_arma_increment():
    # If p=q, the accumulated common-mode beam trace cancels completely.
    # Only the adjacent ARMA increment z_{2k+1} remains.
    p = np.array([4.0, 4.0, 4.0, 4.0])
    q = p.copy()
    sigma = 0.01
    out = oracle_marginal_variance(
        p,
        q,
        i0=0,
        beam_ar=(),
        beam_ma=(),
        beam_sigma=sigma,
        bias_sigma=0,
        bias_gradient_sigma=0,
        dark_sigma=0,
    )
    np.testing.assert_allclose(out.variance, 16 * sigma**2 * np.ones(4))


def test_heteroscedasticity_and_no_realized_measurement_in_api():
    out = oracle_marginal_variance(
        np.array([3.0, 7.0, 11.0]),
        np.array([2.0, 4.0, 1.0]),
        i0=22.0,
        beam_ar=(-0.3,),
        beam_ma=(0.2,),
        beam_sigma=0.02,
        bias_sigma=0.0003,
        bias_gradient_sigma=1e-5,
        dark_sigma=0.1,
    )
    assert out.variance.dtype == out.weight.dtype == np.float64
    assert np.unique(out.variance).size == 3
    assert np.isfinite(out.weight).all()
    assert np.all(out.variance > 0)
    np.testing.assert_allclose(
        out.variance,
        out.dark + out.bias_white + out.bias_drift + out.beam_integrated_arma,
        rtol=2e-15,
    )


@pytest.mark.parametrize(
    "patch",
    [
        dict(dark_sigma=0.0, bias_sigma=0.0, bias_gradient_sigma=0.0, beam_sigma=0.0),
        dict(dark_sigma=-0.1),
        dict(beam_sigma=-0.1),
    ],
)
def test_invalid_or_degenerate_noise_does_not_generate_false_weights(patch):
    defaults = dict(
        dark_sigma=0.1,
        bias_sigma=0.01,
        bias_gradient_sigma=1e-5,
        beam_ar=(-0.1,),
        beam_ma=(0.3,),
        beam_sigma=0.01,
    )
    defaults.update(patch)
    with pytest.raises(ValueError):
        oracle_marginal_variance(
            np.array([1.0]),
            np.array([1.0]),
            i0=100.0,
            **defaults,
        )


def test_nonfinite_current_rejected():
    with pytest.raises(ValueError):
        oracle_marginal_variance(
            np.array([np.nan]),
            np.array([1.0]),
            i0=100.0,
            beam_ar=(),
            beam_ma=(),
            beam_sigma=0,
            bias_sigma=0,
            bias_gradient_sigma=0,
            dark_sigma=0.5,
        )


def test_empty_vector_rejected():
    with pytest.raises(ValueError):
        oracle_marginal_variance(
            np.array([]),
            np.array([]),
            i0=1.0,
            beam_ar=(),
            beam_ma=(),
            beam_sigma=0,
            bias_sigma=0,
            bias_gradient_sigma=0,
            dark_sigma=0.5,
        )
