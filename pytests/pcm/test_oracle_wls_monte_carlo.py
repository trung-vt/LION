"""Focused Step 4 tests: actual simulator equivalence, streaming accuracy and guards."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import (
    _sample_split_measurements,
    simulate_pcm_acquisition,
)
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import (
    ACCEPTANCE,
    independent_mc_seed,
    noise_parameters_for_mode,
    numerical_diagnostics,
    one_realization,
    streaming_moments,
)
from scripts.experiments.pcm.oracle_wls.oracle_variance import (
    oracle_marginal_variance,
)


def _tiny():
    indices = np.arange(51)
    x = torch.rand(
        (1, 1, 16, 16),
        generator=torch.Generator().manual_seed(17),
        dtype=torch.float64,
    )
    op = PhotocurrentMapOp(j_order=4, s_indices=indices, normalized=True, device="cpu")
    plus, minus = _sample_split_measurements(x, op)
    i0 = float(x.sum() / 16 / 0.3)
    return x, indices, plus.reshape(-1), minus.reshape(-1), i0


def test_fast_mc_noise_path_is_byte_exact_to_full_pcm_simulator():
    """Fixed x,A,RNG produces exactly the same noisy difference as the full code."""
    x, indices, plus, minus, i0 = _tiny()
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8,
        j_order=4,
    )
    for seed in (11, 239, 442):
        slow = simulate_pcm_acquisition(
            x,
            j_order=4,
            sampled_indices=indices,
            noise=noise,
            rng=np.random.default_rng(seed),
            device="cpu",
        )
        direct_error = one_realization(
            plus, minus, i0, noise.noise_params, np.random.default_rng(seed)
        )
        clean = (plus - minus).cpu().numpy()
        np.testing.assert_array_equal(
            slow.measurement.reshape(-1).cpu().numpy() - clean,
            direct_error,
        )


@pytest.mark.parametrize("mode", ["dark", "bias_white", "bias_drift", "beam"])
def test_component_toggle_keeps_only_the_named_std(mode):
    defaults = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8, j_order=4
    ).noise_params
    altered = noise_parameters_for_mode(defaults, mode)
    mapping = {
        "dark": "dark_sigma",
        "bias_white": "bias_sigma",
        "bias_drift": "bias_gradient_sigma",
        "beam": "beam_sigma",
    }
    for other_mode, field in mapping.items():
        assert getattr(altered, field) == (
            getattr(defaults, field) if other_mode == mode else 0
        )


def test_independent_seed_streams_across_images_and_modes():
    assert independent_mc_seed("CIGS", 733, "full") == independent_mc_seed(
        "CIGS", 733, "full"
    )
    assert (
        len(
            {
                independent_mc_seed(category, 733, mode)
                for category in ("CIGS", "Organic", "Silicone")
                for mode in ("full", "beam", "dark")
            }
        )
        == 9
    )


def test_welford_matches_numpy_exact_variance_for_nonzero_small_mean():
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(345, 127)) * 0.03 + 1e-5
    mean, variance = streaming_moments((x for x in samples), n=len(samples))
    np.testing.assert_allclose(mean, samples.mean(axis=0), rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        variance, samples.var(axis=0, ddof=1), rtol=2e-14, atol=1e-19
    )


def test_welford_rejects_nonfinite_and_wrong_count():
    with pytest.raises(ValueError):
        streaming_moments((np.array([1.0]),), n=1)
    with pytest.raises(ValueError, match="nonfinite"):
        streaming_moments((np.array([float("nan")]), np.array([2.0])), n=2)
    with pytest.raises(ValueError, match="wrong"):
        streaming_moments((np.array([1.0]), np.array([2.0])), n=3)


def test_mc_dark_only_matches_oracle_variance_statistically_on_tiny_image():
    x, indices, plus, minus, i0 = _tiny()
    base = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8, j_order=4
    ).noise_params
    params = noise_parameters_for_mode(base, "dark")
    terms = oracle_marginal_variance(
        plus.numpy(),
        minus.numpy(),
        i0,
        beam_ar=base.beam_ar,
        beam_ma=base.beam_ma,
        beam_sigma=base.beam_sigma,
        bias_sigma=base.bias_sigma,
        bias_gradient_sigma=base.bias_gradient_sigma,
        dark_sigma=base.dark_sigma,
    )
    draws = np.random.default_rng(123)
    mu, empirical = streaming_moments(
        (one_realization(plus, minus, i0, params, draws) for _ in range(1024)),
        n=1024,
    )
    target = terms.dark
    assert target.shape == (51,)
    d = numerical_diagnostics(mu, empirical, target, sample_count=1024, mode="dark")
    assert 0.93 < d["variance_ratio_mean"] < 1.07
    assert d["predeclared_checks_passed"]


def test_predeclared_checks_do_not_accept_biased_variance():
    expected = np.ones(100, dtype=np.float64)
    empirical = np.full(100, 1.5)
    d = numerical_diagnostics(
        np.zeros(100), empirical, expected, sample_count=1024, mode="full"
    )
    assert not d["predeclared_checks_passed"]
    assert d["median_abs_variance_relative_error"] == pytest.approx(0.5)


def test_constant_analytical_variance_passes_ideal_sample_statistics():
    predicted = np.array([0.02, 0.09, 0.1], dtype=np.float64)
    result = numerical_diagnostics(
        np.zeros(3),
        predicted.copy(),
        predicted,
        sample_count=1024,
        mode="full",
    )
    assert result["predeclared_checks_passed"]
    assert result["p95_abs_variance_relative_error"] == 0.0


def test_production_gate_is_at_least_1024_repetitions():
    assert ACCEPTANCE["min_repeats_full"] >= 1024
    assert ACCEPTANCE["min_repeats_control"] >= 512
