"""Exact float64 diagonal marginal oracle variance for the current PCM noise code.

This module is experiment-only and does not change the existing PCM simulator.
The variance is conditional on clean x and sampling A, but marginalized over
the independent random within-pair swaps and all Gaussian noise sources.

References: LION/pcm_tools/noise_generate.py (generate_noise, add_noise) and
LION/pcm/simulation.py (simulate_pcm_acquisition).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class OracleVarianceTerms:
    """Variance decomposition in normalized differential measurement units."""

    variance: np.ndarray
    weight: np.ndarray
    inverse_std: np.ndarray
    dark: np.ndarray
    bias_white: np.ndarray
    bias_drift: np.ndarray
    beam_integrated_arma: np.ndarray


def beam_pair_variances(num_pairs: int, *, beam_ar, beam_ma, beam_sigma: float):
    r"""Return Var(R_{2k}+R_{2k+1}) and Var(R_{2k}-R_{2k+1}).

    R_t is the cumsum of the actual zero-initial-state ARMA output. This is
    calculated from its impulse response rather than a stationary approximation.
    Each pair has trace indices (2k, 2k+1), with k zero-based.

    If h_j is the ARMA impulse response and H_j=sum_{i<=j} h_i, then:
      Var(R_{2k}-R_{2k+1}) = sigma^2 * sum_{j<=2k+1} h_j^2;
      Var(R_{2k}+R_{2k+1}) = sigma^2 *
                            sum_{j<=2k+1} (H_j+H_{j-1})^2.
    H_{-1}=0. Both sums are nonnegative and avoid large covariance subtraction.
    """
    if num_pairs < 1:
        raise ValueError("At least one measurement pair is required")
    if beam_sigma < 0:
        raise ValueError("Beam-noise standard deviation must be nonnegative")

    n = 2 * num_pairs
    impulse = np.zeros(n, dtype=np.float64)
    impulse[0] = 1.0
    ar = np.r_[1.0, -np.asarray(beam_ar, dtype=np.float64)]
    ma = np.r_[1.0, np.asarray(beam_ma, dtype=np.float64)]
    # scipy.signal.lfilter has the same zero-state ARMA convention as
    # statsmodels.tsa.arima_process.arma_generate_sample (burnin=0).
    h = lfilter(ma, ar, impulse).astype(np.float64)
    H = np.cumsum(h)
    H_previous = np.empty_like(H)
    H_previous[0] = 0
    H_previous[1:] = H[:-1]
    plus = beam_sigma**2 * np.cumsum((H + H_previous) ** 2)[1::2]
    minus = beam_sigma**2 * np.cumsum(h**2)[1::2]
    if not (np.isfinite(plus).all() and np.isfinite(minus).all()):
        raise ValueError("Non-finite integrated ARMA variance")
    return plus, minus


def oracle_marginal_variance(
    clean_plus: np.ndarray,
    clean_minus: np.ndarray,
    i0: float,
    *,
    beam_ar,
    beam_ma,
    beam_sigma: float,
    bias_sigma: float,
    bias_gradient_sigma: float,
    dark_sigma: float,
    repeat_acquisitions: int = 1,
) -> OracleVarianceTerms:
    r"""Compute exact marginal diagonal Var(y_k - (Ax)_k | x, A).

    The noisy differential pair is p + n^+ - (q + n^-), where the noise
    trace within a pair is independently swapped with probability 1/2.

    The constant I0 bias cancels from the differential measurement.
    Dark and independent bias fluctuations contribute 2*sigma_d^2 and
    2*I0^2*sigma_bias^2, and the shared bias gradient contributes
    I0^2*sigma_gradient^2.

    For integrated beam R_a, R_b, let D=p-q and S=p+q. Averaging over the
    within-pair swap yields:
      Var_beam = (D^2/4)*Var(R_a+R_b)
               + (S^2/4)*Var(R_a-R_b).
    This includes the adjacent-sample beam covariance exactly.

    Independent repeated acquisitions are averaged, so the final variances
    are divided by repeat_acquisitions. Only marginal variances are returned;
    the full measurement covariance may be non-diagonal.
    """
    p = np.asarray(clean_plus, dtype=np.float64)
    q = np.asarray(clean_minus, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or not p.size:
        raise ValueError(
            "Clean split measurements must be nonempty equal-length vectors"
        )
    if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(i0)):
        raise ValueError("Non-finite clean measurements or I0")
    if repeat_acquisitions < 1 or int(repeat_acquisitions) != repeat_acquisitions:
        raise ValueError("repeat_acquisitions must be a positive integer")
    if min(beam_sigma, bias_sigma, bias_gradient_sigma, dark_sigma) < 0:
        raise ValueError("Noise standard deviations must be nonnegative")
    if beam_ar is None or beam_ma is None:
        if beam_sigma != 0:
            raise ValueError("Nonzero beam sigma requires AR and MA coefficients")
        beam_ar, beam_ma = (), ()

    pair_plus, pair_minus = beam_pair_variances(
        p.size, beam_ar=beam_ar, beam_ma=beam_ma, beam_sigma=beam_sigma
    )
    variance_beam = 0.25 * ((p - q) ** 2 * pair_plus + (p + q) ** 2 * pair_minus)
    variance_dark = np.full(p.size, 2 * dark_sigma**2, dtype=np.float64)
    variance_bias_white = np.full(p.size, 2 * (i0 * bias_sigma) ** 2, dtype=np.float64)
    variance_bias_drift = np.full(
        p.size, (i0 * bias_gradient_sigma) ** 2, dtype=np.float64
    )
    repeated = float(repeat_acquisitions)
    variance = (
        variance_dark + variance_bias_white + variance_bias_drift + variance_beam
    ) / repeated
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Oracle variances must be positive and finite")
    weight = 1.0 / variance
    inverse_std = 1.0 / np.sqrt(variance)
    if not (np.isfinite(weight).all() and np.isfinite(inverse_std).all()):
        raise ValueError("Oracle weights or whiteners are not finite")

    return OracleVarianceTerms(
        variance=variance,
        weight=weight,
        inverse_std=inverse_std,
        dark=variance_dark / repeated,
        bias_white=variance_bias_white / repeated,
        bias_drift=variance_bias_drift / repeated,
        beam_integrated_arma=variance_beam / repeated,
    )
