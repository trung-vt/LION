"""Closed-form proximal operator for the PCM data-fidelity term."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class PCMProxParameter(LIONParameter):
    """Configuration for the PCM data-fidelity proximal operator.

    Attributes:
        rho: Positive quadratic penalty in the supplied A,y units.
            CAUTION: for raw-to-normalized Hadamard OLS, rho_norm=rho_raw/n.
            See scripts/experiments/pcm/README_FIDELITY_SCALING.md.
    """

    rho: float | None = None


def pcm_prox_transform_rho(op: PhotocurrentMapOp, rho: float) -> float:
    """Return rho/kappa for the chosen A,y units (W W^T = kappa I).

    CAUTION: This internal transform factor does not convert an experiment's
    rho between raw and normalized Hadamard measurement conventions.
    For OLS such a conversion requires rho_norm = rho_raw/n.
    """
    return rho if op.w.sqrt_n is not None else rho / op.num_pixels


def closed_form_pcm_prox(
    op: PhotocurrentMapOp, *, rho: float, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Compute the closed-form PCM data-fidelity proximal update.

    Solves

        x* = argmin_x { 1/2 ||y - A_PCM x||_2^2 + rho/2 ||x - z||_2^2 },

    where

        A_PCM = S P W,

    W is a normalized or raw Walsh-Hadamard transform, P is a
    permutation, and S selects rows. Write W W^T = kappa I,
    with kappa = 1 for normalized W and kappa = n for raw W.
    In transform coordinates q = P W z and m = diag(S^T S),

        rho_transform = rho / kappa,
        alpha = (S^T y + rho_transform q) / (m + rho_transform).

    The division is componentwise. The image-domain result is

        x* = W^{-1} P^T alpha.

    Args:
        op: Photocurrent map operator containing S, P, and W.
        rho: Positive quadratic penalty parameter.
        y: Measurement.
        z: Input image to the proximal update.

    Returns:
        The minimizer x* (same shape as z)
    """
    z_vectorized = op.vectorize(z)
    q = op.p(op.w(z_vectorized))
    m = op.s.gram(torch.ones_like(z_vectorized))
    # CAUTION: External rho must already match the supplied A,y units.
    rho_transform = pcm_prox_transform_rho(op, rho)
    alpha = (op.s.adjoint(y) + rho_transform * q) / (m + rho_transform)
    x_vectorized = op.w.inv(op.p.adjoint(alpha))
    return op.unvectorize(x_vectorized)


def precompute_pcm_prox_terms(
    op: PhotocurrentMapOp, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute some terms for the closed-form PCM data-fidelity proximal update.

    Args:
        op: Photocurrent map operator containing S, P, and W.
        y: Measurement.

    Returns:
        ST_y: Reusable precomputed adjoint of the measurement vector.
        m: Reusable diagonal of the Gram matrix S^T S.
    """
    ST_y = op.s.adjoint(y)
    ones_vector = torch.ones_like(op.w.adjoint(op.sp.adjoint(y)))
    m = op.s.gram(ones_vector)
    return ST_y, m


def closed_form_pcm_prox_with_precomputed_terms(
    op: PhotocurrentMapOp,
    z: torch.Tensor,
    *,
    rho: float,
    ST_y: torch.Tensor,
    m_plus_rho: torch.Tensor,
) -> torch.Tensor:
    """Compute the closed-form PCM data-fidelity proximal update.

    Same as `closed_form_pcm_prox` except this version takes some precomputed terms to avoid redundant computations.

    Args:
        op: Photocurrent map operator containing S, P, and W.
        z: Input image to the proximal update.
        rho: Positive quadratic penalty parameter.
        ST_y: Reusable precomputed adjoint of the measurement vector.
        m_plus_rho: Reusable diagonal of S^T S plus the transform-domain
            penalty rho/kappa. This is not an external fidelity rescaling.

    Returns:
        The minimizer x* (same shape as z)
    """
    # return z
    z_vectorized = op.vectorize(z)
    debug_value(z_vectorized, "z_vectorized")
    wz = op.w(z_vectorized)
    debug_value(wz, "wz")
    q = op.p(wz)
    # q = op.p(op.w(z_vectorized))  # THIS STEP PRODUCES NaN VALUES
    # debug_value(q, "q")
    rho_transform = pcm_prox_transform_rho(op, rho)
    alpha = (ST_y + rho_transform * q) / m_plus_rho
    # debug_value(alpha, "alpha")
    x_vectorized = op.w.inv(op.p.adjoint(alpha))
    # debug_value(x_vectorized, "x_vectorized")
    return op.unvectorize(x_vectorized)


def debug_value(value: torch.Tensor, name: str) -> None:
    """Print debug information about a tensor value."""
    # # print(f"{name} has dtype: {value.dtype}, has NaN: {torch.isnan(value).any()}, has Inf: {torch.isinf(value).any()}")
    # print(
    #     f"{name} range: [{value.min()}, {value.max()}], has NaN: {torch.isnan(value).any()}, has Inf: {torch.isinf(value).any()}"
    # )
    # # if torch.isnan(value).any() or torch.isinf(value).any():
    # #     print(f"{name} has NaN: {torch.isnan(value).any()}, has Inf: {torch.isinf(value).any()}")
