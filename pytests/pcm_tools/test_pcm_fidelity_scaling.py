"""Fidelity scaling checks for PCM proximal algorithms (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from LION.classical_algorithms.conjugate_gradient import CGProxParameter
from LION.experiments.config.other_config import NormParameter
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import PCMProxParameter, closed_form_pcm_prox
from LION.reconstructors.GSPnP import (
    GSDenoiser,
    GSPnP,
    GSPnPDeepInv,
    PCMDeepInvPhysics,
)
from LION.reconstructors.PnPADMM import PnPADMM, PnPADMMParameter
from LION.reconstructors.BasePnP import DenoiserParameter


def _case(normalized: bool) -> tuple[PhotocurrentMapOp, torch.Tensor, torch.Tensor]:
    op = PhotocurrentMapOp(
        j_order=2,
        s_indices=np.array([0, 1, 4, 7, 10, 13]),
        normalized=normalized,
        device="cpu",
    )
    torch.manual_seed(57)
    y = torch.randn(op.range_shape[0], dtype=torch.float64)
    z = torch.randn(*op.domain_shape, dtype=torch.float64)
    return op, y, z


@pytest.mark.parametrize("normalized", [True, False])
def test_pcm_ols_raw_normalized_equivalent_with_rescaled_rho(
    normalized: bool,
) -> None:
    op, y, z = _case(normalized)
    rho = 0.31
    result = closed_form_pcm_prox(op, rho=rho, y=y, z=z)
    other = PhotocurrentMapOp(
        j_order=2,
        s_indices=np.array([0, 1, 4, 7, 10, 13]),
        normalized=not normalized,
        device="cpu",
    )
    # A_raw = sqrt(n) A_norm; y_raw = sqrt(n) y_norm.
    scale = op.num_pixels if normalized else 1.0 / op.num_pixels
    y_other = y * (op.num_pixels**0.5 if normalized else op.num_pixels**-0.5)
    result_other = closed_form_pcm_prox(other, rho=rho * scale, y=y_other, z=z)
    torch.testing.assert_close(result_other, result, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("normalized", [True, False])
def test_pnp_admm_raw_normalized_equivalent_with_rescaled_rho(
    normalized: bool,
) -> None:
    op, y, _ = _case(normalized)
    other = PhotocurrentMapOp(
        j_order=2,
        s_indices=np.array([0, 1, 4, 7, 10, 13]),
        normalized=not normalized,
        device="cpu",
    )
    rho = 0.23
    scale = op.num_pixels if normalized else 1.0 / op.num_pixels
    y_other = y * (op.num_pixels**0.5 if normalized else op.num_pixels**-0.5)

    def denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return 0.8 * x + 0.1

    base = PnPADMM(op, denoiser)
    shifted = PnPADMM(other, denoiser)
    args = dict(denoiser=DenoiserParameter(name="drunet"), iters=5, sigma=0.05)
    recon = base.admm_algorithm(
        y, PnPADMMParameter(**args, prox=PCMProxParameter(rho=rho))
    )
    recon_other = shifted.admm_algorithm(
        y_other,
        PnPADMMParameter(**args, prox=PCMProxParameter(rho=rho * scale)),
    )
    torch.testing.assert_close(recon_other, recon, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("normalize_fidelity", [True, False])
@pytest.mark.parametrize("data_weight", [1.0, 2.0])
def test_pnp_admm_cg_honours_fidelity_scaling(
    normalized: bool, normalize_fidelity: bool, data_weight: float
) -> None:
    op, y, z = _case(normalized)
    penalty = 0.45
    config = CGProxParameter(
        iters=40,
        eta=penalty,
        eps=1e-30,
        rel_tol=1e-12,
        normalize=normalize_fidelity,
        data_weight=data_weight,
    )
    solver = PnPADMM(op, denoiser=lambda x, sigma: x)
    cg_result = solver.prep_prox(config, y)(z)
    c = data_weight / (len(y) if normalize_fidelity else 1)
    expected = closed_form_pcm_prox(op, rho=penalty / c, y=y, z=z)
    torch.testing.assert_close(cg_result, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("normalize_fidelity", [True, False])
@pytest.mark.parametrize("factor", [None, 1.5])
@pytest.mark.parametrize("method", ["pcm", "cg"])
def test_gspnp_native_uses_same_ols_fidelity_in_objective_and_prox(
    normalized: bool, normalize_fidelity: bool, factor: float | None, method: str
) -> None:
    op, y, z = _case(normalized)
    denoiser = GSDenoiser(torch.nn.Identity())
    solver = GSPnP(op, denoiser)
    tau = 1.7
    fidelity = NormParameter(ord=2, normalize=normalize_fidelity, factor=factor)
    prox_params = (
        PCMProxParameter()
        if method == "pcm"
        else CGProxParameter(
            iters=40, eps=1e-30, rel_tol=1e-12, normalize=not normalize_fidelity
        )
    )
    prox = solver.prep_data_fidelity_prox(prox_params, y, (-1,), data_fidelity=fidelity)
    c = 2 * (0.5 if factor is None else factor)
    if normalize_fidelity:
        c /= len(y)
    expected = closed_form_pcm_prox(op, rho=1.0 / (tau * c), y=y, z=z)
    actual = prox(z, tau)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    objective = solver.prep_data_fidelity(fidelity, (-1,))(z, y)
    true_objective = c * 0.5 * torch.sum((op(z) - y) ** 2)
    assert objective == pytest.approx(true_objective.item(), rel=1e-12)


def test_gspnp_native_rejects_incompatible_nonquadratic_prox() -> None:
    op, y, _ = _case(True)
    solver = GSPnP(op, GSDenoiser(torch.nn.Identity()))
    with pytest.raises(ValueError, match="quadratic|squared"):
        solver.prep_data_fidelity_prox(
            PCMProxParameter(),
            y,
            (-1,),
            data_fidelity=NormParameter(ord=1, normalize=False),
        )


@pytest.mark.parametrize(
    "fidelity",
    [
        NormParameter(ord=2, normalize=True),
        NormParameter(ord=2, normalize=False, factor=0.75),
        NormParameter(ord=1, normalize=False),
    ],
)
def test_gspnp_deepinv_rejects_inconsistent_data_fidelity(
    fidelity: NormParameter,
) -> None:
    op, _, _ = _case(True)
    solver = GSPnPDeepInv(PCMDeepInvPhysics(op), GSDenoiser(torch.nn.Identity()))
    config = SimpleNamespace(data_fidelity=fidelity)
    with pytest.raises(ValueError, match="unnormalized squared L2"):
        solver._build_pgd(config)


@pytest.mark.parametrize("factor", [None, 0.5])
def test_gspnp_deepinv_accepts_unnormalized_l2(
    factor: float | None,
) -> None:
    op, _, _ = _case(True)
    solver = GSPnPDeepInv(PCMDeepInvPhysics(op), GSDenoiser(torch.nn.Identity()))
    config = SimpleNamespace(
        data_fidelity=NormParameter(ord=2, normalize=False, factor=factor),
        stepsize=None,
        lambd=0.1,
        use_backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.9,
        max_backtracking_attempts=4,
        sigma=0.05,
        iters=4,
        eps=1e-6,
    )
    pgd = solver._build_pgd(config)
    assert pgd is not None


@pytest.mark.parametrize("normalize", [True, False])
def test_gspnp_cg_direct_fallback_uses_data_weight(
    normalize: bool,
) -> None:
    op, y, z = _case(True)
    solver = GSPnP(op, GSDenoiser(torch.nn.Identity()))
    tau = 1.4
    config = CGProxParameter(
        iters=40, eps=1e-30, rel_tol=1e-12, normalize=normalize, data_weight=2.0
    )
    actual = solver.prep_data_fidelity_prox(config, y, (-1,))(z, tau)
    scale = 2.0 / len(y) if normalize else 2.0
    expected = closed_form_pcm_prox(op, rho=1.0 / (tau * scale), y=y, z=z)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
