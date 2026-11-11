"""Regression tests for canonical Chambolle-Pock extrapolation in LION PDHG.

MRPro corrected the old lerp(old, new, relaxation) expression in PR #999.
These tests catch accidental reinstatement of the old theta=0 behaviour.
"""

import pytest
import torch
from mrpro.operators import IdentityOp
from mrpro.operators.functionals import L2NormSquared

from LION.classical_algorithms.pdhg_mrpro import pdhg


def _reference_two_iterations(
    target: torch.Tensor, theta: float, tau: float, sigma: float
) -> torch.Tensor:
    """Direct scalar Chambolle-Pock update for f(z)=0.5||z-target||^2."""
    x = torch.zeros_like(target)
    y = torch.zeros_like(target)
    extrapolated = x
    for _ in range(2):
        y = (y + sigma * (extrapolated - target)) / (1.0 + sigma)
        new_x = x - tau * y
        extrapolated = new_x + theta * (new_x - x)
        x = new_x
    return x


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_matches_canonical_two_step_chambolle_pock(theta: float) -> None:
    target = torch.tensor([[[[0.1, -0.2], [0.5, 0.9]]]], dtype=torch.float64)
    initial = torch.zeros_like(target)
    tau, sigma = 0.2, 0.3
    expected = _reference_two_iterations(target, theta, tau, sigma)
    with torch.no_grad():
        (result,) = pdhg(
            f=0.5 * L2NormSquared(target=target),
            g=None,
            operator=IdentityOp(),
            initial_values=(initial,),
            max_iterations=2,
            primal_stepsize=tau,
            dual_stepsize=sigma,
            relaxation=theta,
        )
    # PDHG stores float step sizes in float32 even for float64 images.
    torch.testing.assert_close(result, expected, atol=2e-8, rtol=2e-7)


def test_default_is_canonical_theta_one_not_theta_zero() -> None:
    target = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    zero = torch.zeros_like(target)
    params = {
        "f": 0.5 * L2NormSquared(target=target),
        "g": None,
        "operator": IdentityOp(),
        "initial_values": (zero,),
        "max_iterations": 2,
        "primal_stepsize": 0.25,
        "dual_stepsize": 0.25,
    }
    with torch.no_grad():
        (default,) = pdhg(**params)
        (explicit_cp,) = pdhg(**params, relaxation=1.0)
        (no_extrapolation,) = pdhg(**params, relaxation=0.0)
    torch.testing.assert_close(default, explicit_cp, atol=0, rtol=0)
    torch.testing.assert_close(
        default,
        _reference_two_iterations(target, 1.0, 0.25, 0.25),
        atol=1e-12,
        rtol=1e-12,
    )
    assert not torch.allclose(default, no_extrapolation, atol=1e-8, rtol=1e-8)


def test_checkpointed_iterations_match_uncached_result_and_gradients() -> None:
    torch.manual_seed(13)
    target = torch.rand((1, 1, 5, 5), dtype=torch.float64)
    initial = torch.zeros_like(target)
    f = 0.5 * L2NormSquared(target=target)

    def reconstruct(checkpointed: bool):
        x = initial.clone().detach().requires_grad_(True)
        (result,) = pdhg(
            f=f,
            g=None,
            operator=IdentityOp(),
            initial_values=(x,),
            max_iterations=6,
            primal_stepsize=0.2,
            dual_stepsize=0.3,
            relaxation=1.0,
            use_checkpoint=checkpointed,
            checkpoint_chunk_size=2,
        )
        loss = result.square().mean()
        (gradient,) = torch.autograd.grad(loss, x)
        return result, gradient

    uncached, uncached_grad = reconstruct(False)
    checkpointed, checkpointed_grad = reconstruct(True)
    torch.testing.assert_close(checkpointed, uncached, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        checkpointed_grad,
        uncached_grad,
        atol=1e-12,
        rtol=1e-12,
    )
