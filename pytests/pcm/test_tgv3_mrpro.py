"""Tests for the MRPro-backed 2-D TGV-3 operator and PDHG integration."""

from __future__ import annotations

import pytest
import torch
from mrpro.operators import IdentityOp

from LION.config.sapm_config import AdaptivePDHGConfig, SAPMModelConfig
from LION.models.AdaptiveTGV3PCMNetwork import AdaptiveTGV3PCMNetwork
from LION.operators.FiniteDifferenceOp import FiniteDifferenceOp
from LION.operators.SymmetrizedSecondOrderGradientOp import (
    SymmetrizedSecondOrderGradientOp,
)


def _model() -> AdaptiveTGV3PCMNetwork:
    cfg = SAPMModelConfig(
        name="tgv3-pcm-test",
        input_image="zeros",
        initial_image="zeros",
        n_iters=20,
        backend="new",
        cnn=None,
        pdhg=AdaptivePDHGConfig(
            regularisation="TGV3",
            primal_stepsize=0.15,
            dual_stepsize=0.15,
            initial_scale_factor_raw=0.1,
            scale_factor_softplus_beta=1.0,
            lambda_softplus_beta=1.0,
            fd_pad_mode=["zeros", "zeros"],
            fd_dim=[-2, -1],
            theta=1.0,
        ),
    )
    return AdaptiveTGV3PCMNetwork(config=cfg, lambda_map_network=None)


@pytest.mark.parametrize("pad", ["zeros", "circular"])
def test_e2_full_symmetry_explicit_and_adjoint(pad: str) -> None:
    torch.manual_seed(42)
    operator = SymmetrizedSecondOrderGradientOp(pad_mode=pad)
    w = torch.randn(1, 2, 2, 1, 7, 9, dtype=torch.float64)
    w = (w + w.transpose(1, 2)) * 0.5
    y = operator(w)[0]
    assert y.shape == (2, 1, 2, 2, 1, 7, 9)
    for i, j in ((0, 2), (0, 3), (2, 3)):
        torch.testing.assert_close(y, y.transpose(i, j), rtol=0, atol=1e-12)
    derivatives = FiniteDifferenceOp(dim=(-2, -1), mode="backward", pad_mode=pad)(w)[0]
    expected = (
        derivatives + derivatives.transpose(0, 2) + derivatives.transpose(0, 3)
    ) / 3.0
    torch.testing.assert_close(y, expected, rtol=0, atol=1e-12)
    dual = torch.randn_like(y)
    torch.testing.assert_close(
        (y * dual).sum(),
        (w * operator.H(dual)[0]).sum(),
        rtol=1e-11,
        atol=1e-11,
    )


def test_e2_general_adjoint_including_non_symmetric_w() -> None:
    torch.manual_seed(23)
    op = SymmetrizedSecondOrderGradientOp()
    w = torch.randn((2, 2, 2, 1, 8, 9), dtype=torch.float64)
    q = torch.randn((2, 2, 2, 2, 1, 8, 9), dtype=torch.float64)
    torch.testing.assert_close(
        (op(w)[0] * q).sum(),
        (w * op.H(q)[0]).sum(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_k_block_shapes_adjoint_and_symmetry() -> None:
    torch.manual_seed(12)
    model = _model()
    op = model.make_stacked_operator(IdentityOp())
    assert op.shape == (4, 3)
    u = torch.randn((1, 1, 7, 9), dtype=torch.float64)
    v = torch.randn((1, 2, 1, 7, 9), dtype=torch.float64)
    w = torch.randn((1, 2, 2, 1, 7, 9), dtype=torch.float64)
    w = (w + w.transpose(1, 2)) / 2.0
    primals = (u, v, w)
    kx = op(*primals)
    assert [tuple(a.shape) for a in kx] == [
        (1, 1, 7, 9),
        (1, 2, 1, 7, 9),
        (1, 2, 2, 1, 7, 9),
        (1, 2, 2, 2, 1, 7, 9),
    ]
    duals = tuple(torch.randn_like(x) for x in kx)
    lhs = sum((a * b).sum() for a, b in zip(kx, duals, strict=True))
    rhs = sum((a * b).sum() for a, b in zip(primals, op.H(*duals), strict=True))
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_weight_maps_and_nonnegativity() -> None:
    weights = torch.ones((2, 3, 9, 7))
    weights[:, 0] *= 0.5
    weights[:, 1] *= 0.3
    weights[:, 2] *= 0.1
    a2, a1, a0 = _model().split_lambda(weights)
    assert a2.shape == (2, 1, 1, 9, 7)
    assert a1.shape == (2, 1, 1, 1, 9, 7)
    assert a0.shape == (2, 1, 1, 1, 1, 9, 7)
    with pytest.raises(ValueError):
        _model().split_lambda(weights[:, :2])
    with pytest.raises(ValueError):
        _model().split_lambda((-0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        _model().split_lambda(weights * float("nan"))


def test_dual_projections_and_fidelity_prox() -> None:
    torch.manual_seed(9)
    model = _model()
    x = torch.randn((1, 1, 7, 9))
    measurement = torch.rand_like(x)
    blocks = model.make_stacked_operator(IdentityOp())(*model.make_initial_values(x))
    f = model.make_f(measurement, lambda_reg=(0.3, 0.2, 0.1))
    vals = tuple(torch.randn_like(block) for block in blocks)
    sigma = 0.4
    prox = f.prox_convex_conj(*vals, sigma=sigma)
    expected_eta = (vals[0] - sigma * measurement) / (1 + sigma)
    torch.testing.assert_close(prox[0], expected_eta, rtol=1e-5, atol=1e-6)
    for tensor, weight in zip(prox[1:], (0.3, 0.2, 0.1), strict=True):
        assert float(tensor.abs().max()) <= weight + 1e-6


def test_scalar_and_constant_maps_give_same_reconstruction() -> None:
    torch.manual_seed(7)
    model = _model()
    u0 = torch.zeros((1, 1, 8, 8))
    y = torch.randn_like(u0)
    weights = (0.07, 0.04, 0.02)
    spatial = torch.tensor(weights, dtype=u0.dtype)[None, :, None, None]
    spatial = spatial.expand(1, 3, 8, 8).clone()
    with torch.no_grad():
        a = model(
            input_image=None,
            initial_image=u0,
            measurement=y,
            key=0,
            forward_op=IdentityOp(),
            lambda_reg=weights,
            max_iterations=25,
        )
        b = model(
            input_image=None,
            initial_image=u0,
            measurement=y,
            key=0,
            forward_op=IdentityOp(),
            lambda_reg=spatial,
            max_iterations=25,
        )
    assert bool(torch.isfinite(a).all())
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_shared_pdhg_preserves_symmetry_and_matches_model_output() -> None:
    """Full primal tuple remains symmetric and uses the shared PDHG loop."""
    from LION.classical_algorithms.pdhg_mrpro import pdhg

    torch.manual_seed(15)
    model = _model()
    image = torch.zeros((1, 1, 8, 8))
    measurement = torch.randn_like(image)
    weights = (0.05, 0.025, 0.01)
    stack = model.make_stacked_operator(IdentityOp())
    primal = pdhg(
        f=model.make_f(measurement, lambda_reg=weights),
        g=model.g,
        operator=stack,
        initial_values=model.make_initial_values(image),
        max_iterations=35,
        primal_stepsize=0.15,
        dual_stepsize=0.15,
    )
    assert len(primal) == 3
    u, v, w = primal
    assert u.shape == image.shape
    assert v.shape == (1, 2, 1, 8, 8)
    assert w.shape == (1, 2, 2, 1, 8, 8)
    assert all(torch.isfinite(x).all() for x in primal)
    torch.testing.assert_close(w, w.transpose(1, 2), atol=1e-6, rtol=0)
    out = model(
        input_image=None,
        initial_image=image,
        measurement=measurement,
        key=0,
        forward_op=IdentityOp(),
        lambda_reg=weights,
        max_iterations=35,
    )
    torch.testing.assert_close(out, u, atol=1e-6, rtol=1e-6)


def test_differentiable_spatial_map_pdhg() -> None:
    """The existing checkpoint-enabled solver propagates finite map gradients."""
    torch.manual_seed(5)
    model = _model()
    image = torch.zeros((1, 1, 5, 5))
    measurement = torch.rand_like(image)
    raw_map = torch.randn((1, 3, 5, 5), requires_grad=True)
    positive_map = torch.nn.functional.softplus(raw_map)
    out = model(
        input_image=None,
        initial_image=image,
        measurement=measurement,
        key=0,
        forward_op=IdentityOp(),
        lambda_reg=positive_map,
        max_iterations=9,
        use_checkpoint=True,
        checkpoint_chunk_size=3,
    )
    out.square().mean().backward()
    assert raw_map.grad is not None
    assert bool(torch.isfinite(raw_map.grad).all())
