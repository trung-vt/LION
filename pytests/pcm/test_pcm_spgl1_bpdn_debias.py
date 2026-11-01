"""CPU-only tests for optional SPGL1-BPDN least-squares debiasing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from LION.operators.DebiasOp import debias_ls
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.spgl1_bpdn import SPGL1BPDNParameter, spgl1_bpdn_reconstruction
from LION.pcm.spgl1_bpdn_debias import debias_bpdn_coefficients


class IdentityLinearOp(Operator):
    """Small identity map implementing the existing DebiasOp interface."""

    def __init__(self, n: int) -> None:
        super().__init__(device="cpu")
        self.n = n

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return (self.n,)

    @property
    def range_shape(self) -> tuple[int, ...]:
        return (self.n,)


def test_debias_matches_legacy_least_squares_on_identical_support() -> None:
    op = IdentityLinearOp(4)
    y = torch.tensor([1.0, 0.5, 0.02, -0.03])
    coeff = torch.tensor([0.6, 0.25, 0.0, 0.0])
    result, info = debias_bpdn_coefficients(
        op, y, coeff, support_tol=1e-5, max_iter=10, tol=1e-7
    )
    legacy_result = debias_ls(op, y, coeff, support_tol=1e-5, max_iter=10, tol=1e-7)
    assert torch.equal(result, legacy_result)
    assert info.applied
    assert info.support_size == 2
    assert result[0] == pytest.approx(1.0, abs=1e-5)
    assert result[1] == pytest.approx(0.5, abs=1e-5)
    assert torch.equal(result[2:], torch.zeros(2))
    assert info.residual_after < info.residual_before
    assert info.l1_after > info.l1_before
    assert info.to_dict()["support_size"] == 2


def test_debias_empty_support_returns_identical_coefficients() -> None:
    op = IdentityLinearOp(4)
    y = torch.tensor([1.0, -0.5, 0.25, 0.1])
    coeff = torch.zeros_like(y)
    result, info = debias_bpdn_coefficients(
        op, y, coeff, support_tol=1e-5, max_iter=10, tol=1e-7
    )
    assert torch.equal(result, coeff)
    assert not info.applied
    assert info.support_size == 0
    assert info.residual_before == pytest.approx(info.residual_after)
    assert info.l1_after == 0


def test_debias_support_uses_same_factor_convention_as_legacy_bp() -> None:
    op = IdentityLinearOp(4)
    y = torch.tensor([0.75, 0.5, 0.2, 0.0])
    coeff = torch.tensor([0.3, 0.1, 0.0, 0.0])
    factor = 10.0
    threshold = 1.5
    result, info = debias_bpdn_coefficients(
        op,
        y,
        coeff,
        factor=factor,
        support_tol=threshold,
        max_iter=10,
        tol=1e-7,
    )
    expected = (
        debias_ls(
            op,
            y * factor,
            coeff * factor,
            support_tol=threshold,
            max_iter=10,
            tol=1e-7,
        )
        / factor
    )
    assert torch.equal(result, expected)
    assert info.support_size == 1
    assert result[0] == pytest.approx(y[0].item(), abs=1e-5)
    assert result[1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("debias_iters", 0),
        ("debias_support_tol", -1.0),
        ("debias_tol", 0.0),
    ],
)
def test_debias_invalid_configuration(parameter: str, value: float) -> None:
    with pytest.raises(ValueError, match=parameter):
        SPGL1BPDNParameter(debias=True, **{parameter: value})


def test_no_debias_is_unchanged_and_debias_reports_two_stages() -> None:
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(128), normalized=True, device="cpu"
    )
    torch.manual_seed(2026)
    target = torch.rand(1, 1, 16, 16)
    observed = op(target) + 0.012 * torch.randn(1, 1, 128)
    base = SPGL1BPDNParameter(
        epsilon=0.20,
        wavelet_name="db1",
        iters=150,
        opt_tol=5e-4,
        factor=1.0,
        debias=False,
    )

    before, before_diag = spgl1_bpdn_reconstruction(op, observed, base)
    again, again_diag = spgl1_bpdn_reconstruction(op, observed, base)
    after, after_diag = spgl1_bpdn_reconstruction(
        op, observed, replace(base, debias=True)
    )

    assert torch.equal(before, again)
    assert before_diag[0] == again_diag[0]
    assert not before_diag[0].debias_enabled
    assert before_diag[0].post_debias_residual_norm is None
    assert after_diag[0].debias_enabled
    assert after_diag[0].debias_applied
    assert after_diag[0].debias_support_size > 0
    assert after_diag[0].solver_status == before_diag[0].solver_status
    assert after_diag[0].iterations == before_diag[0].iterations
    assert after_diag[0].residual_norm == pytest.approx(
        before_diag[0].residual_norm, abs=1e-6
    )
    assert after_diag[0].post_debias_residual_norm is not None
    assert after_diag[0].post_debias_residual_norm <= (
        after_diag[0].residual_norm + 2e-4
    )
    assert not torch.equal(after, before)
    assert torch.isfinite(after).all()


def test_debias_method_ids_are_registered_without_mutating_original_bp() -> None:
    from LION.experiments.pcm_presets.presets import (
        NOISY_PCM_METHODS,
        PRESETS,
        QUICK_METHODS,
    )

    assert "spgl1" in QUICK_METHODS
    assert "spgl1_bpdn" not in QUICK_METHODS
    assert "spgl1_bpdn_debiased" not in QUICK_METHODS
    assert "spgl1_bpdn" in NOISY_PCM_METHODS
    assert "spgl1_bpdn_debiased" in NOISY_PCM_METHODS
    assert not NOISY_PCM_METHODS["spgl1_bpdn"].debias
    assert NOISY_PCM_METHODS["spgl1_bpdn_debiased"].debias
    preset = PRESETS["synthetic_png_noisy_bpdn"]
    assert set(preset.methods) == {"spgl1", "spgl1_bpdn", "spgl1_bpdn_debiased"}
    assert not preset.methods["spgl1_bpdn"].debias
    assert preset.methods["spgl1_bpdn_debiased"].debias
