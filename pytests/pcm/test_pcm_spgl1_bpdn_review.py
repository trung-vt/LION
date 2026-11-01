"""Regression checks added after the external SPGL1-BPDN review (CPU only)."""

from __future__ import annotations

import torch
import pytest
from spgl1 import spg_bp
from scipy.sparse.linalg import LinearOperator
import numpy as np

from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.spgl1_bpdn import (
    SPGL1BPDNParameter,
    SPGL1BPDNReconFn,
    within_bpdn_bound,
    spgl1_bpdn_reconstruction,
)
from LION.pcm.spgl1_bpdn_debias import debias_bpdn_coefficients
from LION.pcm.reconstruct import ReconInput, SolverDiagnosticsReconOutput


class IdentityOp:
    device = torch.device("cpu")

    def forward(self, v):
        return v

    def __call__(self, v):
        return self.forward(v)

    def adjoint(self, v):
        return v


def test_feasibility_uses_spgl1_scaled_stopping_tolerance():
    # Old fixed 0.1% check falsely rejects this accepted root.
    assert within_bpdn_bound(0.0100284, 0.01, opt_tol=1e-4, factor=1.0)
    assert not within_bpdn_bound(0.011, 0.01, opt_tol=1e-4, factor=1.0)
    assert within_bpdn_bound(0.0100001, 0.01, opt_tol=1e-4, factor=10.0)
    assert not within_bpdn_bound(0.012, 0.01, opt_tol=1e-4, factor=10.0)


def test_legacy_bp_solver_still_uses_same_exact_problem():
    y = torch.tensor([1.0, 0.5, 0.25, 0.1], dtype=torch.float32)
    a = LinearOperator(
        (4, 4), matvec=lambda v: v, rmatvec=lambda v: v, dtype=np.float32
    )
    expected, _, _, expected_info = spg_bp(a, y.numpy(), iter_lim=100, verbosity=0)
    actual, info = spgl1_torch(IdentityOp(), y, iter_lim=100, verbosity=0)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert info["stat"] == expected_info["stat"]


def test_empty_pcm_batch_is_rejected_with_a_clear_message():
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(64), normalized=True, device="cpu"
    )
    with pytest.raises(ValueError, match="nonempty"):
        spgl1_bpdn_reconstruction(
            op, torch.zeros(0, 1, 64), SPGL1BPDNParameter(epsilon=0.3)
        )


def test_reconstruction_returns_diagnostics_in_output():
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(64), normalized=True, device="cpu"
    )
    torch.manual_seed(9)
    x = torch.rand(1, 1, 16, 16)
    y = op(x)
    method = SPGL1BPDNReconFn(
        SPGL1BPDNParameter(epsilon=0.5, wavelet_name="db1", iters=100),
        device=torch.device("cpu"),
    )
    result = method(ReconInput(op=op, y=y, x0=op.pseudo_inv(y), i0=None))
    assert isinstance(result, SolverDiagnosticsReconOutput)
    assert result.diagnostics[0]["solver_status_name"] in {
        "root_found",
        "optimal_fixed_tau",
        "suboptimal_basis_pursuit",
    }
    assert "coefficient_l1_norm" in result.diagnostics[0]
    assert "coefficient_nnz_at_1e_minus_5" in result.diagnostics[0]


def test_unit_step_debias_avoids_power_method(monkeypatch):
    import importlib

    module = importlib.import_module("LION.operators.DebiasOp")

    def forbidden(*args, **kwargs):
        raise AssertionError("power_method must not run when step_size=1")

    monkeypatch.setattr(module, "power_method", forbidden)
    y = torch.tensor([1.0, 0.5, 0.1, 0.0])
    w = torch.tensor([0.5, 0.2, 0.0, 0.0])
    debiased, report = debias_bpdn_coefficients(
        IdentityOp(),
        y,
        w,
        max_iter=10,
        step_size=1.0,
        support_tol=1e-5,
        tol=1e-7,
    )
    torch.testing.assert_close(debiased[:2], y[:2])
    assert report.residual_after < report.residual_before


def test_independent_collection_does_not_alias_historical_bp():
    from LION.experiments.pcm_presets.presets import NOISY_PCM_METHODS, QUICK_METHODS

    assert NOISY_PCM_METHODS["spgl1"] is not QUICK_METHODS["spgl1"]
    assert NOISY_PCM_METHODS["spgl1"].factor == QUICK_METHODS["spgl1"].factor
