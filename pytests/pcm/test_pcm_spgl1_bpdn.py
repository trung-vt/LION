"""CPU-only unit and smoke tests for the new, separate SPGL1 BPDN method."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from LION.operators.CompositeOp import CompositeOp
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.reconstruct import ReconInput
from LION.pcm.spgl1_bpdn import (
    SPGL1BPDNParameter,
    SPGL1BPDNReconFn,
    spgl1_bpdn_reconstruction,
    spgl1_bpdn_torch,
)


class IdentityOp:
    """Minimal test operator with matching forward and adjoint."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_bpdn_parameter_and_threshold() -> None:
    params = SPGL1BPDNParameter(noise_rms=0.05, factor=1.0)
    assert params.residual_bound(100) == pytest.approx(0.5)
    fixed = SPGL1BPDNParameter(noise_rms=0.1, epsilon=0.25)
    assert fixed.residual_bound(100) == pytest.approx(0.25)
    assert fixed.noise_rms is None  # Inactive when an explicit epsilon is used.
    with pytest.raises(ValueError, match="Supply"):
        SPGL1BPDNParameter(noise_rms=None, epsilon=None)
    with pytest.raises(ValueError, match="factor"):
        SPGL1BPDNParameter(factor=0)
    with pytest.raises(ValueError, match="noise_rms"):
        SPGL1BPDNParameter(noise_rms=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        SPGL1BPDNParameter(epsilon=-0.1)
    with pytest.raises(ValueError, match="iters"):
        SPGL1BPDNParameter(iters=0)
    with pytest.raises(ValueError, match="empty"):
        params.residual_bound(0)


def test_bpdn_uses_a_nonzero_residual_tolerance_and_has_diagnostics() -> None:
    observed = torch.tensor([1.0, -0.5, 0.03, 0.0], dtype=torch.float32)
    epsilon = 0.12
    coefficients, info = spgl1_bpdn_torch(
        IdentityOp(), observed, epsilon=epsilon, factor=10.0, iters=200
    )
    assert coefficients.shape == observed.shape
    assert info.feasible_with_tolerance
    assert info.epsilon == pytest.approx(epsilon)
    assert info.residual_norm == pytest.approx(epsilon, abs=2e-3)
    assert info.iterations > 0
    assert info.solver_status == 1
    assert info.solver_converged
    assert info.solver_status_name == "root_found"
    assert torch.linalg.vector_norm(coefficients, ord=1) < torch.linalg.vector_norm(
        observed, ord=1
    )


def test_bpdn_factor_scales_measurements_and_bound_consistently() -> None:
    observed = torch.tensor([1.0, -0.5, 0.03, 0.0], dtype=torch.float32)
    coefficients_a, diag_a = spgl1_bpdn_torch(
        IdentityOp(), observed, epsilon=0.12, factor=1.0, iters=200
    )
    coefficients_b, diag_b = spgl1_bpdn_torch(
        IdentityOp(),
        observed,
        epsilon=0.12,
        factor=10.0,
        iters=200,
        opt_tol=1e-3,
    )
    assert diag_b.scaled_epsilon == pytest.approx(1.2)
    assert diag_a.feasible_with_tolerance
    assert diag_b.feasible_with_tolerance
    assert torch.allclose(coefficients_a, coefficients_b, atol=5e-3, rtol=5e-3)


def test_pcm_wavelet_operator_and_adjoint_agree_on_cpu() -> None:
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(64), normalized=True, device="cpu"
    )
    wavelet = Wavelet2D((16, 16), wavelet_name="db1", device="cpu")
    composite = CompositeOp(wavelet, op, device="cpu")
    torch.manual_seed(1234)
    coefficients = torch.randn(wavelet.size)
    residual = torch.randn(64)
    lhs = torch.dot(composite(coefficients), residual)
    rhs = torch.dot(coefficients, composite.adjoint(residual))
    assert torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5)


def test_bpdn_pcm_batch_shape_and_residual_cpu() -> None:
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(128), normalized=True, device="cpu"
    )
    torch.manual_seed(876)
    target = torch.rand(2, 1, 16, 16)
    measurement = op(target)
    noise = 0.01 * torch.randn_like(measurement)
    y = measurement + noise
    config = SPGL1BPDNParameter(
        epsilon=0.15, wavelet_name="db1", iters=120, opt_tol=5e-4
    )
    reconstruction, reports = spgl1_bpdn_reconstruction(
        op, y, config, device=torch.device("cpu")
    )
    assert reconstruction.shape == target.shape
    assert reconstruction.isfinite().all()
    assert len(reports) == 2
    assert all(r.feasible_with_tolerance for r in reports)
    residual = torch.linalg.vector_norm(
        (op(reconstruction) - y).flatten(start_dim=1), dim=1
    )
    assert bool((residual < 0.152).all())

    recon_fn = SPGL1BPDNReconFn(config, device=torch.device("cpu"))
    result = recon_fn(ReconInput(op=op, y=y, x0=op.pseudo_inv(y), i0=None))
    assert result.reconstruction.shape == target.shape
    assert len(result.diagnostics) == 2
    assert result.diagnostics[0]["solver_status"] in (1, 2, 4)
    assert "coefficient_l1_norm" in result.diagnostics[0]


def test_bpdn_registration_does_not_change_legacy_bp() -> None:
    from LION.experiments.pcm_presets.presets import (
        NOISY_PCM_METHODS,
        PRESETS,
        QUICK_METHODS,
    )

    assert "spgl1" in QUICK_METHODS
    assert "spgl1_bpdn" not in QUICK_METHODS
    assert "spgl1_bpdn" in NOISY_PCM_METHODS
    assert isinstance(NOISY_PCM_METHODS["spgl1_bpdn"], SPGL1BPDNParameter)
    assert "synthetic_png_noisy_bpdn" in PRESETS
    preset = PRESETS["synthetic_png_noisy_bpdn"]
    assert set(preset.methods) == {"spgl1", "spgl1_bpdn", "spgl1_bpdn_debiased"}
    assert not preset.methods["spgl1_bpdn"].debias
    assert preset.methods["spgl1_bpdn_debiased"].debias
    assert preset.noise is not None
    assert preset.sampling.sampling_ratios == [0.2]
    assert preset.sampling.coarse_j_values == [6]
    assert math.isclose(preset.methods["spgl1_bpdn"].factor, 1.0)
