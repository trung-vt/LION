"""Independent SPGL1 Basis Pursuit Denoising (BPDN) for noisy PCM.

Solve min ||w||_1 subject to ||A Psi^{-1} w - y||_2 <= epsilon.

The existing SPGL1 Basis Pursuit method is unchanged. The solver receives
only A and noisy y, never the ground truth. Epsilon is the entire L2 residual
bound in the normalized PCM measurement convention.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, replace

import torch

from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators.CompositeOp import CompositeOp
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.reconstruct import ReconInput, SolverDiagnosticsReconOutput
from LION.pcm.spgl1_bpdn_debias import debias_bpdn_coefficients
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class SPGL1BPDNParameter(LIONParameter):
    """Settings for BPDN (not BP), with optional post-solve debiasing.

    Attributes:
        noise_rms: Assumed RMS per differential measurement in normalized
            measurement units. Bound = noise_rms * sqrt(M). The default 0.06
            is a pilot starting value for 256x256 NPL-default noise, not an
            established statistical confidence bound.
        epsilon: Optional absolute residual bound in model measurement units.
            Overrides noise_rms when supplied.
        factor: Positive numerical scale applied to both y and epsilon.
            The final coefficient vector is divided by the same factor.
            For synthetic images in [0,1], use factor=1.
        iters: Maximum solver iterations, not a convergence guarantee.
        wavelet_name: Wavelet used for synthesis, e.g. db4.
        opt_tol: SPGL1 optimality tolerance.
        verbosity: SPGL1 verbosity, one of 0, 1 or 2.
        debias: Whether to run least-squares debiasing after BPDN (default False).
        debias_iters: Maximum support-restricted LS gradient steps (default 10).
        debias_support_tol: Absolute support threshold on factor-scaled coefficients.
        debias_tol: Relative stopping tolerance for the LS iteration.
    """

    noise_rms: float | None = 0.06
    epsilon: float | None = None
    factor: float = 1.0
    iters: int = 300
    wavelet_name: str = "db4"
    opt_tol: float = 1e-4
    verbosity: int = 0
    debias: bool = False
    debias_iters: int = 10
    debias_support_tol: float = 1e-5
    debias_tol: float = 1e-7

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("BPDN factor must be finite and positive.")
        if self.epsilon is None and self.noise_rms is None:
            raise ValueError("Supply BPDN epsilon or noise_rms.")
        if self.epsilon is not None and (
            not math.isfinite(self.epsilon) or self.epsilon <= 0
        ):
            raise ValueError("BPDN epsilon must be finite and positive.")
        if self.noise_rms is not None and (
            not math.isfinite(self.noise_rms) or self.noise_rms <= 0
        ):
            raise ValueError("BPDN noise_rms must be finite and positive.")
        if self.iters < 1:
            raise ValueError("BPDN iters must be >= 1.")
        if not math.isfinite(self.opt_tol) or self.opt_tol <= 0:
            raise ValueError("BPDN opt_tol must be finite and positive.")
        if self.epsilon is not None:
            # The explicit bound takes precedence; do not save an inactive RMS.
            self.noise_rms = None
        if self.verbosity not in (0, 1, 2):
            raise ValueError("BPDN verbosity must be 0, 1 or 2.")
        if self.debias_iters < 1:
            raise ValueError("BPDN debias_iters must be >= 1.")
        if not math.isfinite(self.debias_support_tol) or self.debias_support_tol < 0:
            raise ValueError("BPDN debias_support_tol must be finite and nonnegative.")
        if not math.isfinite(self.debias_tol) or self.debias_tol <= 0:
            raise ValueError("BPDN debias_tol must be finite and positive.")

    def residual_bound(self, measurement_count: int) -> float:
        """Return the full residual L2 threshold before numeric scaling."""
        if measurement_count < 1:
            raise ValueError("BPDN measurement vector cannot be empty.")
        if self.epsilon is not None:
            return float(self.epsilon)
        assert self.noise_rms is not None
        return float(self.noise_rms * math.sqrt(measurement_count))


@dataclass(frozen=True, kw_only=True)
class SPGL1BPDNDiagnostics:
    """Solver status and directly checked residual for a single image."""

    measurement_count: int
    epsilon: float
    factor: float
    scaled_epsilon: float
    residual_norm: float
    relative_residual: float
    solver_status: int
    iterations: int
    feasible_with_tolerance: bool
    solver_status_name: str = "unknown"
    solver_converged: bool = False
    # SPGL1 root_found only checks the residual constraint, not optimality.
    # rgap is SPGL1's own relative duality-gap diagnostic for the L1 subproblem.
    relative_duality_gap: float | None = None
    spgl1_l1_ball_radius: float | None = None
    coefficient_l1_norm: float = 0.0
    coefficient_nnz_at_1e_minus_5: int = 0
    lambda_equivalent: float | None = None
    debias_enabled: bool = False
    debias_applied: bool = False
    debias_support_size: int | None = None
    pre_debias_l1_norm: float | None = None
    post_debias_l1_norm: float | None = None
    post_debias_residual_norm: float | None = None
    post_debias_feasible_with_tolerance: bool | None = None
    post_debias_residual_to_epsilon: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics."""
        return asdict(self)


SPGL1_STATUS_NAMES = {
    1: "root_found",
    2: "basis_pursuit_solution",
    3: "least_squares_solution",
    4: "optimal_fixed_tau",
    5: "iteration_limit",
    6: "line_search_failure",
    7: "suboptimal_basis_pursuit",
    8: "operator_call_limit",
    9: "active_set_stalled",
}


def within_bpdn_bound(
    residual_norm: float, epsilon: float, opt_tol: float, factor: float
) -> bool:
    """Test the SPGL1 residual bound in the solver's scaled measurement units.

    SPGL1 checks |r-sigma|/max(1,r) <= opt_tol, not a fixed 0.1 percent
    bound. Only the upper side matters for primal feasibility; a small
    float32 roundoff allowance accounts for recomputing the residual.
    """
    scaled_residual = factor * residual_norm
    scaled_epsilon = factor * epsilon
    roundoff = (
        64 * torch.finfo(torch.float32).eps * max(1.0, scaled_residual, scaled_epsilon)
    )
    return scaled_residual <= (
        scaled_epsilon + opt_tol * max(1.0, scaled_residual) + roundoff
    )


def spgl1_bpdn_torch(
    op: Operator,
    measurement: torch.Tensor,
    *,
    epsilon: float,
    factor: float = 1.0,
    iters: int = 300,
    opt_tol: float = 1e-4,
    verbosity: int = 0,
) -> tuple[torch.Tensor, SPGL1BPDNDiagnostics]:
    """Solve one unweighted BPDN problem using a torch forward/adjoint operator.

    The 1D measurement and epsilon are in normalized PCM measurement units.
    No ground truth or post-solver least-squares debiasing is used. The caller
    should inspect both the returned status and the measured residual.
    """
    if measurement.ndim != 1 or measurement.numel() == 0:
        raise ValueError("BPDN expects a nonempty 1D measurement vector.")
    if not torch.isfinite(measurement).all():
        raise ValueError("BPDN measurement has nonfinite values.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("BPDN epsilon must be finite and positive.")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("BPDN factor must be finite and positive.")
    if iters < 1:
        raise ValueError("BPDN iters must be >= 1.")
    if not math.isfinite(opt_tol) or opt_tol <= 0:
        raise ValueError("BPDN opt_tol must be finite and positive.")
    if verbosity not in (0, 1, 2):
        raise ValueError("BPDN verbosity must be 0, 1 or 2.")

    # CompositeOp and Wavelet2D already run float32; preserve their dtype.
    y = measurement.detach().to(dtype=torch.float32)
    scaled_measurement = y * factor
    scaled_epsilon = epsilon * factor

    # Reuse LION's BP operator adapter. Exact BP still calls spg_bp;
    # a positive sigma calls spg_bpdn without duplicating the adapter.
    coefficients_scaled, info = spgl1_torch(
        op,
        scaled_measurement,
        sigma=scaled_epsilon,
        iter_lim=iters,
        opt_tol=opt_tol,
        verbosity=verbosity,
    )
    coefficients = coefficients_scaled / factor
    residual_norm = float(torch.linalg.vector_norm(op(coefficients) - y))
    feasible = within_bpdn_bound(residual_norm, epsilon, opt_tol, factor)
    status = int(info.get("stat", -1))
    diagnostic = SPGL1BPDNDiagnostics(
        measurement_count=y.numel(),
        epsilon=float(epsilon),
        factor=float(factor),
        scaled_epsilon=float(scaled_epsilon),
        residual_norm=residual_norm,
        relative_residual=residual_norm / (float(torch.linalg.vector_norm(y)) + 1e-12),
        solver_status=status,
        iterations=int(info.get("niters", -1)),
        feasible_with_tolerance=bool(feasible),
        solver_status_name=SPGL1_STATUS_NAMES.get(status, "unknown"),
        solver_converged=status in (1, 2, 4),
        relative_duality_gap=(float(info["rgap"]) if "rgap" in info else None),
        spgl1_l1_ball_radius=(float(info["tau"]) / factor if "tau" in info else None),
        coefficient_l1_norm=float(torch.linalg.vector_norm(coefficients, ord=1)),
        coefficient_nnz_at_1e_minus_5=int(
            torch.count_nonzero(coefficients.abs() > 1e-5)
        ),
        lambda_equivalent=(float(info["gnorm"]) / factor if "gnorm" in info else None),
    )
    if not feasible:
        warnings.warn(
            f"SPGL1-BPDN residual {residual_norm:.6g} exceeds epsilon "
            f"{epsilon:.6g}; check convergence before comparing methods.",
            RuntimeWarning,
            stacklevel=2,
        )
    return coefficients, diagnostic


def spgl1_bpdn_reconstruction(
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    config: SPGL1BPDNParameter,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list[SPGL1BPDNDiagnostics]]:
    """Reconstruct a PCM batch; return BCHW images and solver diagnostics."""
    if measurement.ndim != 3 or measurement.shape[1] != 1 or measurement.shape[0] == 0:
        raise ValueError("BPDN requires nonempty PCM measurements of shape (B,1,M).")
    if measurement.shape[-1] != op.range_shape[-1]:
        raise ValueError("BPDN measurement length does not match the operator.")

    height, width = op.domain_shape
    actual_device = device or measurement.device
    wavelet = Wavelet2D(
        (height, width), wavelet_name=config.wavelet_name, device=actual_device
    )
    forward_op = CompositeOp(wavelet, op, device=actual_device)

    images: list[torch.Tensor] = []
    diagnostics: list[SPGL1BPDNDiagnostics] = []
    for one in measurement:
        y = one.squeeze(0)
        coefficients, report = spgl1_bpdn_torch(
            forward_op,
            y,
            epsilon=config.residual_bound(y.numel()),
            factor=config.factor,
            iters=config.iters,
            opt_tol=config.opt_tol,
            verbosity=config.verbosity,
        )
        if config.debias:
            coefficients, db = debias_bpdn_coefficients(
                forward_op,
                y,
                coefficients,
                factor=config.factor,
                support_tol=config.debias_support_tol,
                max_iter=config.debias_iters,
                tol=config.debias_tol,
                # Unit step is valid only for orthonormal normalized PCM rows
                # and the periodized orthogonal wavelet synthesis transform.
                step_size=(
                    1.0
                    if op.w.sqrt_n is not None and wavelet.mode == "periodization"
                    else None
                ),
            )
            post_feasible = within_bpdn_bound(
                db.residual_after, report.epsilon, config.opt_tol, config.factor
            )
            report = replace(
                report,
                debias_enabled=True,
                debias_applied=db.applied,
                debias_support_size=db.support_size,
                pre_debias_l1_norm=db.l1_before,
                post_debias_l1_norm=db.l1_after,
                post_debias_residual_norm=db.residual_after,
                post_debias_feasible_with_tolerance=bool(post_feasible),
                post_debias_residual_to_epsilon=(db.residual_after / report.epsilon),
            )
            if not post_feasible:
                warnings.warn(
                    "Post-BPDN debiasing produced a residual above the BPDN "
                    "constraint; inspect solver_diagnostics.json.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        images.append(wavelet.inverse(coefficients))
        diagnostics.append(report)
    return torch.stack(images, dim=0).unsqueeze(1), diagnostics


@dataclass
class SPGL1BPDNReconFn:
    """Method-dispatch adapter for one stored or simulated PCM acquisition."""

    config: SPGL1BPDNParameter
    device: torch.device | None = None

    def __call__(self, recon_input: ReconInput) -> SolverDiagnosticsReconOutput:
        """Reconstruct an acquisition and return serializable diagnostics."""
        reconstruction, diagnostics = spgl1_bpdn_reconstruction(
            recon_input.op, recon_input.y, self.config, device=self.device
        )
        return SolverDiagnosticsReconOutput(
            reconstruction=reconstruction,
            diagnostics=[item.to_dict() for item in diagnostics],
        )
