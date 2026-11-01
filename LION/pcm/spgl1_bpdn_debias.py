"""Optional support-restricted least-squares debiasing after SPGL1-BPDN.

The algorithm deliberately reuses LION's existing debias_ls routine
from LION/operators/DebiasOp.py. This is the same support thresholding
and gradient-descent least-squares step used after the legacy SPGL1 BP solve.

Debiasing is a POST-PROCESSING step. It is not part of the BPDN constrained
L1 optimisation, and it may change the coefficient L1 norm or feasibility.
The returned diagnostics measure the residual before and after the step.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from LION.operators.DebiasOp import debias_ls
from LION.operators.Operator import Operator


@dataclass(frozen=True, kw_only=True)
class BPDNDebiasDiagnostics:
    """Measured effects of support-restricted least-squares debiasing.

    Attributes:
        applied: Whether the least-squares step was called (nonempty support).
        support_size: Number of coefficients with abs(w) > support_tol.
        residual_before: Norm of A w - y before least-squares debiasing.
        residual_after: Norm of A w - y after least-squares debiasing.
        l1_before: Coefficient L1 norm before least-squares debiasing.
        l1_after: Coefficient L1 norm after least-squares debiasing.
    """

    applied: bool
    support_size: int
    residual_before: float
    residual_after: float
    l1_before: float
    l1_after: float

    def to_dict(self) -> dict[str, bool | int | float]:
        """Convert the diagnostics to JSON-compatible values."""
        return asdict(self)


@torch.no_grad()
def debias_bpdn_coefficients(
    op: Operator,
    measurement: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    factor: float = 1.0,
    support_tol: float = 1e-5,
    max_iter: int = 10,
    tol: float = 1e-7,
    step_size: float | None = None,
) -> tuple[torch.Tensor, BPDNDebiasDiagnostics]:
    r"""Refit wavelet coefficients by least squares on their BPDN support.

    Let S = {j : |factor*w_j| > support_tol}. The method starts from the
    BPDN coefficients w, fixes all coefficients outside S to zero, and
    approximately solves

        min_{v supported on S} 0.5 * ||A v - y||_2**2.

    This exactly follows the legacy BP's debiasing scaling convention:
    the support threshold is applied in SPGL1-scaled coefficient units.
    debias_ls receives factor*y and factor*w, and the returned
    coefficients are then divided by factor.

    Args:
        op: Combined measurement and wavelet synthesis operator.
        measurement: One 1D differential measurement vector in model units.
        coefficients: One 1D BPDN wavelet-coefficient vector in model units.
        factor: The BPDN numeric scaling multiplier, positive.
        support_tol: Strict support threshold on absolute scaled coefficients.
        max_iter: Number of legacy gradient-descent LS steps at most.
        tol: Legacy debias_ls relative coefficient-change stopping tolerance.
        step_size: Known safe gradient step. None uses legacy power iteration.

    Returns:
        Debiased coefficients, plus before/after residual and L1 diagnostics.

    Important:
        The result is generally NOT the constrained-L1 BPDN optimiser.
        The LS step can increase the L1 norm. An insufficiently small
        support may also leave the final residual above BPDN's epsilon.
        Therefore the caller must retain the original BPDN diagnostics
        and check the post-processing residual separately.
    """
    if measurement.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("Debiasing expects 1D measurements and coefficients.")
    if not torch.isfinite(measurement).all() or not torch.isfinite(coefficients).all():
        raise ValueError("Debiasing requires finite measurements and coefficients.")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Debias factor must be finite and positive.")
    if not math.isfinite(support_tol) or support_tol < 0:
        raise ValueError("Debias support_tol must be finite and nonnegative.")
    if max_iter < 1:
        raise ValueError("Debias max_iter must be >= 1.")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("Debias tol must be finite and positive.")

    residual_before = float(torch.linalg.vector_norm(op(coefficients) - measurement))
    l1_before = float(torch.linalg.vector_norm(coefficients, ord=1))
    scaled_coefficients = coefficients * factor
    scaled_measurement = measurement * factor
    support_size = int(torch.count_nonzero(scaled_coefficients.abs() > support_tol))

    if support_size == 0:
        # No selected coefficients: the existing debias_ls also returns
        # the input unchanged in this case.
        result = coefficients.clone()
    else:
        scaled_result = debias_ls(
            op=op,
            y=scaled_measurement,
            w=scaled_coefficients,
            support_tol=support_tol,
            max_iter=max_iter,
            tol=tol,
            step_size=step_size,
        )
        result = scaled_result / factor

    residual_after = float(torch.linalg.vector_norm(op(result) - measurement))
    l1_after = float(torch.linalg.vector_norm(result, ord=1))
    return result, BPDNDebiasDiagnostics(
        applied=support_size > 0,
        support_size=support_size,
        residual_before=residual_before,
        residual_after=residual_after,
        l1_before=l1_before,
        l1_after=l1_after,
    )
