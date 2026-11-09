"""Third-order TGV model reusing the existing LION block PDHG solver."""

from __future__ import annotations
from collections.abc import Sequence
from numbers import Real

import torch
from mrpro.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
    RearrangeOp,
    ZeroOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

from LION.models.BasePdhgNet import BasePdhgNet
from LION.operators.FiniteDifferenceOp import FiniteDifferenceOp, PadModeIn
from LION.operators.SymmetrizedGradientOp import SymmetrizedGradientOp
from LION.operators.SymmetrizedSecondOrderGradientOp import (
    SymmetrizedSecondOrderGradientOp,
)


class AdaptiveTGV3Network(BasePdhgNet):
    """TGV-3 with three primal fields and four dual terms.

    Componentwise L1 on full tensors includes multiplicities of
    off-diagonal components. This is an anisotropic discrete TGV-3.
    """

    def init_ops(self, fd_pad_mode: PadModeIn | Sequence[PadModeIn]) -> None:
        if len(self.dim) != 2:
            raise ValueError("TGV-3 currently supports only 2-D images.")
        zero = ZeroOp()
        self.zero_op = zero
        self._g = ProximableFunctionalSeparableSum(*(ZeroFunctional(),) * 3)
        to_batch = RearrangeOp("grad batch ... -> batch grad ...")
        grad = to_batch @ FiniteDifferenceOp(
            dim=self.dim,
            mode="forward",
            pad_mode=fd_pad_mode,
        )
        e1 = to_batch @ SymmetrizedGradientOp(
            dim=self.dim,
            transpose_dim=2,
            mode="backward",
            pad_mode=fd_pad_mode,
        )
        e2 = to_batch @ SymmetrizedSecondOrderGradientOp(
            dim=self.dim,
            mode="backward",
            pad_mode=fd_pad_mode,
        )
        self.grad_term_row = (grad, -1.0 * IdentityOp(), zero)
        self.sym_grad_term_row = (zero, e1, -1.0 * IdentityOp())
        self.third_grad_term_row = (zero, zero, e2)

    @property
    def g(self) -> ProximableFunctionalSeparableSum | ProximableFunctional:
        return self._g

    @staticmethod
    def split_lambda(
        lambda_reg: torch.Tensor | tuple[float, float, float],
    ) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
        """Return alpha2, alpha1 and alpha0 with block-broadcastable shapes."""
        if isinstance(lambda_reg, tuple):
            if len(lambda_reg) != 3 or not all(
                isinstance(a, Real) and 0 <= a < float("inf") for a in lambda_reg
            ):
                raise ValueError("Expected three finite nonnegative scalar weights.")
            return lambda_reg

        if lambda_reg.ndim != 4 or lambda_reg.shape[1] != 3:
            raise ValueError("TGV-3 maps must have shape (batch, 3, height, width).")
        if not bool(torch.isfinite(lambda_reg).all()) or bool((lambda_reg < 0).any()):
            raise ValueError("All TGV-3 maps must be finite and nonnegative.")
        # [B,2,C,H,W], [B,2,2,C,H,W], [B,2,2,2,C,H,W]
        alpha2 = lambda_reg[:, 0:1, None, :, :]
        alpha1 = lambda_reg[:, 1:2, None, None, :, :]
        alpha0 = lambda_reg[:, 2:3, None, None, None, :, :]
        return alpha2, alpha1, alpha0

    def make_f(
        self,
        measurement: torch.Tensor,
        *,
        lambda_reg: torch.Tensor | tuple[float, float, float],
        l2_weight: torch.Tensor | None = None,
    ) -> ProximableFunctionalSeparableSum:
        if l2_weight is None:
            l2_weight = measurement.new_ones(())
        fidelity = 0.5 * L2NormSquared(target=measurement, weight=l2_weight)
        alpha2, alpha1, alpha0 = self.split_lambda(lambda_reg)
        return ProximableFunctionalSeparableSum(
            fidelity,
            L1NormViewAsReal(weight=alpha2),
            L1NormViewAsReal(weight=alpha1),
            L1NormViewAsReal(weight=alpha0),
        )

    def make_initial_values(
        self, initial_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if initial_image.ndim != 4:
            raise ValueError("Expected (batch, channel, height, width) image.")
        n = len(self.dim)
        b = initial_image.shape[0]
        field = initial_image.new_zeros((b, n, *initial_image.shape[1:]))
        tensor = initial_image.new_zeros((b, n, n, *initial_image.shape[1:]))
        return initial_image, field, tensor

    def make_stacked_operator(self, forward_op: LinearOperator) -> LinearOperatorMatrix:
        zero = self.zero_op
        return LinearOperatorMatrix(
            (
                (forward_op, zero, zero),
                self.grad_term_row,
                self.sym_grad_term_row,
                self.third_grad_term_row,
            )
        )
