"""Fully symmetrised third-order derivative of a symmetric rank-two field.

The expected input tensor is batch first:
    (batch, 2, 2, channel, height, width).
The FiniteDifferenceOp adds the derivative index before the batch index.
The returned tensor therefore has axes
    (derivative, batch, 2, 2, channel, height, width).

The input field is assumed to be symmetric in its two tensor axes. For a
symmetric w, E2(w)[i,j,k] = (D_i w[j,k] + D_j w[i,k] + D_k w[i,j])/3.

This operator uses MRPro composition so that the exact discrete adjoint
automatically includes the selected finite-difference boundary conditions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from mrpro.operators import IdentityOp, LinearOperator, RearrangeOp

from LION.operators.FiniteDifferenceOp import FiniteDifferenceOp, PadModeIn


class SymmetrizedSecondOrderGradientOp(LinearOperator):
    """Map a symmetric rank-two tensor to a full symmetric rank-three tensor.

    Tensors use full 2x2 and 2x2x2 representations, including multiplicities
    of off-diagonal entries. No compressed-component convention is used.
    """

    def __init__(
        self,
        dim: Sequence[int] = (-2, -1),
        mode: Literal["central", "forward", "backward"] = "backward",
        pad_mode: PadModeIn | Sequence[PadModeIn] = "zeros",
    ) -> None:
        super().__init__()
        if len(dim) != 2:
            raise ValueError("Third-order 2-D PCM TGV requires two spatial dimensions.")
        deriv = FiniteDifferenceOp(dim=dim, mode=mode, pad_mode=pad_mode)
        # FiniteDifferenceOp(w) shape: (derivative, batch, i, j, ...).
        # Symmetrisation of the derivative index over the three tensor axes.
        interchange_d_i = RearrangeOp("d b i j ... -> i b d j ...")
        interchange_d_j = RearrangeOp("d b i j ... -> j b i d ...")
        self._operator = (
            (1.0 / 3.0) * (IdentityOp() + interchange_d_i + interchange_d_j)
        ) @ deriv

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor]:
        """Compute E2(w); w must have full symmetric 2x2 tensor axes."""
        if w.ndim < 6 or w.shape[1:3] != (2, 2):
            raise ValueError(
                "Expected a rank-two field with shape "
                "(batch, 2, 2, channel, height, width)."
            )
        return self._operator(w)

    def adjoint(self, tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the Euclidean/Frobenius adjoint of the full-tensor operator."""
        if tensor.ndim < 7 or tensor.shape[0] != 2 or tensor.shape[2:4] != (2, 2):
            raise ValueError(
                "Expected rank-three gradient with shape "
                "(2, batch, 2, 2, channel, height, width)."
            )
        return self._operator.H(tensor)
