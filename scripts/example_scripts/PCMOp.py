"""Class for Symmetrized Gradient Operator."""

import torch

from mrpro.operators.LinearOperator import LinearOperator

from LION.operators import PhotocurrentMapOp


class PCMOp(LinearOperator):
    def __init__(self, J: int, subsampler, wht_dim=-1) -> None:
        super().__init__()
        self.pcm_op = PhotocurrentMapOp(
            J=J, subsampler=subsampler, wht_dim=wht_dim
        )  # Dummy operator to get device

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        w = self.pcm_op.forward(v)
        return (w,)

    def adjoint(self, w: torch.Tensor) -> tuple[torch.Tensor,]:
        v = self.pcm_op.adjoint(w)
        return (v,)
