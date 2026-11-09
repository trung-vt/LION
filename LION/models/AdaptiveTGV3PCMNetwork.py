"""PCM third-order TGV using the shared MRPro-backed LION PDHG."""

from __future__ import annotations
import torch
from mrpro.operators import LinearOperator

from LION.models.AdaptiveTGV3Network import AdaptiveTGV3Network


class AdaptiveTGV3PCMNetwork(AdaptiveTGV3Network):
    """2-D PCM TGV-3 with three scalar weights or three spatial maps."""

    def __init__(
        self, config, lambda_map_network=None, forward_op=None, example_image=None
    ):
        if config.pdhg.regularisation != "TGV3":
            raise ValueError("TGV-3 requires config.pdhg.regularisation='TGV3'.")
        super().__init__(config, lambda_map_network, forward_op, example_image)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def forward(
        self,
        *,
        input_image: torch.Tensor | None,
        initial_image: torch.Tensor,
        measurement: torch.Tensor,
        key: int,
        forward_op: LinearOperator,
        lambda_reg: torch.Tensor | tuple[float, float, float] | None = None,
        l2_weight: torch.Tensor | None = None,
        max_iterations: int | None = None,
        primal_stepsize: float | None = None,
        dual_stepsize: float | None = None,
        use_checkpoint: bool = False,
        checkpoint_chunk_size: int | None = None,
        checkpoint_use_reentrant: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        if self.backend not in ("new", "mrpro"):
            raise ValueError("TGV-3 PCM requires the MRPro-backed PDHG.")
        if not isinstance(forward_op, LinearOperator):
            raise TypeError("Expected the PCM MRPro LinearOperator wrapper.")
        return super().forward(
            input_image=input_image,
            initial_image=initial_image,
            measurement=measurement,
            key=key,
            forward_op=forward_op,
            lambda_reg=lambda_reg,
            l2_weight=l2_weight,
            max_iterations=max_iterations,
            primal_stepsize=primal_stepsize,
            dual_stepsize=dual_stepsize,
            use_checkpoint=use_checkpoint,
            checkpoint_chunk_size=checkpoint_chunk_size,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            use_tqdm=use_tqdm,
        )
