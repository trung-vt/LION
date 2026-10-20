"""Small CUDA smoke tests for the PCM reconstructor integration."""

import numpy as np
import pytest
import torch

from LION.experiments.config.other_config import NormParameter
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import PCMProxParameter
from LION.reconstructors.BasePnP import DenoiserParameter
from LION.reconstructors.GSPnP import GSDenoiser, GSPnP, GSPnPParameter
from LION.reconstructors.PnPADMM import PnPADMM, PnPADMMParameter


class ScaleNetwork(torch.nn.Module):
    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        return 0.8 * x


def identity_denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x


@pytest.mark.cuda
@pytest.mark.parametrize("gpu_index", [0, 1])
def test_pcm_admm_and_gspnp_on_gpu(gpu_index: int):
    if torch.cuda.device_count() <= gpu_index:
        pytest.skip(f"CUDA device {gpu_index} unavailable")
    device = torch.device(f"cuda:{gpu_index}")
    torch.manual_seed(7)

    op = PhotocurrentMapOp(
        j_order=4,
        s_indices=np.arange(64),
        device=device,
    )
    x_true = torch.rand(2, *op.domain_shape, device=device)
    measurements = op(x_true) + 0.01 * torch.randn(2, *op.range_shape, device=device)

    admm = PnPADMM(physics=op, denoiser=identity_denoiser)
    admm_config = PnPADMMParameter(
        denoiser=DenoiserParameter(name="identity"),
        iters=5,
        prox=PCMProxParameter(rho=0.1),
        sigma=0.05,
    )
    admm_recon = admm.admm_algorithm(measurements, admm_config)

    gs = GSPnP(physics=op, denoiser=GSDenoiser(ScaleNetwork()).to(device))
    gs_config = GSPnPParameter(
        denoiser=DenoiserParameter(name="scaled"),
        iters=5,
        lambd=0.01,
        stepsize=0.5,
        eps=0.0,
        use_backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.7,
        sigma=0.05,
        data_fidelity=NormParameter(normalize=False),
        data_fidelity_prox=PCMProxParameter(rho=1.0),
    )
    gs_recon = gs.reconstruct_sample(measurements, config=gs_config)

    for name, recon in [("PnP-ADMM", admm_recon), ("GS-PnP", gs_recon)]:
        assert recon.shape == x_true.shape, name
        assert recon.device == device, name
        assert torch.isfinite(recon).all(), name
        assert torch.linalg.norm(op(recon) - measurements) < torch.linalg.norm(
            measurements
        ), name
