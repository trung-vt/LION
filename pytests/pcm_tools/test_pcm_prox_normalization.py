import numpy as np
import pytest
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import (
    PCMProxParameter,
    closed_form_pcm_prox,
    closed_form_pcm_prox_with_precomputed_terms,
    precompute_pcm_prox_terms,
)
from LION.reconstructors.GSPnP import GSPnP
from LION.reconstructors.PnPADMM import PnPADMM


def _dense_pcm_matrix(op: PhotocurrentMapOp) -> torch.Tensor:
    eye = torch.eye(op.num_pixels, dtype=torch.float64)
    return torch.stack(
        [op(op.unvectorize(row)) for row in eye],
        dim=1,
    )


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("rho", [0.01, 0.7, 10.0])
def test_pcm_prox_raw_and_normalized_match_dense_solve(
    normalized: bool, rho: float
) -> None:
    op = PhotocurrentMapOp(
        j_order=2,
        s_indices=np.array([0, 1, 4, 7, 10, 13]),
        normalized=normalized,
        device="cpu",
    )
    torch.manual_seed(21)
    z = torch.randn(2, 4, 4, dtype=torch.float64)
    y = torch.randn(2, 6, dtype=torch.float64)

    matrix = _dense_pcm_matrix(op)
    lhs = matrix.T @ matrix + rho * torch.eye(op.num_pixels, dtype=torch.float64)
    rhs = y @ matrix + rho * op.vectorize(z)
    expected = op.unvectorize(torch.linalg.solve(lhs, rhs.T).T)

    direct = closed_form_pcm_prox(op, rho=rho, y=y, z=z)
    st_y, m = precompute_pcm_prox_terms(op, y)
    transform_rho = rho if normalized else rho / op.num_pixels
    precomputed = closed_form_pcm_prox_with_precomputed_terms(
        op,
        z,
        rho=rho,
        ST_y=st_y,
        m_plus_rho=m + transform_rho,
    )

    torch.testing.assert_close(direct, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(precomputed, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(precomputed, direct, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("normalized", [True, False])
def test_pcm_prox_reconstructors_use_correct_precomputed_scale(
    normalized: bool,
) -> None:
    op = PhotocurrentMapOp(
        j_order=2,
        s_indices=np.array([0, 1, 4, 7, 10, 13]),
        normalized=normalized,
        device="cpu",
    )
    rho = 0.01
    torch.manual_seed(32)
    y = torch.randn(6, dtype=torch.float64)
    z = torch.randn(4, 4, dtype=torch.float64)
    expected = closed_form_pcm_prox(op, rho=rho, y=y, z=z)

    admm = PnPADMM(physics=op, denoiser=lambda x, sigma: x)
    from_admm = admm.prep_prox(PCMProxParameter(rho=rho), y)(z)
    gs = GSPnP(physics=op, denoiser=torch.nn.Identity())
    from_gs = gs.prep_data_fidelity_prox(PCMProxParameter(), y, measurement_dims=(-1,))(
        z, 1.0 / rho
    )

    torch.testing.assert_close(from_admm, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(from_gs, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("normalized", [True, False])
def test_pcm_prox_256x256_optimality(
    normalized: bool,
) -> None:
    op = PhotocurrentMapOp(
        j_order=8,
        s_indices=np.arange(0, 65536, 5, dtype=np.int64),
        normalized=normalized,
        device="cpu",
    )
    torch.manual_seed(43)
    z = torch.randn(256, 256, dtype=torch.float64)
    y = torch.randn(op.range_shape[0], dtype=torch.float64)
    rho = 0.01
    st_y, m = precompute_pcm_prox_terms(op, y)
    transform_rho = rho if normalized else rho / op.num_pixels
    x = closed_form_pcm_prox_with_precomputed_terms(
        op,
        z,
        rho=rho,
        ST_y=st_y,
        m_plus_rho=m + transform_rho,
    )

    gradient = op.adjoint(op(x) - y) + rho * (x - z)
    assert float(gradient.abs().max()) < 1e-8
