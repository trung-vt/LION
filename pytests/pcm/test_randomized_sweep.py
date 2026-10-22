"""Small CPU checks for the bounded randomized-noise experiment runners."""

from types import SimpleNamespace

import numpy as np
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.randomized.sweep import parameter_grid, solve
from LION.pcm.randomized.selection import ssdu_validation_score
from LION.reconstructors.GSPnP import GSDenoiser


class IdentityDenoiser(torch.nn.Module):
    def forward(self, x, sigma=None):
        return x


class HalfDenoiser(torch.nn.Module):
    def forward(self, x, sigma=None):
        return 0.5 * x


def test_admm_identity_denoiser_recovers_full_data():
    op = PhotocurrentMapOp(j_order=4, normalized=True, device="cpu")
    image = torch.rand(1, 1, 16, 16)
    result, trace = solve(
        op,
        op(image),
        IdentityDenoiser(),
        {"algorithm": "pnp_admm", "rho": 0.1, "sigma": 0.2},
        12,
    )
    torch.testing.assert_close(result, image, atol=2e-6, rtol=1e-5)
    assert len(trace) == 12
    assert all(t["seconds"] >= 0 for t in trace)


def test_gs_objective_accepted_iterations_descend():
    op = PhotocurrentMapOp(
        j_order=4, s_indices=np.arange(50), normalized=True, device="cpu"
    )
    image = torch.rand(1, 1, 16, 16)
    model = GSDenoiser(HalfDenoiser())
    result, trace = solve(
        op,
        op(image),
        model,
        {"algorithm": "gs_proximal_gradient", "lambda": 1.0, "tau": 2.0, "sigma": 0.2},
        8,
    )
    objectives = [t["objective"] for t in trace]
    assert all(b <= a + 1e-5 for a, b in zip(objectives, objectives[1:]))
    assert torch.isfinite(result).all()


def test_dwt_sigma_is_not_a_spurious_tuning_axis():
    for model in ("dwtdnet", "gs_dwtdnet"):
        assert len({row["sigma"] for row in parameter_grid(model)}) == 1
    assert len({row["sigma"] for row in parameter_grid("drunet")}) == 2


def test_ssdu_selection_uses_prediction_of_noisy_measurements(monkeypatch):
    import LION.pcm.randomized.selection as selection

    calls = []

    class Model:
        def eval(self):
            return self

    class Corpus:
        def fixed_cases(self, split, per_domain):
            assert split == "validation"
            return [(torch.ones(1, 1, 16, 16), {})]

    monkeypatch.setattr(
        selection, "randomized_acquisition", lambda *args, **kwargs: (object(), {})
    )
    monkeypatch.setattr(
        selection,
        "split_ssdu",
        lambda *args: (
            object(),
            torch.zeros(1),
            lambda prediction: prediction,
            torch.full((1,), 2.0),
            {},
        ),
    )

    def reconstruction(model, op, y):
        calls.append(y)
        return torch.zeros(1)

    monkeypatch.setattr(selection, "regularized_reconstruction", reconstruction)
    score = ssdu_validation_score(
        Model(), SimpleNamespace(device="cpu", validation_per_domain=1), Corpus()
    )
    assert score == 4.0  # (predicted coefficient 0 - observed heldout coefficient 2)^2
    assert len(calls) == 9
