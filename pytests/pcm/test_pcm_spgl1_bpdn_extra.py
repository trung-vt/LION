"""Extra SPGL1-BPDN tests: known answers, optimality checks, corner cases and runner wiring.

Solver status codes come from ``spgl1.spgl1`` (1 root found, 4 optimal, 5 iteration limit).
"""

from __future__ import annotations

import json
import math
import warnings
from copy import deepcopy

import numpy as np
import pytest
import torch

from LION.operators.CompositeOp import CompositeOp
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.spgl1_bpdn import (
    SPGL1BPDNParameter,
    spgl1_bpdn_reconstruction,
    spgl1_bpdn_torch,
)
from LION.pcm.spgl1_bpdn_debias import debias_bpdn_coefficients

EXIT_ROOT_FOUND, EXIT_OPTIMAL, EXIT_ITERATIONS = 1, 4, 5


class IdentityOp:
    """Identity map with forward and adjoint."""

    def __call__(self, v):
        return v

    def adjoint(self, v):
        return v


def pcm_problem(noise=0.1, wavelet="db4", seed=0):
    """32x32 PCM problem with 300 measurements; returns op, wavelet, B, x, y, e."""
    torch.manual_seed(seed)
    op = PhotocurrentMapOp(
        j_order=5, s_indices=np.arange(300), normalized=True, device="cpu"
    )
    wav = Wavelet2D((32, 32), wavelet_name=wavelet, device="cpu")
    x = torch.rand(32, 32)
    e = noise * torch.randn(300)
    return op, wav, CompositeOp(wav, op, device="cpu"), x, op(x) + e, e


def duality_gap(B, y, w, eps):
    """Relative BPDN duality gap; dual is max <y,u> - eps*||u|| s.t. ||B^T u||_inf <= 1."""
    r = y - B(w)
    u = r / B.adjoint(r).abs().max()
    primal = w.abs().sum()
    return float((primal - (y @ u - eps * u.norm())) / primal)


@pytest.mark.parametrize("factor", [10.0, 100.0])
def test_identity_operator_matches_soft_threshold(factor):
    y = torch.tensor([1.0, -0.5, 0.03, 0.0])
    eps = 0.12
    lam = math.sqrt(
        (eps**2 - 0.03**2) / 2
    )  # soft-threshold level whose residual is eps
    expected = torch.sign(y) * torch.clamp(y.abs() - lam, min=0)
    w, info = spgl1_bpdn_torch(IdentityOp(), y, epsilon=eps, factor=factor, iters=200)
    assert info.solver_status == EXIT_ROOT_FOUND
    torch.testing.assert_close(w, expected, atol=1e-5, rtol=0)


def test_epsilon_above_data_norm_gives_zero_image_even_with_debias():
    op, _, _, _, y, _ = pcm_problem()
    cfg = SPGL1BPDNParameter(
        epsilon=2 * float(y.norm()), wavelet_name="db4", debias=True
    )
    image, (report,) = spgl1_bpdn_reconstruction(op, y.reshape(1, 1, -1), cfg)
    assert torch.count_nonzero(image) == 0
    assert report.solver_status == EXIT_OPTIMAL
    assert not report.debias_applied


def test_iteration_limit_is_reported_and_warned():
    _, _, B, _, y, e = pcm_problem()
    with pytest.warns(RuntimeWarning, match="exceeds epsilon"):
        _, info = spgl1_bpdn_torch(B, y, epsilon=1.2 * float(e.norm()), iters=1)
    assert info.solver_status == EXIT_ITERATIONS
    assert not info.feasible_with_tolerance


def test_solution_is_feasible_sparser_than_truth_and_near_optimal():
    _, wav, B, x, y, e = pcm_problem()
    eps = 1.2 * float(e.norm())  # the true image is feasible
    w, info = spgl1_bpdn_torch(B, y, epsilon=eps, iters=2000, opt_tol=1e-6)
    assert info.solver_status == EXIT_ROOT_FOUND
    assert info.residual_norm == pytest.approx(eps, rel=1e-3)  # constraint is active
    assert float(w.abs().sum()) <= float(wav(x).abs().sum())
    assert duality_gap(B, y, w, eps) < 1e-2


def test_l1_norm_decreases_as_epsilon_grows():
    _, _, B, _, y, e = pcm_problem()
    l1_norms = []
    for scale in (0.8, 1.0, 1.5, 3.0):
        eps = scale * float(e.norm())
        w, info = spgl1_bpdn_torch(B, y, epsilon=eps, iters=2000, opt_tol=1e-6)
        assert info.residual_norm == pytest.approx(eps, rel=1e-3)
        l1_norms.append(float(w.abs().sum()))
    assert l1_norms == sorted(l1_norms, reverse=True)


@pytest.mark.parametrize("c", [0.5, 4.0])
def test_scaling_data_and_epsilon_scales_the_solution(c):
    _, _, B, _, y, e = pcm_problem()
    eps = 1.2 * float(e.norm())
    w1, _ = spgl1_bpdn_torch(B, y, epsilon=eps, iters=2000, opt_tol=1e-6)
    w2, _ = spgl1_bpdn_torch(B, c * y, epsilon=c * eps, iters=2000, opt_tol=1e-6)
    assert float((w2 - c * w1).norm() / (c * w1).norm()) < 1e-3


def test_batch_items_are_solved_independently():
    op, _, _, _, _, _ = pcm_problem()
    torch.manual_seed(3)
    y = op(torch.rand(3, 1, 32, 32)) + 0.1 * torch.randn(3, 1, 300)
    cfg = SPGL1BPDNParameter(epsilon=2.0, wavelet_name="db4")
    batch, _ = spgl1_bpdn_reconstruction(op, y, cfg)
    single, _ = spgl1_bpdn_reconstruction(op, y[1:2], cfg)
    torch.testing.assert_close(batch[1:2], single)


@pytest.mark.parametrize("wavelet", ["db1", "db4"])
def test_composite_operator_is_adjoint_and_has_orthonormal_rows(wavelet):
    _, wav, B, _, _, _ = pcm_problem(wavelet=wavelet)
    torch.manual_seed(5)
    w, r = torch.randn(wav.size), torch.randn(300)
    assert float(B(w) @ r) == pytest.approx(float(w @ B.adjoint(r)), rel=1e-4)
    torch.testing.assert_close(B(B.adjoint(r)), r, atol=1e-5, rtol=1e-5)


def test_debias_only_changes_coefficients_on_the_support():
    _, _, B, _, y, e = pcm_problem()
    w, _ = spgl1_bpdn_torch(B, y, epsilon=1.2 * float(e.norm()), iters=2000)
    w_debiased, info = debias_bpdn_coefficients(
        B, y, w, support_tol=1e-5, max_iter=10, tol=1e-7
    )
    assert torch.count_nonzero(w_debiased[w.abs() <= 1e-5]) == 0
    assert info.residual_after <= info.residual_before


def test_converged_small_epsilon_solution_is_not_flagged():
    _, _, B, _, y, _ = pcm_problem(noise=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, info = spgl1_bpdn_torch(B, y, epsilon=0.01, iters=2000)
    assert info.solver_status == EXIT_ROOT_FOUND


def test_runner_writes_solver_diagnostics_per_trial(tmp_path):
    from LION.data_loaders.pcm.data_primitives import PCMSamplingParameter
    from LION.experiments.pcm_experiments import PCMLogParameter
    from LION.experiments.pcm_presets.presets import PRESETS
    from LION.pcm.noise import PCMNoiseSimulation
    from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment

    np.save(
        tmp_path / "img.npy",
        np.random.default_rng(0).random((32, 32)).astype(np.float32),
    )
    cfg = deepcopy(PRESETS["cigs_example_256x256"])
    cfg.data.root, cfg.data.j_order = tmp_path, 5
    cfg.this_sample.data_filename = cfg.this_sample.ground_truth_filename = "img.npy"
    cfg.noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / 3.5e-8, j_order=5
    )
    cfg.sampling = PCMSamplingParameter(
        randomising_scheme="multilevel", sampling_ratios=[0.3], coarse_j_values=[3]
    )
    cfg.methods = deepcopy(PRESETS["synthetic_png_noisy_bpdn"].methods)
    cfg.runtime.root_output_dir, cfg.runtime.device, cfg.runtime.num_trials = (
        tmp_path / "out",
        "cpu",
        2,
    )
    cfg.log = PCMLogParameter(save_metrics=True, save_config=True)

    exp_dir = run_single_sample_experiment(
        cfg, sample_name="t", show_progress_bars=False
    )

    files = sorted(exp_dir.rglob("solver_diagnostics.json"))
    assert len(files) == 4  # 2 trials x 2 BPDN methods
    report = json.loads(files[0].read_text())[0]
    assert {
        "solver_status",
        "epsilon",
        "residual_norm",
        "debias_enabled",
    } <= report.keys()
    assert (exp_dir / "spgl1_bpdn" / "results.csv").exists()
    assert "spgl1_bpdn_debiased" in (exp_dir / "config.yaml").read_text()
