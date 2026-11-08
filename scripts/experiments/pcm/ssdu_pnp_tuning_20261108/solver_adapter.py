"""Unsupervised PnP adapters matching the established PCM numerical implementations.

Both accept ONLY normalized PCM operator and a noisy measurement tensor.
No GT and no test-set data are accepted.
"""
from __future__ import annotations

from copy import deepcopy
import time

import torch

from LION.experiments.pcm_presets.presets import QUICK_METHODS
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.reconstructors.GSPnP import GSDenoiser, PCMDeepInvPhysics
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import live_admm
from scripts.experiments.pcm.gs_pnp_gs_drunet_tuned_20261104.step0_step1_paper_ablation_20261104 import (
    ObservedGS,
)


def make_model(method: str, device: torch.device) -> torch.nn.Module:
    if method == "admm":
        config = deepcopy(QUICK_METHODS["pnp_admm_drunet"].denoiser)
    elif method == "gs":
        config = deepcopy(QUICK_METHODS["pnp_gs_deepinv_drunet"].denoiser)
    else:
        raise ValueError("Invalid PnP method")
    model = build_grayscale_denoiser(config, device)
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def reconstruct_admm(
    op: PhotocurrentMapOp,
    observation: torch.Tensor,
    denoiser: torch.nn.Module,
    candidate: dict,
) -> tuple[torch.Tensor, dict]:
    assert op.w.sqrt_n is not None
    result = live_admm(
        op,
        observation,
        denoiser,
        rho=float(candidate["rho"]),
        sigma=float(candidate["sigma"]),
    )
    x = result["x"]
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite ADMM final x")
    detail = {k: v for k, v in result.items() if k not in ("x", "snapshots", "trace")}
    detail["output"] = "ADMM final x (matches historical oracle reference)"
    return x.detach(), detail


@torch.no_grad()
def reconstruct_gs(
    op: PhotocurrentMapOp,
    observation: torch.Tensor,
    denoiser: GSDenoiser,
    candidate: dict,
) -> tuple[torch.Tensor, dict]:
    """Reproduce exactly the previous frozen Step6 GS-PnP paper-final-z equations."""
    assert op.w.sqrt_n is not None
    assert isinstance(denoiser, GSDenoiser)
    if observation.ndim != 3 or observation.shape[:2] != (1, 1):
        raise ValueError("Expected normalized PCM (1,1,M)")
    cfg = deepcopy(QUICK_METHODS["pnp_gs_deepinv_drunet"])
    cfg.lambd = float(candidate["lambda"])
    cfg.sigma = float(candidate["sigma"])
    cfg.stepsize = float(candidate["tau0"])
    cfg.iters = int(candidate["max_accepted_iterations"])
    cfg.max_backtracking_attempts = int(candidate["max_backtracking_attempts"])
    cfg.gamma_backtracking = float(candidate["gamma_backtracking"])
    cfg.eta_backtracking = float(candidate["eta_backtracking"])
    cfg.early_stop = True
    cfg.eps = float(candidate["stopping_eps"])
    cfg.continue_after_early_stop = False
    assert candidate["output"] == "paper_final_z"
    assert candidate["stopping_policy"] == "deepinv_relative_objective"

    engine = ObservedGS(PCMDeepInvPhysics(op), denoiser, "adjoint")
    visited = [0]
    lasttau = [None]
    events = []

    def accepted(k, x):
        visited[0] = k
        r = engine.last_accepted
        if r is None:
            raise RuntimeError("Missing last accepted GS line-search state")
        lasttau[0] = float(r["stepsize"])

    def on_exhausted(k, x, attempted, rejected, tau):
        events.append({"accepted": k, "rejected": rejected, "step_size": float(tau)})

    if observation.device.type == "cuda":
        torch.cuda.synchronize(observation.device)
    start = time.perf_counter()
    native = engine.reconstruct_sample(
        observation,
        config=cfg,
        iteration_callback=accepted,
        backtracking_stop_callback=on_exhausted,
    )
    if visited[0] < 1 or lasttau[0] is None:
        raise RuntimeError("No GS iteration accepted")
    grad, _ = denoiser.grad(native, cfg.sigma, set_reset_grad=True)
    paper_z = native - cfg.lambd * lasttau[0] * grad
    if observation.device.type == "cuda":
        torch.cuda.synchronize(observation.device)
    if not torch.isfinite(paper_z).all():
        raise FloatingPointError("Nonfinite GS paper-final-z")
    detail = {
        "output": "paper_final_z",
        "accepted_iterations": visited[0],
        "last_tau": lasttau[0],
        "lambda_tau": cfg.lambd * lasttau[0],
        "termination": (
            "backtracking_exhausted"
            if events
            else ("iteration_cap" if visited[0] >= cfg.iters else "deepinv_objective")
        ),
        "rejections": sum(not r["accepted"] for r in engine.attempts),
        "backtracking_exhausted": bool(events),
        "wall_seconds_with_final_gradient": time.perf_counter() - start,
        "uses_ground_truth": False,
    }
    return paper_z.detach(), detail


def reconstruct(
    method: str,
    op: PhotocurrentMapOp,
    observation: torch.Tensor,
    denoiser: torch.nn.Module,
    candidate: dict,
) -> tuple[torch.Tensor, dict]:
    if method == "admm":
        return reconstruct_admm(op, observation, denoiser, candidate)
    if method == "gs":
        return reconstruct_gs(op, observation, denoiser, candidate)
    raise ValueError("Unknown method")
