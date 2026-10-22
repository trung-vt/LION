"""Validation-only PnP parameter search using frozen pilot denoisers.

Ordinary denoisers use LION PnP-ADMM. Gradient-step denoisers use instrumented
proximal gradient with LION's potential and PCM proximal operator.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from LION.pcm.randomized.data import MixedImageCorpus, sha256, write_json
from LION.pcm.randomized.models import build_denoiser
from LION.pcm.randomized.noise import (
    NoiseRange,
    randomized_acquisition,
    sample_parameters,
    valid_coarse_levels,
)
from LION.pcm.randomized.runner import metrics, synchronize, write_csv
from LION.pcm_tools.pcm_prox import PCMProxParameter, closed_form_pcm_prox
from LION.reconstructors.PnPADMM import PnPADMM, PnPADMMParameter


@torch.no_grad()
def solve(
    op,
    y,
    network,
    config: dict,
    iterations: int,
    *,
    reference: torch.Tensor | None = None,
    snapshots: dict[str, np.ndarray] | None = None,
    snapshot_steps: tuple[int, ...] = (0, 1, 5, 10, 20),
) -> tuple[torch.Tensor, list[dict]]:
    device = y.device
    synchronize(device)
    start = time.perf_counter()
    trace = []
    initial = (
        torch.zeros_like(op.adjoint(y))
        if config["algorithm"] == "pnp_admm"
        else op.adjoint(y)
    )
    previous = initial

    def record(step: int, image: torch.Tensor, **details) -> None:
        nonlocal previous
        norm = float(torch.linalg.vector_norm(image))
        rel_change = (
            (
                float(torch.linalg.vector_norm(image - previous))
                / max(float(torch.linalg.vector_norm(previous)), 1e-12)
            )
            if step > 0
            else None
        )
        row = {
            "iteration": step,
            "seconds": time.perf_counter() - start,
            "data_residual_mse": float((op(image) - y).square().mean()),
            "relative_update_norm": rel_change,
            "image_norm": norm,
            "objective": details.get("objective"),
            "accepted_step": details.get("accepted_step"),
            "backtracks": details.get("backtracks", 0),
            "stop_reason": details.get("stop_reason", ""),
        }
        if reference is not None:
            row.update(
                metrics(image, reference)
            )  # Diagnostic only; not used for model selection.
        trace.append(row)
        if snapshots is not None and step in snapshot_steps:
            snapshots[str(step)] = image.detach().cpu().numpy()
        previous = image.detach().clone()

    if reference is not None or snapshots is not None:
        record(0, initial, objective=None)
    if config["algorithm"] == "pnp_admm":
        algorithm = PnPADMM(op, network)
        parameters = PnPADMMParameter(
            denoiser=None,
            iters=iterations,
            prox=PCMProxParameter(rho=config["rho"]),
            sigma=config["sigma"],
        )

        def callback(step, x):
            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite PnP-ADMM iterate")
            synchronize(device)
            record(step, x)

        result = algorithm.admm_algorithm(y, parameters, iteration_callback=callback)
    elif config["algorithm"] == "gs_proximal_gradient":
        weight, tau = config["lambda"], config["tau"]
        x = op.adjoint(y)
        gradient, potential = network.grad(x, config["sigma"], set_reset_grad=True)
        objective = float(0.5 * (op(x) - y).square().sum()) + weight * potential
        for step in range(1, iterations + 1):
            accepted, stagnated = False, False
            for backtracks in range(24):
                candidate = closed_form_pcm_prox(
                    op, rho=1.0 / tau, y=y, z=x - tau * weight * gradient
                )
                difference = float((candidate - x).square().sum())
                # Do not force Armijo comparisons below float32 image-change precision.
                if difference <= 1e-12 * max(1.0, float(x.square().sum())):
                    stagnated = True
                    break
                candidate_gradient, candidate_potential = network.grad(
                    candidate, config["sigma"], set_reset_grad=True
                )
                next_objective = (
                    float(0.5 * (op(candidate) - y).square().sum())
                    + weight * candidate_potential
                )
                tolerance = 1e-7 * max(1.0, abs(objective))
                if (
                    math.isfinite(next_objective)
                    and next_objective <= objective - 0.1 / tau * difference + tolerance
                ):
                    accepted = True
                    break
                tau *= 0.5
            if stagnated:
                synchronize(device)
                if not trace:
                    record(
                        0,
                        x,
                        objective=objective,
                        accepted_step=tau,
                        backtracks=backtracks,
                    )
                trace[-1].update(
                    stop_reason="numerical_stagnation",
                    seconds=time.perf_counter() - start,
                )
                break
            if not accepted:
                raise RuntimeError("GS objective line search failed after 24 attempts")
            x, gradient, objective = candidate, candidate_gradient, next_objective
            synchronize(device)
            record(
                step, x, objective=objective, accepted_step=tau, backtracks=backtracks
            )
        result = x
    else:
        raise ValueError(config["algorithm"])
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite PnP output")
    if not trace[-1]["stop_reason"]:
        trace[-1]["stop_reason"] = "iteration_budget"
    return result, trace


def parameter_grid(model: str) -> list[dict]:
    # DWTDNet ignores sigma; do not manufacture duplicate sigma-search results.
    sigmas = [0.05, 0.2] if "drunet" in model else [0.2]
    if model.startswith("gs_"):
        return [
            {
                "algorithm": "gs_proximal_gradient",
                "lambda": weight,
                "tau": tau,
                "sigma": sigma,
            }
            for weight, tau, sigma in itertools.product(
                [0.01, 0.1, 1.0], [0.5, 2.0], sigmas
            )
        ]
    return [
        {"algorithm": "pnp_admm", "rho": rho, "sigma": sigma}
        for rho, sigma in itertools.product([0.01, 0.1, 1.0, 10.0], sigmas)
    ]


def make_cases(
    corpus: MixedImageCorpus,
    split: str,
    ratio: float,
    dark: float,
    per_domain: int,
    device,
):
    cases = []
    for i, (image, record) in enumerate(corpus.fixed_cases(split, per_domain)):
        rng = np.random.default_rng(
            4_000_000 + i * 100 + (50_000 if split == "test" else 0)
        )
        clean = image.to(device)
        p = sample_parameters(rng, NoiseRange())
        p.dark_sigma, p.beam_sigma = dark, 6e-4
        count = int(np.ceil(ratio * corpus.side * corpus.side))
        coarse = min(4, max(valid_coarse_levels(corpus.side.bit_length() - 1, count)))
        acquisition, info = randomized_acquisition(
            clean, rng, NoiseRange(), ratio=ratio, coarse_j=coarse, parameters=p
        )
        cases.append((clean, record, acquisition, info))
    return cases


def run_sweep(
    manifest: Path,
    runs: Path,
    output: Path,
    models: list[str],
    *,
    device: str = "cuda:0",
    iterations: int = 20,
    per_domain: int = 2,
):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dev = torch.device(device)
    torch.set_num_threads(2)
    corpus = MixedImageCorpus(manifest, side=64)
    validation, test, chosen, failures, traces = [], [], [], [], []
    write_json(
        output / "protocol.json",
        {
            "models": models,
            "iterations": iterations,
            "per_domain": per_domain,
            "ratios": [0.01, 0.2, 1.0],
            "dark_sigmas": [0.05, 0.2],
            "beam_sigma": 6e-4,
            "seed_family": 4_000_000,
            "manifest_sha256": sha256(manifest),
            "selection": "lowest domain-balanced validation MSE per model/rate/noise condition; test never searched",
            "gs": "prox-gradient on 0.5||Ax-y||^2 + lambda*g(x), bounded monotone backtracking; last accepted x",
            "gs_relative_step_tolerance": 1e-6,
            "admm": "existing LION PnPADMM.admm_algorithm, zero initialization, exact normalized PCM proximal",
            "fairness": "identical source images, sampling indices and noise draws within each case",
            "status": "running",
            "device": device,
        },
    )
    for name in models:
        checkpoint_path = runs / f"{name}_mixture" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        network = build_denoiser(name, corpus.side, dev).eval()
        network.load_state_dict(checkpoint["model_state_dict"])
        for p in network.parameters():
            p.requires_grad_(False)
        grid = parameter_grid(name)
        for ratio, dark in itertools.product([0.01, 0.2, 1.0], [0.05, 0.2]):
            cases = make_cases(corpus, "validation", ratio, dark, per_domain, dev)
            scores = []
            for configuration_id, config in enumerate(grid):
                losses = []
                for case, (clean, record, acquisition, info) in enumerate(cases):
                    try:
                        reconstruction, trace = solve(
                            acquisition.op,
                            acquisition.measurement,
                            network,
                            config,
                            iterations,
                        )
                        row = {
                            "model": name,
                            "ratio": ratio,
                            "dark_sigma": dark,
                            "configuration_id": configuration_id,
                            "config": json.dumps(config, sort_keys=True),
                            "domain": record["domain"],
                            "case": case,
                            "source_hash": record["sha256"],
                            **metrics(reconstruction, clean),
                            "seconds": trace[-1]["seconds"],
                            "coarse_j": info["coarse_j"],
                            "iterations_completed": trace[-1]["iteration"],
                            "stop_reason": trace[-1]["stop_reason"],
                        }
                        validation.append(row)
                        losses.append(row["mse"])
                    except (FloatingPointError, RuntimeError) as exc:
                        failures.append(
                            {
                                "model": name,
                                "ratio": ratio,
                                "dark_sigma": dark,
                                "configuration_id": configuration_id,
                                "case": case,
                                "error": str(exc),
                            }
                        )
                        losses.append(math.inf)
                scores.append(float(np.mean(losses)))
            write_json(output / "failures.json", failures)
            if not any(math.isfinite(score) for score in scores):
                protocol = json.loads((output / "protocol.json").read_text())
                protocol.update(
                    status="failed",
                    error=f"All configurations failed: {name}, {ratio}, {dark}",
                )
                write_json(output / "protocol.json", protocol)
                raise RuntimeError(protocol["error"])
            best_id = int(np.argmin(scores))
            best = grid[best_id]
            chosen.append(
                {
                    "model": name,
                    "ratio": ratio,
                    "dark_sigma": dark,
                    "config": best,
                    "validation_mse": scores[best_id],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
            for case, (clean, record, acquisition, info) in enumerate(
                make_cases(corpus, "test", ratio, dark, per_domain, dev)
            ):
                reconstruction, trace = solve(
                    acquisition.op, acquisition.measurement, network, best, iterations
                )
                baseline = acquisition.op.adjoint(acquisition.measurement)
                test.append(
                    {
                        "model": name,
                        "ratio": ratio,
                        "dark_sigma": dark,
                        "coarse_j": info["coarse_j"],
                        "domain": record["domain"],
                        "case": case,
                        "source_hash": record["sha256"],
                        "configuration_id": best_id,
                        "config": json.dumps(best, sort_keys=True),
                        **metrics(reconstruction, clean),
                        **{
                            "baseline_" + k: v
                            for k, v in metrics(baseline, clean).items()
                        },
                        "seconds": trace[-1]["seconds"],
                        "iterations_completed": trace[-1]["iteration"],
                        "stop_reason": trace[-1]["stop_reason"],
                    }
                )
                traces.extend(
                    {
                        "model": name,
                        "ratio": ratio,
                        "dark_sigma": dark,
                        "case": case,
                        **item,
                    }
                    for item in trace
                )
                if ratio == 0.2 and dark == 0.2 and case in (0, per_domain):
                    np.savez_compressed(
                        output / f"{name}_{record['domain']}_example.npz",
                        clean=clean.cpu().numpy(),
                        initial=baseline.cpu().numpy(),
                        reconstruction=reconstruction.cpu().numpy(),
                    )
            write_json(output / "selected.json", chosen)
            write_csv(output / "validation.csv", validation)
            write_csv(output / "test.csv", test)
            write_csv(output / "iteration_traces.csv", traces)
            print(
                f"SELECTED {name} ratio={ratio} dark={dark} config={best} validation_mse={scores[best_id]:.6g}",
                flush=True,
            )
        del network, checkpoint
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    protocol = json.loads((output / "protocol.json").read_text())
    protocol.update(
        status="completed",
        validation_reconstructions=len(validation),
        test_reconstructions=len(test),
        configurations_selected=len(chosen),
        failures=len(failures),
    )
    write_json(output / "protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["dwtdnet", "gs_dwtdnet", "drunet", "gs_drunet"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--per-domain", type=int, default=2)
    args = parser.parse_args()
    result = run_sweep(
        args.manifest,
        args.runs,
        args.output,
        args.models,
        device=args.device,
        iterations=args.iterations,
        per_domain=args.per_domain,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
