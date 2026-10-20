"""Reproducible PCM PnP-ADMM OLS/WLS noise robustness experiment.

Use disjoint synthetic split positions: validation 801-900, test 901-1000
in the established 5678 permutation. No ground truth is used to set WLS weights.
Outputs are under /store/LION/ttv22/pcm/results, outside the repository.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from LION.data_loaders.pcm.data_primitives import PCMSingleSampleParameter
from LION.experiments.pcm_presets.presets import PRESETS, QUICK_METHODS
from LION.pcm.data import prepare_data
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.pcm.simulation import sample_and_simulate_pcm_acquisition
from LION.pcm_tools.noise_generate import generate_beam_noise
from LION.pcm_tools.pcm_prox import (
    closed_form_pcm_prox_with_precomputed_terms,
    precompute_pcm_prox_terms,
)
from LION.utils.metrics import compute_metrics
from scripts.snippets.random_utils import seed_from_parts

ROOT = Path("/store/LION/ttv22/pcm/results/pcm_admm_noise_study_20261020")
DATA = Path(
    "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
)
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
TEST_PERM = np.random.default_rng(5678).permutation(np.arange(1, 1001))
CSV_COLUMNS = (
    "phase",
    "case",
    "category",
    "sample_id",
    "noise_seed",
    "fidelity",
    "denoiser",
    "rho",
    "sigma",
    "iter",
    "psnr",
    "mse",
    "ssim",
    "haarpsi",
    "duration",
    "weight_min",
    "weight_max",
)


@dataclass
class Case:
    name: str
    category: str
    sample_id: int
    seed: int
    target: torch.Tensor
    op: object
    y: torch.Tensor
    zf: torch.Tensor
    noise: object


def load_case(
    category: str,
    sample_id: int,
    noise_seed: int,
    device: torch.device,
    ratio: float,
    coarse: int,
    *,
    noiseless: bool = False,
) -> Case:
    config = PRESETS["synthetic_png_dataset_box_plots"]
    import copy

    config = copy.deepcopy(config)
    filename = f"{category}/image{sample_id:04d}.png"
    config.this_sample = PCMSingleSampleParameter(
        data_filename=filename, ground_truth_filename=filename
    )
    config.sampling.sampling_ratios = [ratio]
    config.sampling.coarse_j_values = [coarse]
    target_2d, _ = prepare_data(config, device=device)
    target = target_2d[None, None]
    name = f"{category}_{sample_id:04d}"
    # Keep mask independent of noise seed; store this seed in the output.
    sample_seed = seed_from_parts(
        "1",
        category,
        str(sample_id),
        name,
        str(ratio),
        str(coarse),
        str(config.base_seed),
    )
    rng = np.random.default_rng(sample_seed)
    noise_rng = np.random.default_rng(
        seed_from_parts(str(sample_seed), "noise", str(noise_seed))
    )
    noise = None if noiseless else config.noise
    acquisition = sample_and_simulate_pcm_acquisition(
        config.scaling.scale(target),
        j_order=8,
        sampling=config.sampling,
        sampling_ratio=ratio,
        coarse_j=coarse,
        noise=noise,
        rng=rng,
        noise_rng=noise_rng,
        device=device,
        seed=sample_seed,
    )
    zf = acquisition.op.pseudo_inv(acquisition.measurement)
    return Case(
        name,
        category,
        sample_id,
        noise_seed,
        target,
        acquisition.op,
        acquisition.measurement,
        zf,
        noise,
    )


@torch.no_grad()
def diagonal_weights(case: Case, draws: int = 24, cap: float = 10.0) -> torch.Tensor:
    """Estimate diagonal inverse variance from observations, not target.

    Simulate integrated ARMA beam traces in chronological acquisition order.
    The pairwise swap in the PCM simulation is reproduced. Cross-measurement
    covariance is ignored (diagonal WLS, not GLS).
    """
    if case.noise is None:
        return torch.ones_like(case.y)
    noise = case.noise
    params = noise.noise_params
    y = case.y.detach().reshape(-1).double().cpu().numpy()
    n = case.target.shape[-1]
    flux = float(case.zf.sum().item()) / n
    plus = np.maximum(0.0, 0.5 * (flux + y))
    minus = np.maximum(0.0, 0.5 * (flux - y))
    i_bias = flux / noise.signal_to_bias_ratio
    white_var = (
        2 * (float(params.dark_sigma or 0.0) ** 2)
        + 2 * (i_bias * float(params.bias_sigma or 0.0)) ** 2
        + (i_bias * float(params.bias_gradient_sigma or 0.0)) ** 2
    )
    mean = np.zeros(len(y), dtype=np.float64)
    m2 = np.zeros_like(mean)
    rng = np.random.default_rng(210123)
    for k in range(draws):
        beam = generate_beam_noise(
            num_measurements=2 * len(y),
            beam_ar=params.beam_ar,
            beam_ma=params.beam_ma,
            beam_sigma=params.beam_sigma,
            rng=rng,
        ).reshape(-1, 2)
        swap = rng.random(len(y)) < 0.5
        a = np.where(swap, beam[:, 1], beam[:, 0])
        b = np.where(swap, beam[:, 0], beam[:, 1])
        residual = plus * a - minus * b
        delta = residual - mean
        mean += delta / (k + 1)
        m2 += delta * (residual - mean)
    var = m2 / max(draws - 1, 1) + white_var
    var = np.maximum(var, 1e-20)
    weight = np.median(var) / var
    weight = np.clip(weight, 1.0 / cap, cap)
    return torch.as_tensor(weight, device=case.y.device, dtype=case.y.dtype).reshape_as(
        case.y
    )


@torch.no_grad()
def weighted_prox_setup(case: Case, weights: torch.Tensor):
    op = case.op
    sty = op.s.adjoint(weights * case.y)
    mask = op.s.adjoint(weights)
    return sty, mask


@torch.no_grad()
def wls_prox(op, z: torch.Tensor, sty: torch.Tensor, mask: torch.Tensor, rho: float):
    z_vec = op.vectorize(z)
    q = op.p(op.w(z_vec))
    alpha = (sty + rho * q) / (mask + rho)
    return op.unvectorize(op.w.adjoint(op.p.adjoint(alpha)))


@torch.no_grad()
def run_admm(
    case: Case,
    denoiser: torch.nn.Module,
    rho: float,
    sigma: float,
    fidelity: str,
    iterations: int,
    eval_every: int,
    draws: int = 24,
) -> tuple[list[dict], dict]:
    op, y = case.op, case.y
    if fidelity == "wls":
        weights = diagonal_weights(case, draws=draws)
        sty, mask = weighted_prox_setup(case, weights)
    else:
        weights = torch.ones_like(y)
        sty, mask = precompute_pcm_prox_terms(op, y)
    x = torch.zeros_like(case.zf)
    v, u = torch.zeros_like(x), torch.zeros_like(x)
    records: list[dict] = []
    start = time.perf_counter()
    rng = float((case.target.max() - case.target.min()).item())
    best = {"psnr": -math.inf, "iter": 0, "mse": float("inf")}
    for step in range(1, iterations + 1):
        z = v - u
        if fidelity == "wls":
            x = wls_prox(op, z, sty, mask, rho)
        else:
            x = closed_form_pcm_prox_with_precomputed_terms(
                op, z=z, rho=rho, ST_y=sty, m_plus_rho=mask + rho
            )
        v = denoiser(x + u, sigma=sigma)
        u = u + (x - v)
        if step == 1 or step % eval_every == 0 or step == iterations:
            mse = float(torch.mean((x - case.target) ** 2).item())
            psnr = 10 * math.log10(rng * rng / mse) if mse > 0 else float("inf")
            rec = {
                "iter": step,
                "psnr": psnr,
                "mse": mse,
                "duration": time.perf_counter() - start,
                "primal_residual": float(torch.linalg.vector_norm(x - v).item()),
                "dual_norm": float(torch.linalg.vector_norm(u).item()),
            }
            records.append(rec)
            if math.isfinite(psnr) and psnr > best["psnr"]:
                best = {"psnr": psnr, "iter": step, "mse": mse}
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(v).all()):
            break
    return records, {
        **best,
        "duration": time.perf_counter() - start,
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
    }


def add_results(output: Path, records: list[dict]):
    file = output / "runs.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if stream.tell() == 0:
            w.writeheader()
        w.writerows(records)


def report_metrics(x: torch.Tensor, target: torch.Tensor) -> dict:
    vals = compute_metrics(
        recon=x, target=target, data_range=float((target.max() - target.min()).item())
    )
    return {
        k: float(v) for k, v in vals.items() if k in ("mse", "psnr", "ssim", "haarpsi")
    }


def select_cases(
    phase: str,
    device: torch.device,
    ratio: float,
    coarse: int,
    limit: int = 0,
    noiseless: bool = False,
) -> list[Case]:
    if phase in ("smoke", "coarse", "refine"):
        # Three disjoint validation samples, 1 per material. Never select on test samples.
        plan = [
            ("CIGS", int(TEST_PERM[800])),
            ("Organic", int(TEST_PERM[801])),
            ("Silicone", int(TEST_PERM[802])),
        ]
        if phase == "smoke":
            plan = plan[:1]
        if limit:
            plan = plan[:limit]
        return [
            load_case(cat, sid, 0, device, ratio, coarse, noiseless=noiseless)
            for cat, sid in plan
        ]
    # Independent held-out test permutation positions.
    plan = [
        (cat, int(TEST_PERM[pos]))
        for pos, cat in zip([990, 991, 992, 993, 994], CATEGORIES, strict=True)
    ]
    if limit:
        plan = plan[:limit]
    return [
        load_case(cat, sid, seed, device, ratio, coarse, noiseless=noiseless)
        for cat, sid in plan
        for seed in (0, 1)
    ]


def summarize_curves(curves: dict, case_names: list[str], output: Path):
    """Rank fixed iteration checkpoints by mean validation PSNR, not oracle per sample."""
    rankings = []
    for (fidelity, rho, sigma), bycase in curves.items():
        if len(bycase) != len(case_names):
            continue
        common = set.intersection(
            *(set(row["iter"] for row in bycase[name]) for name in case_names)
        )
        for step in common:
            nums = [
                next(row["psnr"] for row in bycase[name] if row["iter"] == step)
                for name in case_names
            ]
            if all(math.isfinite(v) for v in nums):
                rankings.append(
                    {
                        "fidelity": fidelity,
                        "rho": rho,
                        "sigma": sigma,
                        "iter": step,
                        "mean_psnr": float(np.mean(nums)),
                        "worst_psnr": float(min(nums)),
                    }
                )
    rankings.sort(key=lambda d: d["mean_psnr"], reverse=True)
    (output / "ranking.json").write_text(json.dumps(rankings[:100], indent=2))
    for item in rankings[:12]:
        print("RANK", json.dumps(item), flush=True)
    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("smoke", "coarse", "refine", "test"), default="smoke"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument(
        "--denoiser",
        choices=("drunet", "dwtdnet", "pilot-gaussian", "pilot-pcm"),
        default="drunet",
    )
    parser.add_argument(
        "--fidelities", nargs="+", choices=("ols", "wls"), default=["ols", "wls"]
    )
    parser.add_argument(
        "--rhos", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1]
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.025, 0.05, 0.1])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--coarse", type=int, default=6)
    parser.add_argument("--draws", type=int, default=24)
    parser.add_argument("--noise-free", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device, "PHASE", args.phase, flush=True)
    output = args.output / args.phase / ("no_noise" if args.noise_free else "npl_noise")
    output.mkdir(parents=True, exist_ok=True)
    meta = vars(args).copy()
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}
    meta["git_revision"] = os.popen("git rev-parse HEAD").read().strip()
    meta["test_ids"] = [int(TEST_PERM[i]) for i in range(990, 995)]
    meta["val_ids"] = [int(TEST_PERM[i]) for i in range(800, 803)]
    (output / "settings.json").write_text(json.dumps(meta, indent=2))
    cases = select_cases(
        args.phase, device, args.ratio, args.coarse, args.case_limit, args.noise_free
    )
    denoiser_id = (
        "pnp_admm_drunet" if args.denoiser == "drunet" else "pnp_admm_dwtdnet_tiny"
    )
    if args.denoiser.startswith("pilot-"):
        from scripts.pcm.pcm_train_noise_denoiser import OUT, ResidualDenoiser

        checkpoint = torch.load(
            OUT / args.denoiser.removeprefix("pilot-") / "best.pt",
            map_location=device,
            weights_only=False,
        )
        denoiser = ResidualDenoiser(**checkpoint["architecture"]).to(device)
        denoiser.load_state_dict(checkpoint["model"])
        denoiser.eval()
    else:
        denoiser = build_grayscale_denoiser(QUICK_METHODS[denoiser_id].denoiser, device)
    curves: dict = {}
    for case in cases:
        zfm = report_metrics(case.zf, case.target)
        print("CASE", case.name, "SEED", case.seed, "ZERO_FILLED", zfm, flush=True)
        add_results(
            output,
            [
                {
                    "phase": args.phase,
                    "case": case.name,
                    "category": case.category,
                    "sample_id": case.sample_id,
                    "noise_seed": case.seed,
                    "fidelity": "zero_filled",
                    "denoiser": "none",
                    "iter": 0,
                    **zfm,
                }
            ],
        )
        for fidelity, rho, sigma in itertools.product(
            args.fidelities, args.rhos, args.sigmas
        ):
            start = time.perf_counter()
            rows, summary = run_admm(
                case,
                denoiser,
                rho,
                sigma,
                fidelity,
                args.iters,
                args.eval_every,
                draws=args.draws,
            )
            key = (fidelity, rho, sigma)
            curves.setdefault(key, {})[case.name + f"_seed{case.seed}"] = rows
            selected = [next((r for r in rows if r["iter"] == args.iters), rows[-1])]
            rec = {
                "phase": args.phase,
                "case": case.name,
                "category": case.category,
                "sample_id": case.sample_id,
                "noise_seed": case.seed,
                "fidelity": fidelity,
                "denoiser": args.denoiser,
                "rho": rho,
                "sigma": sigma,
                "iter": selected[0]["iter"],
                "psnr": selected[0]["psnr"],
                "mse": selected[0]["mse"],
                **{k: summary[k] for k in ("weight_min", "weight_max")},
                "duration": summary["duration"],
            }
            add_results(output, [rec])
            curves_dir = output / "curves"
            curves_dir.mkdir(exist_ok=True)
            curve_file = curves_dir / (
                f"{case.name}_seed{case.seed}_{fidelity}_"
                f"rho{rho:g}_sigma{sigma:g}.json"
            )
            curve_file.write_text(json.dumps(rows, indent=2))
            print(
                "RUN",
                case.name,
                fidelity,
                rho,
                sigma,
                "FINAL",
                round(rec["psnr"], 3),
                "BEST",
                round(summary["psnr"], 3),
                "ITER",
                summary["iter"],
                "SEC",
                round(time.perf_counter() - start, 2),
                flush=True,
            )
    if args.phase != "test":
        names = [case.name + f"_seed{case.seed}" for case in cases]
        summarize_curves(curves, names, output)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
