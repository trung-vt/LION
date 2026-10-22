"""SSDU diagnostics with randomized training, fixed validation/test grids.

Three training arms isolate noiseless, independent dark Gaussian, and
correlated default PCM noise. Test/reference images never enter the loss or
checkpoint selection; validation uses held-out measured coefficients only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from LION.pcm.randomized.data import MixedImageCorpus, sha256, write_json
from LION.pcm.randomized.fixed_protocol import (
    NOISE_SCENARIOS,
    SAMPLING_RATIOS,
    fixed_acquisition,
    fixed_coarse_j,
    fixed_noise_config,
    protocol_summary,
)
from LION.pcm.randomized.models import build_regularizer, regularized_reconstruction
from LION.pcm.randomized.monitoring import append_event, training_preview
from LION.pcm.randomized.noise import sample_pattern_indices, split_ssdu
from LION.pcm.randomized.runner import metrics, seed_model, synchronize
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.pcm.randomized.unseen_benchmark import write_rows


@dataclass
class Diagnostic:
    model: str
    regime: str
    steps: int = 192
    side: int = 64
    batch_size: int = 2
    unroll: int = 64
    validation_every: int = 64
    per_domain: int = 4
    seed: int = 20261014
    lr: float = 2e-4
    device: str = "cuda:1"


def _acquire_train(clean: torch.Tensor, scenario: str, rng: np.random.Generator):
    side = clean.shape[-1]
    j = side.bit_length() - 1
    # Randomize sampling rate and noise realizations in training, but use fixed
    # coefficient magnitudes per arm so the causal comparison is interpretable.
    ratio = float(rng.uniform(0.1, 0.5))
    indices, sampling = sample_pattern_indices(
        side,
        rng,
        ratio=ratio,
        coarse_j=fixed_coarse_j(j),
    )
    noise = fixed_noise_config(scenario, j)
    acquisition = simulate_pcm_acquisition(
        clean,
        j_order=j,
        sampled_indices=indices,
        noise=noise,
        rng=rng,
        device=clean.device,
    )
    return acquisition, sampling


def _fixed_evaluate(
    model,
    corpus: MixedImageCorpus,
    *,
    split: str,
    scenario: str,
    per_domain: int,
    device: torch.device,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    model.eval()
    rows = []
    examples = {}
    for case, (clean_cpu, record) in enumerate(corpus.fixed_cases(split, per_domain)):
        clean = clean_cpu.to(device)
        for ratio in SAMPLING_RATIOS:
            acquisition, meta = fixed_acquisition(
                clean,
                split=split,
                case_index=case,
                ratio=ratio,
                scenario=scenario,
            )
            split_rng = np.random.default_rng(
                7_000_000 + case * 10_000 + int(round(ratio * 1000))
            )
            input_op, input_y, heldout_op, heldout_y, partition = split_ssdu(
                acquisition, split_rng
            )
            with torch.no_grad():
                image = regularized_reconstruction(model, input_op, input_y)
                # For final inference one would pass *all* data. Report both,
                # keeping measurement-only checkpoint selection explicit.
                full = regularized_reconstruction(
                    model, acquisition.op, acquisition.measurement
                )
                heldout_mse = float((heldout_op(image) - heldout_y).square().mean())
                full_metrics = metrics(full, clean)
                input_metrics = metrics(input_op.adjoint(input_y), clean)
            if not math.isfinite(heldout_mse):
                raise FloatingPointError("Non-finite held-out measurement score")
            rows.append(
                {
                    "split": split,
                    "case": case,
                    "domain": record["domain"],
                    "source_hash": record["sha256"],
                    "scenario": scenario,
                    "ratio": ratio,
                    "coarse_j": meta["sampling"]["coarse_j"],
                    "mask_sha256": meta["mask_sha256"],
                    "input_rows": len(partition["input_rows"]),
                    "holdout_rows": len(partition["loss_rows"]),
                    "heldout_measurement_mse": heldout_mse,
                    "image_mse": full_metrics["mse"],
                    "psnr": full_metrics["psnr"],
                    "ssim": full_metrics["ssim"],
                    "baseline_psnr": input_metrics["psnr"],
                    "baseline_mse": input_metrics["mse"],
                }
            )
            if np.isclose(ratio, 0.2) and scenario in (
                "noiseless",
                "gaussian_dark",
                "pcm_default_10ms",
            ):
                key = f'{record["domain"]}_{case}'
                if not (key + "_clean") in examples:
                    examples[key + "_clean"] = clean.detach().cpu().numpy()
                    examples[key + "_input"] = (
                        input_op.adjoint(input_y).detach().cpu().numpy()
                    )
                    examples[key + "_prediction"] = full.detach().cpu().numpy()
    return rows, examples


def train_one(
    manifest: Path,
    output: Path,
    cfg: Diagnostic,
) -> dict:
    if cfg.model not in ("utv_ssdu", "utgv_ssdu"):
        raise ValueError("Only SSDU models are supported in this diagnostic")
    if cfg.regime not in ("noiseless", "gaussian_dark", "pcm_default_10ms"):
        raise ValueError("Regime must be a diagnostic arm")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(cfg.device)
    torch.set_num_threads(2)
    seed_model(cfg.seed)
    corpus = MixedImageCorpus(manifest, side=cfg.side)
    model = build_regularizer(cfg.model, cfg.side, cfg.unroll, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    meta = {
        "status": "running",
        "config": asdict(cfg),
        "manifest_sha256": sha256(manifest),
        "fixed_evaluation_grid": protocol_summary(cfg.side.bit_length() - 1),
        "selection": "validation held-out measurement MSE on regime-specific 10/20/50% fixed conditions",
        "no_reference_based_checkpoint_selection": True,
        "split_policy": "DC retained in SSDU input, 20% of rows held out",
        "training": "random image/crop, flips and rotations, random rate uniform[.1,.5], fixed NPL noise amplitudes per regime with new noise realizations per crop",
        "data_provenance": "training on DIV2K/PCM synthetic manifest; no unseen NPL maps",
    }
    write_json(output / "run.json", meta)
    best_loss = math.inf
    best_step = None
    start = time.perf_counter()
    try:
        for step in range(1, cfg.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for batch in range(cfg.batch_size):
                rng = np.random.default_rng(cfg.seed + step * 1009 + batch)
                clean, record = corpus.draw(rng)
                clean = clean.to(device)
                acquisition, info = _acquire_train(clean, cfg.regime, rng)
                input_op, input_y, loss_op, loss_y, partition = split_ssdu(
                    acquisition, rng
                )
                predicted = regularized_reconstruction(model, input_op, input_y)
                loss = (loss_op(predicted) - loss_y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / cfg.batch_size).backward()
                loss_value += float(loss.detach()) / cfg.batch_size
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            update = {
                "step": step,
                "training_measurement_loss": loss_value,
                "gradient_norm": float(grad_norm),
                "seconds": time.perf_counter() - start,
                "validation_measurement_mse": None,
                "validation_image_mse_audit_only": None,
            }
            if step % cfg.validation_every == 0 or step == cfg.steps:
                val, images = _fixed_evaluate(
                    model,
                    corpus,
                    split="validation",
                    scenario=cfg.regime,
                    per_domain=cfg.per_domain,
                    device=device,
                )
                score = float(np.mean([r["heldout_measurement_mse"] for r in val]))
                audit_mse = float(np.mean([r["image_mse"] for r in val]))
                update.update(
                    validation_measurement_mse=score,
                    validation_image_mse_audit_only=audit_mse,
                )
                write_rows(output / f"validation_step_{step:06d}.csv", val)
                training_preview(output, step, images)
                if score < best_loss:
                    best_loss, best_step = score, step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "completed_updates": step,
                            "config": asdict(cfg),
                            "manifest_sha256": sha256(manifest),
                        },
                        output / "best.pt",
                    )
                print(
                    f"{cfg.model}/{cfg.regime}: {step}/{cfg.steps} "
                    f"train={loss_value:.5g} val_ssdu={score:.5g} audit_image_mse={audit_mse:.5g}",
                    flush=True,
                )
            append_event(output / "history.jsonl", update)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "completed_updates": cfg.steps,
                "config": asdict(cfg),
                "manifest_sha256": sha256(manifest),
            },
            output / "last.pt",
        )
        model.load_state_dict(
            torch.load(output / "best.pt", map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        tests = []
        for scenario in NOISE_SCENARIOS:
            rows, _ = _fixed_evaluate(
                model,
                corpus,
                split="test",
                scenario=scenario,
                per_domain=cfg.per_domain,
                device=device,
            )
            tests.extend(rows)
            write_rows(output / "test_fixed_grid.csv", tests)
        synchronize(device)
        meta.update(
            status="completed",
            best_step=best_step,
            best_validation_measurement_mse=best_loss,
            completed_updates=cfg.steps,
            test_rows=len(tests),
            seconds=time.perf_counter() - start,
        )
        write_json(output / "run.json", meta)
        return meta
    except Exception as e:
        meta.update(status="failed", error=f"{type(e).__name__}: {e}")
        write_json(output / "run.json", meta)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", choices=["utv_ssdu", "utgv_ssdu"], required=True)
    p.add_argument(
        "--regime",
        choices=["noiseless", "gaussian_dark", "pcm_default_10ms"],
        required=True,
    )
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--unroll", type=int, default=64)
    p.add_argument("--per-domain", type=int, default=4)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    print(
        json.dumps(
            train_one(
                args.manifest,
                args.output,
                Diagnostic(
                    model=args.model,
                    regime=args.regime,
                    steps=args.steps,
                    unroll=args.unroll,
                    per_domain=args.per_domain,
                    device=args.device,
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
