"""Train and evaluate randomized-noise models with explicit pilot budgets."""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from LION.pcm.randomized.data import MixedImageCorpus, sha256, write_json
from LION.pcm.randomized.monitoring import append_event, training_preview
from LION.pcm.randomized.models import (
    DENOISERS,
    REGULARIZERS,
    build_denoiser,
    build_regularizer,
    denoise,
    regularized_reconstruction,
)
from LION.pcm.randomized.noise import (
    NoiseRange,
    image_corruption,
    randomized_acquisition,
    sample_parameters,
    split_ssdu,
)
from LION.pcm.randomized.selection import ssdu_validation_score


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    steps: int = 128
    side: int = 64
    batch_size: int = 2
    learning_rate: float = 2e-4
    unroll_iterations: int = 16
    validation_every: int = 32
    validation_per_domain: int = 2
    seed: int = 20261014
    noise_mode: str = "mixture"
    device: str = "cuda:0"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metrics(prediction: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    predicted = prediction.detach().float().cpu().squeeze().numpy()
    target = clean.detach().float().cpu().squeeze().numpy()
    mse = float(np.mean((predicted.astype(np.float64) - target) ** 2))
    if not np.isfinite(mse):
        raise FloatingPointError("Non-finite reconstruction metric")
    return {
        "mse": mse,
        "psnr": float(-10 * math.log10(max(mse, 1e-12))),
        "ssim": float(structural_similarity(target, predicted, data_range=1.0)),
    }


def seed_model(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    config: TrainingConfig,
    corpus: MixedImageCorpus,
    *,
    split: str,
    per_domain: int,
    save_predictions: Path | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fixed cases and RNG seeds across methods; test never influences selection."""
    device = torch.device(config.device)
    model.eval()
    rows, examples = [], {}
    cases = corpus.fixed_cases(split, per_domain)
    denoising = config.model in DENOISERS
    for index, (clean_cpu, record) in enumerate(cases):
        clean = clean_cpu.to(device)
        for severity, dark in (("low", 0.05), ("medium", 0.2), ("extreme", 0.8)):
            conditions = (
                [("gaussian", None), ("pcm_like", None)]
                if denoising
                else [("physical_pcm", ratio) for ratio in (0.01, 0.2, 1.0)]
            )
            for kind, ratio in conditions:
                seed = 900_000 + index * 100 + (0 if split == "validation" else 50_000)
                rng = np.random.default_rng(seed)
                p = sample_parameters(rng, NoiseRange())
                p.dark_sigma = dark
                p.beam_sigma = {"low": 1.2e-4, "medium": 6e-4, "extreme": 1e-3}[
                    severity
                ]
                if denoising:
                    noisy, sigma, _ = image_corruption(
                        clean, rng, NoiseRange(), mode=kind, parameters=p
                    )
                    synchronize(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        predicted = denoise(model, noisy, sigma, training=False)
                    synchronize(device)
                    duration = time.perf_counter() - start
                    baseline = noisy
                else:
                    acquisition, _ = randomized_acquisition(
                        clean, rng, NoiseRange(), ratio=ratio, parameters=p
                    )
                    synchronize(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        predicted = regularized_reconstruction(
                            model, acquisition.op, acquisition.measurement
                        )
                    synchronize(device)
                    duration = time.perf_counter() - start
                    baseline = acquisition.op.adjoint(acquisition.measurement)
                row = {
                    "model": config.model,
                    "training_noise": config.noise_mode,
                    "split": split,
                    "case": index,
                    "domain": record["domain"],
                    "source_hash": record["sha256"],
                    "source_path": record["path"],
                    "severity": severity,
                    "noise_kind": kind,
                    "sampling_ratio": ratio if ratio is not None else 1.0,
                    "seconds": duration,
                    **metrics(predicted, clean),
                    **{
                        "baseline_" + key: value
                        for key, value in metrics(baseline, clean).items()
                    },
                }
                rows.append(row)
                if severity == "medium" and (
                    denoising and kind == "pcm_like" or not denoising and ratio == 0.2
                ):
                    key = f"{record['domain']}_{index}"
                    examples[key + "_clean"] = clean.detach().cpu().numpy()
                    examples[key + "_input"] = baseline.detach().cpu().numpy()
                    examples[key + "_prediction"] = predicted.detach().cpu().numpy()
    if save_predictions is not None:
        np.savez_compressed(save_predictions, **examples)
    return rows, examples


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty results")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config: TrainingConfig, manifest: Path, output: Path) -> dict:
    """Perform a bounded run. Complete means the requested updates, not convergence."""
    if config.model not in DENOISERS + REGULARIZERS:
        raise ValueError(config.model)
    if config.steps < 1 or config.batch_size < 1 or config.validation_every < 1:
        raise ValueError("Training budgets must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run {output}")
    output.mkdir(parents=True)
    device = torch.device(config.device)
    corpus = MixedImageCorpus(manifest, config.side)
    seed_model(config.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    denoising = config.model in DENOISERS
    self_supervised = config.model.endswith("ssdu")
    model = (
        build_denoiser(config.model, config.side, device)
        if denoising
        else build_regularizer(
            config.model, config.side, config.unroll_iterations, device
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    params = sum(p.numel() for p in model.parameters())
    metadata = {
        "config": asdict(config),
        "noise_bounds": asdict(NoiseRange()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "parameters": params,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "interpretation": "bounded pilot from random initialization, not a converged training result",
        "self_supervision": "measurement split of one correlated PCM acquisition; noise-unbiasedness not claimed",
        "selection_criterion": "heldout_measurement_mse"
        if self_supervised
        else "clean_validation_mse",
        "denoiser_conditioning": "sampled absolute dark/Gaussian sigma; DWTDNet currently ignores the sigma argument",
    }
    write_json(output / "run.json", metadata)
    bounds = NoiseRange()
    history, best_score, best_step, best_image_mse = [], math.inf, 0, math.inf
    synchronize(device)
    start = time.perf_counter()
    draws_file = (output / "draws.jsonl").open("w")
    progress_file = output / "history.jsonl"

    def save_checkpoint(path: Path, step: int) -> None:
        temporary = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_updates": step,
                "config": asdict(config),
                "manifest_sha256": metadata["manifest_sha256"],
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            temporary,
        )
        temporary.replace(path)

    try:
        initial_rows, _ = evaluate(
            model,
            config,
            corpus,
            split="validation",
            per_domain=config.validation_per_domain,
        )
        initial_mse = float(np.mean([r["mse"] for r in initial_rows]))
        initial = {
            "step": 0,
            "training_loss": None,
            "validation_mse": initial_mse,
            "seconds": time.perf_counter() - start,
        }
        history.append(initial)
        append_event(progress_file, initial)
        # Only trained checkpoints compete; initialization remains an explicit baseline.
        for step in range(1, config.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for batch_index in range(config.batch_size):
                draw_seed = config.seed + step * 1009 + batch_index
                rng = np.random.default_rng(draw_seed)
                clean_cpu, record = corpus.draw(rng)
                clean = clean_cpu.to(device)
                if denoising:
                    noisy, sigma, info = image_corruption(
                        clean, rng, bounds, mode=config.noise_mode
                    )
                    predicted = denoise(model, noisy, sigma, training=True)
                    loss = (predicted - clean).square().mean()
                else:
                    acquisition, info = randomized_acquisition(clean, rng, bounds)
                    if self_supervised:
                        input_op, input_y, loss_op, loss_y, partition = split_ssdu(
                            acquisition, rng
                        )
                        predicted = regularized_reconstruction(model, input_op, input_y)
                        # Neither training loss nor checkpoint selection uses clean targets.
                        loss = (loss_op(predicted) - loss_y).square().mean()
                        info["ssdu_input_count"] = len(partition["input_rows"])
                        info["ssdu_loss_count"] = len(partition["loss_rows"])
                    else:
                        predicted = regularized_reconstruction(
                            model, acquisition.op, acquisition.measurement
                        )
                        loss = (predicted - clean).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}")
                (loss / config.batch_size).backward()
                total_loss += float(loss.detach()) / config.batch_size
                draws_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "batch": batch_index,
                            "seed": draw_seed,
                            "domain": record["domain"],
                            "source_hash": record["sha256"],
                            **info,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            item = {
                "step": step,
                "training_loss": total_loss,
                "gradient_norm": float(norm),
                "validation_mse": None,
                "seconds": time.perf_counter() - start,
            }
            if step % config.validation_every == 0 or step == config.steps:
                validation_rows, validation_examples = evaluate(
                    model,
                    config,
                    corpus,
                    split="validation",
                    per_domain=config.validation_per_domain,
                )
                image_mse = float(np.mean([r["mse"] for r in validation_rows]))
                figures_folder = output / "validation_steps"
                figures_folder.mkdir(parents=True, exist_ok=True)
                write_csv(figures_folder / f"step_{step:06d}.csv", validation_rows)
                training_preview(output, step, validation_examples)
                score = (
                    ssdu_validation_score(model, config, corpus)
                    if self_supervised
                    else image_mse
                )
                item.update(validation_mse=image_mse, selection_score=score)
                if score < best_score:
                    best_score, best_step, best_image_mse = score, step, image_mse
                    save_checkpoint(output / "best.pt", step)
                    write_csv(output / "validation.csv", validation_rows)
                save_checkpoint(output / "last.pt", step)
                draws_file.flush()
                print(
                    f"{config.model}/{config.noise_mode} step={step}/{config.steps} loss={total_loss:.6g} val_mse={image_mse:.6g} selection_score={score:.6g}",
                    flush=True,
                )
            history.append(item)
            append_event(progress_file, item)
            if step % config.validation_every == 0 or step == config.steps:
                write_json(output / "history.json", history)
        checkpoint = torch.load(
            output / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_rows, _ = evaluate(
            model,
            config,
            corpus,
            split="test",
            per_domain=config.validation_per_domain,
            save_predictions=output / "examples.npz",
        )
        write_csv(output / "test.csv", test_rows)
        synchronize(device)
        metadata.update(
            status="completed",
            completed_updates=config.steps,
            best_step=best_step,
            initial_validation_mse=initial_mse,
            best_validation_mse=best_image_mse,
            best_selection_score=best_score,
            elapsed_seconds=time.perf_counter() - start,
            test_mean_mse=float(np.mean([r["mse"] for r in test_rows])),
            test_mean_psnr=float(np.mean([r["psnr"] for r in test_rows])),
        )
        write_json(output / "run.json", metadata)
        return metadata
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", metadata)
        raise
    finally:
        draws_file.close()
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
