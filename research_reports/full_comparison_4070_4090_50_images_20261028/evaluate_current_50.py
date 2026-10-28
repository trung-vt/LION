"""Independent paired PCM evaluation on two predeclared acquisition distributions."""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.data_loaders.pcm.data_primitives import PCMRandomRange, SamplesParameter
from LION.pcm.noise import PCMAcquisitionVariation
from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
    supervised_pcm_scale_keys,
)
from LION.pcm.pcm_sapm import (
    forward_func_sapm_pcm,
    load_trained_pcm_checkpoint_model,
    make_pcm_forward_data,
)
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import sample_indices
from LION.utils.metrics import compute_metrics
from LION.utils.reproduce import seed_from_parts

OUT = Path("/store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028")
CONF = Path(
    "/tmp/lion_pcm_utv_zero_4070_train_20261026/config/pcm/pcm_utv_zero_inclusive_4070_adam_lr2e4_scale1e3.yaml"
)
NS = "pcm_independent_holdout_v1_20261027"
MEASURES = ("mse", "psnr", "ssim", "haarpsi")
REGIMES = ("zero_inclusive_4070", "positive_4090")


def seed(*args: str) -> int:
    return int(seed_from_parts(NS, *args))


def measure_digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main(category: str) -> None:
    protocol = json.loads((OUT / "protocol_manifest_50.json").read_text())
    prior = json.loads(Path(protocol["protocol_reused_from"]).read_text())
    assert category in protocol["categories"]
    config = load_pcm_sapm_supervised_config_from_path(CONF)
    assert config.validation_sampling is not None
    assert config.acquisition_variation is not None and config.noise is not None
    assert config.data.test_perm_range == [911, 1000]
    assert config.data.samples is not None and config.data.samples.ids is not None
    ids = list(map(int, protocol["subset_per_category"][category]))
    assert (
        len(ids) == 10
        and len(set(ids)) == 10
        and set(ids).issubset(set(map(int, config.data.samples.ids[910:1000])))
    )
    assert config.sampling.randomising_scheme == "multilevel"
    assert config.data.j_order == 8
    var_4070 = config.acquisition_variation
    ranges = prior["noise"]["positive_4090"]
    var_4090 = PCMAcquisitionVariation(
        noise_sigma_ranges={
            k: PCMRandomRange(**v)
            for k, v in ranges.items()
            if k != "current_to_model_scale"
        },
        current_to_model_scale=float(ranges["current_to_model_scale"]),
    )
    variations = {"zero_inclusive_4070": var_4070, "positive_4090": var_4090}
    cases = make_supervised_pcm_training_cases(config)
    keys = supervised_pcm_scale_keys(config)
    assert keys == (0,) and len(cases) == 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "CATEGORY",
        category,
        "DEVICE",
        device,
        "NAME",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        flush=True,
    )
    models = {}
    for label in ("4070", "4090"):
        ckpt = OUT / f"{label}_best_20261028.pt"
        model, epoch, step = load_trained_pcm_checkpoint_model(
            config=config,
            cases=cases,
            checkpoint_path=ckpt,
            device=device,
            model_keys=keys,
        )
        assert epoch == protocol["checkpoints_current_frozen"][label]["epoch"]
        model.eval()
        models[label] = model
        print("MODEL_LOADED", label, epoch, step, flush=True)
    dataset = PCMSyntheticPNGTorchDataset(
        dataclasses.replace(
            config.data, samples=SamplesParameter(ids=ids), categories=[category]
        )
    )
    dest = OUT / f"current_{category}.csv"
    assert not dest.exists(), f"Existing file: {dest}"
    old_rows = list(csv.DictReader((OUT / "old_reference_metrics_subset.csv").open()))
    old_index = {
        (r["category"], int(r["sample_id"]), r["condition"]): r for r in old_rows
    }
    fields = [
        "category",
        "sample_id",
        "condition",
        "sampling_ratio",
        "num_patterns",
        "pattern_sha256",
        "measurement_sha256",
        "dark_sigma_raw",
        "bias_sigma",
        "bias_gradient_sigma",
        "beam_sigma",
    ] + [f"{name}_{metric}" for name in models for metric in MEASURES]
    # Load every independent test image, and simulate one acquisition per regime.
    start = time.monotonic()
    with dest.open("w", newline="") as fh, torch.no_grad():
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for j, sid in enumerate(ids):
            target = dataset[j]
            sample = target[None, None, :, :].to(device)
            ratio = float(
                np.random.default_rng(seed(category, str(sid), "ratio")).uniform(
                    0.05, 0.5
                )
            )
            indices = sample_indices(
                j_order=8,
                sampling_ratio=ratio,
                coarse_j=6,
                randomising_scheme=config.sampling.randomising_scheme,
                rng=np.random.default_rng(seed(category, str(sid), "pattern")),
            )
            idx_hash = hashlib.sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest()
            for cond in REGIMES:
                sig_rng = np.random.default_rng(
                    seed(category, str(sid), cond, "sigmas")
                )
                drawn = variations[cond].draw_noise(
                    sig_rng, noise=config.noise, j_order=8
                )
                acq = simulate_pcm_acquisition(
                    sample,
                    j_order=8,
                    sampled_indices=indices,
                    noise=drawn,
                    rng=np.random.default_rng(seed(category, str(sid), cond, "noise")),
                    device=device,
                )
                data = make_pcm_forward_data(
                    config=config,
                    op=acq.op,
                    measurement=acq.measurement,
                    key=0,
                    sampling_ratio=ratio,
                    coarse_j=6,
                    device=device,
                )
                orig = measure_digest(data.measurement)
                row = {
                    "category": category,
                    "sample_id": int(sid),
                    "condition": cond,
                    "sampling_ratio": ratio,
                    "num_patterns": len(indices),
                    "pattern_sha256": idx_hash,
                    "measurement_sha256": orig,
                    "dark_sigma_raw": drawn.noise_params.dark_sigma
                    * (2**8)
                    / variations[cond].current_to_model_scale,
                    "bias_sigma": drawn.noise_params.bias_sigma,
                    "bias_gradient_sigma": drawn.noise_params.bias_gradient_sigma,
                    "beam_sigma": drawn.noise_params.beam_sigma,
                }
                old = old_index[(category, int(sid), cond)]
                assert row["pattern_sha256"] == old["pattern_sha256"], (
                    category,
                    sid,
                    cond,
                    "pattern differs",
                )
                assert row["measurement_sha256"] == old["measurement_sha256"], (
                    category,
                    sid,
                    cond,
                    "measurement differs",
                )
                assert row["sampling_ratio"] == float(old["sampling_ratio"]), (
                    category,
                    sid,
                    cond,
                    "ratio differs",
                )
                assert row["num_patterns"] == int(old["num_patterns"]), (
                    category,
                    sid,
                    cond,
                    "pattern count differs",
                )
                for field in (
                    "dark_sigma_raw",
                    "bias_sigma",
                    "bias_gradient_sigma",
                    "beam_sigma",
                ):
                    assert abs(float(row[field]) - float(old[field])) < 1e-10, (
                        category,
                        sid,
                        cond,
                        field,
                    )
                # A single forward-data object is passed unchanged to each frozen model.
                examples = (
                    {"ground_truth": target.detach().cpu().numpy()} if j == 0 else None
                )
                if examples is not None:
                    examples["zero_filled"] = (
                        acq.op.pseudo_inv(acq.measurement).detach().cpu().numpy()
                    )
                for label, model in models.items():
                    recon = (
                        forward_func_sapm_pcm(
                            model=model,
                            data=data,
                            config=config,
                            skip_grad_checkpoint=True,
                        )
                        .reconstruction.detach()
                        .cpu()
                    )
                    results = compute_metrics(
                        recon=recon,
                        target=target[None, None, :, :],
                        data_range=float((target.max() - target.min()).item()),
                    )
                    for metric in MEASURES:
                        value = float(results[metric])
                        if not math.isfinite(value):
                            raise ValueError(
                                f"Non-finite metric at {category} {sid} {cond} {label}: {metric}={value}"
                            )
                        row[f"{label}_{metric}"] = value
                    assert (
                        measure_digest(data.measurement) == orig
                    ), "Input was mutated by model"
                    if examples is not None:
                        examples[f"{label}_current"] = recon.numpy()
                if examples is not None:
                    np.savez_compressed(
                        OUT / f"example_{category}_{cond}.npz", **examples
                    )
                writer.writerow(row)
            fh.flush()
            time.sleep(
                0.5
            )  # Low-priority evaluation: reduce concurrent GPU contention.
            if (j + 1) % 10 == 0:
                print(
                    "PROGRESS",
                    category,
                    j + 1,
                    "/",
                    len(ids),
                    "seconds",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
    print(
        "COMPLETE",
        category,
        "records",
        len(ids) * len(REGIMES),
        "seconds",
        round(time.monotonic() - start, 1),
        "csv",
        dest,
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        required=True,
        choices=["CIGS", "Organic", "Perovskite", "Reference", "Silicone"],
    )
    main(parser.parse_args().category)
