"""Matched 450-image PCM test: fixed 20% Hadamard sampling and fixed NPL default noise.

Seven methods see exactly the same acquisition per image. Different images get
independent deterministic masks and noise traces. Existing results are never read.
"""
import argparse, csv, dataclasses, hashlib, json, os, subprocess, time
from pathlib import Path
from copy import deepcopy
import numpy as np, torch
from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.data_loaders.pcm.data_primitives import SamplesParameter
from LION.experiments.pcm_presets.presets import QUICK_METHODS
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
)
from LION.pcm.pcm_sapm import (
    load_trained_pcm_checkpoint_model,
    make_pcm_forward_data,
    forward_func_sapm_pcm,
)
from LION.pcm.reconstruct import GSPnPReconFn, ReconInput, spgl1_reconstruction
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import sample_indices
from LION.utils.metrics import compute_metrics
from LION.utils.reproduce import seed_from_parts

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "pcm_utv_fixed20_npl_default_20261103"
DATA = Path(
    "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
)
GPU = "GPU-05267a12-f71f-101d-59e5-05e5c430d8ef"
CATS = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
METHODS = ("zero_filled", "spgl1_bp", "gs_pnp", "A", "B", "C", "old_utv")
METRICS = ("mse", "psnr", "ssim", "haarpsi")
NS = "pcm_utv_fixed20_npl_default_20261103_v1"
J = 8
COARSE_J = 6
RATIO = 0.20
VISUAL_SEED = 2026110316
BASE = (
    ROOT
    / "pcm_utv_zero_inclusive_4070/20261026_122910_TV_supervised_zero_inclusive_rtx4070_adam_base2e4_scale1e3_raw01_1000ep/resolved_config.yaml"
)
RUNS = {
    "A": (
        ROOT
        / "pcm_supervised_varied_noise/20261024_231223_TV_supervised_varied_noise_ratio005_05_variable_npl/resolved_config.yaml",
        ROOT
        / "pcm_supervised_varied_noise/20261024_231223_TV_supervised_varied_noise_ratio005_05_variable_npl/checkpoints/best_val_loss_s0.20_cJ6.pt",
        38,
    ),
    "B": (
        BASE,
        ROOT / "pcm_utv_4070_4x128_final_20261101/checkpoints/final_best_epoch44.pt",
        44,
    ),
    "C": (
        ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/requested_config.yaml",
        ROOT
        / "pcm_utv_4090_3x56_winner_snapshot_20261103/checkpoints/best_val_epoch60.pt",
        60,
    ),
    "old_utv": (
        ROOT / "pcm_supervised/20260920_025657_TV/resolved_config.yaml",
        ROOT
        / "pcm_supervised/20260920_025657_TV/checkpoints/best_val_loss_s0.20_cJ6.pt",
        99,
    ),
}
EXPECTED_NOISE = {
    "beam_sigma": 1.2e-4,
    "bias_sigma": 2.9e-5,
    "bias_gradient_sigma": 1.6e-7,
    "dark_sigma": 0.040290178571428574,
}


def sha_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed(*parts):
    return int(seed_from_parts(NS, *parts))


def config_and_ids():
    cfg = load_pcm_sapm_supervised_config_from_path(BASE)
    assert cfg.data.j_order == J and cfg.data.train_perm_range == [1, 900]
    assert cfg.data.val_perm_range == [901, 910] and cfg.data.test_perm_range == [
        911,
        1000,
    ]
    assert cfg.sampling.randomising_scheme == "multilevel"
    assert (
        cfg.noise is not None and cfg.noise.name == "PCM noise simulation - NPL default"
    )
    for name, value in EXPECTED_NOISE.items():
        assert abs(float(getattr(cfg.noise.noise_params, name)) - value) < 1e-12
    ids = [int(v) for v in cfg.data.samples.ids[910:1000]]
    assert len(ids) == len(set(ids)) == 90
    for cat in CATS:
        assert all((DATA / cat / f"image{x:04d}.png").is_file() for x in ids)
    return cfg, ids


def visual_ids(ids):
    return {
        cat: sorted(
            map(
                int,
                np.random.default_rng(VISUAL_SEED + sum(map(ord, cat))).choice(
                    ids, size=10, replace=False
                ),
            )
        )
        for cat in CATS
    }


def prepare():
    cfg, ids = config_and_ids()
    modelinfo = {}
    for name, (cf, cp, epoch) in RUNS.items():
        assert cf.is_file() and cp.is_file()
        mc = load_pcm_sapm_supervised_config_from_path(cf)
        assert [int(v) for v in mc.data.samples.ids[910:1000]] == ids
        modelinfo[name] = {
            "checkpoint": str(cp),
            "config": str(cf),
            "checkpoint_sha256": sha_file(cp),
            "config_sha256": sha_file(cf),
            "epoch": epoch,
            "cnn_n_blocks": mc.model.cnn.n_blocks,
            "cnn_init_filters": mc.model.cnn.init_filters,
            "unrolled_iterations": mc.model.n_iters,
            "trained_noise": mc.noise is not None,
        }
    protocol = {
        "protocol": "fixed20-npl-default-one-acquisition-per-image-v1",
        "source_revision": subprocess.check_output(
            ["git", "-C", "/home/t/Documents/GIT/LION", "rev-parse", "HEAD"], text=True
        ).strip(),
        "test_permutation_positions": [911, 1000],
        "test_ids": ids,
        "categories": CATS,
        "sampling_ratio": RATIO,
        "coarse_J": COARSE_J,
        "J": J,
        "sampling_scheme": "multilevel",
        "noise_preset": cfg.noise.name,
        "noise_parameters_fixed": EXPECTED_NOISE,
        "noise_parameter_sampling": False,
        "randomisation": "Per-image seeded mask and PCM noise realisation; common to every method",
        "seed_namespace": NS,
        "selected_50_seed": VISUAL_SEED,
        "selected_50_ids": visual_ids(ids),
        "methods": METHODS,
        "models": modelinfo,
    }
    for d in (OUT, OUT / "results", OUT / "cases", OUT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    p = OUT / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol, "Protocol drift"
    else:
        p.write_text(json.dumps(protocol, indent=2) + "\n")
    print("PREPARED", p, "sampling", RATIO, "noise", EXPECTED_NOISE)
    for cat, choices in protocol["selected_50_ids"].items():
        print("SELECTED", cat, choices)


def acquisition(gt, base, cat, sid, device):
    # Fixed 20% multilevel Hadamard sampling. Only mask identities vary by image.
    indices = sample_indices(
        j_order=J,
        sampling_ratio=RATIO,
        coarse_j=COARSE_J,
        randomising_scheme="multilevel",
        rng=np.random.default_rng(seed(cat, str(sid), "mask")),
    )
    assert len(indices) == int(RATIO * 2 ** (2 * J)), len(indices)
    # The NPL-default noise standard deviations remain unchanged for every image.
    obs = simulate_pcm_acquisition(
        gt,
        j_order=J,
        sampled_indices=indices,
        noise=base.noise,
        rng=np.random.default_rng(seed(cat, str(sid), "noise")),
        device=device,
    )
    return obs, indices


def load_models(device):
    models = {}
    for name, (cf, cp, expected) in RUNS.items():
        cfg = load_pcm_sapm_supervised_config_from_path(cf)
        cases = make_supervised_pcm_training_cases(cfg)
        assert len(cases) == 1
        model, epoch, steps = load_trained_pcm_checkpoint_model(
            config=cfg, cases=cases, checkpoint_path=cp, device=device
        )
        assert epoch == expected, (name, epoch)
        model.eval()
        models[name] = (cfg, model)
        print("CHECKPOINT", name, "epoch", epoch, "steps", steps, flush=True)
    return models


def score(x, target_cpu, data_range):
    assert x.shape == target_cpu.shape and bool(torch.isfinite(x).all())
    values = compute_metrics(
        recon=x.detach().cpu(), target=target_cpu, data_range=data_range
    )
    return {k: float(values[k]) for k in METRICS}


def run(cat, start, end):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
    assert torch.cuda.is_available() and "4070" in torch.cuda.get_device_name(0)
    assert cat in CATS and 0 <= start < end <= 90
    assert (OUT / "protocol.json").exists(), "Run --stage prepare first"
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        protocol["sampling_ratio"] == RATIO
        and protocol["noise_parameters_fixed"] == EXPECTED_NOISE
    )
    base, ids = config_and_ids()
    segment = ids[start:end]
    ds = PCMSyntheticPNGTorchDataset(
        dataclasses.replace(
            base.data, samples=SamplesParameter(ids=segment), categories=[cat]
        )
    )
    dst = OUT / "results" / f"{cat}_{start:02d}_{end:02d}.csv"
    assert not dst.exists(), dst
    tmp = dst.with_suffix(".partial")
    assert not tmp.exists()
    device = torch.device("cuda:0")
    models = load_models(device)
    gs_cfg = deepcopy(QUICK_METHODS["pnp_gs_deepinv_drunet"])
    assert (gs_cfg.iters, gs_cfg.sigma, gs_cfg.lambd, gs_cfg.uses_deepinv) == (
        50,
        0.05,
        0.1,
        True,
    )
    denoiser = build_grayscale_denoiser(deepcopy(gs_cfg.denoiser), device).eval()
    denoiser.requires_grad_(False)
    sp_cfg = deepcopy(QUICK_METHODS["spgl1"])
    assert (sp_cfg.factor, sp_cfg.iters, sp_cfg.debias_iters, sp_cfg.wavelet_name) == (
        1e5,
        100,
        10,
        "db4",
    )
    fields = [
        "category",
        "sample_id",
        "sampling_ratio",
        "num_patterns",
        "pattern_sha256",
        "measurement_sha256",
        "fixed_noise_sha256",
        "spgl1_runtime_s",
        "gs_runtime_s",
        "gs_accepted_iterations",
    ]
    fields += [f"{method}_{metric}" for method in METHODS for metric in METRICS]
    visual = set(protocol["selected_50_ids"][cat])
    noise_sha = hashlib.sha256(
        json.dumps(EXPECTED_NOISE, sort_keys=True).encode()
    ).hexdigest()
    t0 = time.monotonic()
    with tmp.open("w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, sid in enumerate(segment):
            target = ds[i]
            gt = target[None, None].to(device)
            ref = gt.detach().cpu()
            data_range = float((target.max() - target.min()).item())
            acq, indices = acquisition(gt, base, cat, sid, device)
            op, y = acq.op, acq.measurement
            yh = sha_tensor(y)
            row = {
                "category": cat,
                "sample_id": sid,
                "sampling_ratio": RATIO,
                "num_patterns": len(indices),
                "pattern_sha256": hashlib.sha256(
                    np.asarray(indices, dtype=np.int64).tobytes()
                ).hexdigest(),
                "measurement_sha256": yh,
                "fixed_noise_sha256": noise_sha,
            }
            images = {}
            images["zero_filled"] = op.pseudo_inv(y).detach()
            t = time.perf_counter()
            images["spgl1_bp"] = spgl1_reconstruction(
                op, y, sp_cfg, device=device, shows_prog_bar=False
            ).detach()
            torch.cuda.synchronize()
            row["spgl1_runtime_s"] = time.perf_counter() - t
            accepted = []
            gs = GSPnPReconFn(
                gs_cfg,
                denoiser,
                device=device,
                iteration_callback=lambda k, x: accepted.append(k),
            )
            t = time.perf_counter()
            images["gs_pnp"] = gs(
                ReconInput(op=op, y=y, x0=op.pseudo_inv(y), i0=None)
            ).reconstruction.detach()
            torch.cuda.synchronize()
            row["gs_runtime_s"] = time.perf_counter() - t
            row["gs_accepted_iterations"] = len(accepted)
            for name, (cfg, model) in models.items():
                data = make_pcm_forward_data(
                    config=cfg,
                    op=op,
                    measurement=y,
                    key=0,
                    sampling_ratio=RATIO,
                    coarse_j=COARSE_J,
                    device=device,
                )
                images[name] = forward_func_sapm_pcm(
                    model=model, data=data, config=cfg, skip_grad_checkpoint=True
                ).reconstruction.detach()
            for name, recon in images.items():
                vals = score(recon, ref, data_range)
                for metric in METRICS:
                    row[f"{name}_{metric}"] = vals[metric]
            assert sha_tensor(y) == yh, "Method changed the shared measurement"
            if sid in visual:
                case = OUT / "cases" / f"{cat}_{sid:04d}.npz"
                assert not case.exists(), case
                np.savez_compressed(
                    case,
                    ground_truth=target.numpy(),
                    measurement=y.detach().cpu().numpy(),
                    sampled_indices=np.asarray(indices, dtype=np.int64),
                    sampling_ratio=np.asarray(RATIO),
                    **{name: x.squeeze().cpu().numpy() for name, x in images.items()},
                )
                print("VISUAL_SAVED", cat, sid, flush=True)
            writer.writerow(row)
            f.flush()
            if (i + 1) % 5 == 0 or i == len(segment) - 1:
                print(
                    "PROGRESS",
                    cat,
                    start + i + 1,
                    "/90",
                    round(time.monotonic() - t0, 1),
                    "seconds",
                    flush=True,
                )
    tmp.replace(dst)
    print("FRAGMENT_COMPLETE", dst, "images", len(segment), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    parser.add_argument("--category", choices=CATS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=15)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        assert args.category
        run(args.category, args.start, args.end)


if __name__ == "__main__":
    main()
