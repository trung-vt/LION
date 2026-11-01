"""CPU-only common-acquisition reconstruction audit for current experiments_10."""
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

from LION.pcm.pcm_config import load_pcm_sapm_supervised_config_from_path
from LION.experiments.pcm_experiments import make_test_cases
from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.pcm.pcm_sapm import (
    make_pcm_forward_data,
    forward_func_sapm_pcm,
    load_supervised_pcm_checkpoint_model,
)
from LION.pcm.simulation import sample_and_simulate_pcm_acquisition
from LION.pcm.reconstruct import (
    ReconInput,
    spgl1_reconstruction,
    GSPnPReconFn,
    PnPADMMReconFn,
)
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.experiments.pcm_presets.presets import QUICK_METHODS
from LION.utils.metrics import compute_metrics
from LION.utils.reproduce import seed_from_parts

ROOT = Path(
    "/store/LION/ttv22/pcm/results/pcm/pcm_supervised_varied_noise/20261024_231223_TV_supervised_varied_noise_ratio005_05_variable_npl"
)
CKPT = ROOT / "final_selection/best_available_validation_epoch38.pt"
OUT = Path("method_audit_evidence").resolve()
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
IMAGE_IDS = (611, 68)
MODE = sys.argv[1] if len(sys.argv) > 1 else "basic"
assert MODE in ("basic", "admm", "gs")
assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_available()
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
import subprocess

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert (
    subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    == "experiments_10"
)
cfg = load_pcm_sapm_supervised_config_from_path(ROOT / "resolved_config.yaml")
dataset = PCMSyntheticPNGTorchDataset(cfg.data)
cases = make_test_cases(cfg)
assert cases == [(0.2, 6)]
assert cfg.data.samples and cfg.data.samples.ids
assert all(s in cfg.data.samples.ids[910:1000] for s in IMAGE_IDS)
assert cfg.noise is not None


def process(category, sid, model=None):
    seed = seed_from_parts(
        str(cfg.reproduce.base_seed), category, str(sid), str(0.2), str(6)
    )
    target = dataset.make_item(dataset.sample_png_path(category, sid))[None, None].to(
        "cpu"
    )
    with torch.inference_mode():
        acq = sample_and_simulate_pcm_acquisition(
            target,
            j_order=8,
            sampling=cfg.sampling,
            sampling_ratio=0.2,
            coarse_j=6,
            noise=cfg.noise,
            rng=np.random.default_rng(seed),
            seed=seed,
            device=torch.device("cpu"),
        )
    fingerprint = hashlib.sha256(acq.measurement.detach().numpy().tobytes()).hexdigest()
    op = acq.op
    y = acq.measurement
    ground = target.detach().cpu()
    out = OUT / f"{category}_{sid:04d}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ground_truth.npy", ground.squeeze().numpy())
    recordpath = out / "provenance.json"
    prov = json.loads(recordpath.read_text()) if recordpath.exists() else {}
    if prov and prov.get("measurement_sha256") != fingerprint:
        raise ValueError("Different measurement seed detected")
    prov.update(
        {
            "category": category,
            "sample_id": sid,
            "branch": "experiments_10",
            "git_commit": commit,
            "acquisition_seed": int(seed),
            "ratio": 0.2,
            "coarse_j": 6,
            "noise": "PCM NPL default",
            "measurement_sha256": fingerprint,
            "device": "cpu",
            "checkpoint_38": str(CKPT),
        }
    )
    recordpath.write_text(json.dumps(prov, indent=2) + "\n")
    data = make_pcm_forward_data(
        config=cfg,
        op=op,
        measurement=y,
        key=0,
        sampling_ratio=0.2,
        coarse_j=6,
        device=torch.device("cpu"),
    )
    zf = op.pseudo_inv(y)
    reconin = ReconInput(op=op, y=y, x0=zf, i0=None)
    if MODE == "basic":
        target_arr = target.squeeze().numpy()
        with torch.inference_mode():
            save_one(out, "zero_filled", zf, target, prov)
            print("BASIC", category, sid, "ZERO", flush=True)
        try:
            t0 = time.monotonic()
            with torch.inference_mode():
                sp = spgl1_reconstruction(
                    op, y, QUICK_METHODS["spgl1"], device=torch.device("cpu")
                )
            save_one(out, "spgl1", sp, target, prov, seconds=time.monotonic() - t0)
            print("BASIC", category, sid, "SPGL1", flush=True)
        except Exception as e:
            print("ERROR SPGL1", type(e).__name__, str(e), flush=True)
            (out / "error_spgl1.txt").write_text(traceback.format_exc())
        try:
            t0 = time.monotonic()
            if model is None:
                model, ep, step = load_supervised_pcm_checkpoint_model(
                    config=cfg,
                    cases=cases,
                    checkpoint_path=CKPT,
                    device=torch.device("cpu"),
                )
                assert ep == 38
            with torch.inference_mode():
                su = forward_func_sapm_pcm(
                    model=model, data=data, config=cfg, skip_grad_checkpoint=True
                ).reconstruction
            save_one(
                out,
                "supervised_utv_ep38",
                su,
                target,
                prov,
                seconds=time.monotonic() - t0,
            )
            print("BASIC", category, sid, "SUPERVISED", flush=True)
        except Exception as e:
            print("ERROR SUPERVISED", type(e).__name__, str(e), flush=True)
            (out / "error_supervised_utv.txt").write_text(traceback.format_exc())
    else:
        method_id = {"admm": "pnp_admm_drunet", "gs": "pnp_gs_deepinv_drunet"}[MODE]
        try:
            pc = QUICK_METHODS[method_id]
            t0 = time.monotonic()
            den = build_grayscale_denoiser(pc.denoiser, torch.device("cpu"))
            den.eval()
            print(
                "MODEL LOADED",
                method_id,
                "SECONDS",
                round(time.monotonic() - t0, 1),
                flush=True,
            )
            fn = (
                PnPADMMReconFn(config=pc, denoiser=den, device=torch.device("cpu"))
                if MODE == "admm"
                else GSPnPReconFn(config=pc, denoiser=den, device=torch.device("cpu"))
            )
            t0 = time.monotonic()
            with torch.inference_mode() if MODE == "admm" else torch.enable_grad():
                output = fn(reconin).reconstruction.detach()
            save_one(
                out, method_id, output, target, prov, seconds=time.monotonic() - t0
            )
            print(
                "COMPLETED",
                method_id,
                category,
                sid,
                "SECONDS",
                round(time.monotonic() - t0, 1),
                flush=True,
            )
        except Exception as e:
            print("ERROR", method_id, type(e).__name__, str(e), flush=True)
            (out / f"error_{method_id}.txt").write_text(traceback.format_exc())


def save_one(out, method, recon, target, prov, seconds=None):
    recon = recon.detach().cpu().contiguous()
    np.save(out / (method + ".npy"), recon.squeeze().numpy())
    met = compute_metrics(
        recon=recon,
        target=target,
        data_range=float((target.max() - target.min()).item()),
    )
    d = {"method": method, "metrics": met, "seconds": seconds}
    (out / (method + ".json")).write_text(json.dumps(d, indent=2) + "\n")
    print(
        "METRIC",
        method,
        os.path.basename(out),
        {k: round(v, 5) for k, v in met.items()},
        flush=True,
    )


if __name__ == "__main__":
    print("METHOD", MODE, "GIT", commit, "CUDA", torch.cuda.is_available(), flush=True)
    chosen = (68,) if MODE != "basic" else IMAGE_IDS
    model = None
    for sid in chosen:
        process("CIGS", sid, model=model)
