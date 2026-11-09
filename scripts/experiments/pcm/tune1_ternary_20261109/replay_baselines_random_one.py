"""Replay old fixed PCM baseline arrays for all 45 selected held-out test cases.

Runs AFTER both PnP tuning studies have all selections frozen.
Reuses exact original acquisitions; never retunes to test metrics.
C60 is handled by replay_c60.py with pinned original epoch-60 source.
GS oracle, GS SSDU-5, and GS expanded are already stored for all 450.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from LION.pcm.reconstruct import (
    ReconInput,
    ScalarTVReconFn,
    ScalarTVParameter,
    spgl1_reconstruction,
)
from LION.pcm.spgl1_bpdn import SPGL1BPDNReconFn
from LION.experiments.pcm_presets.presets import QUICK_METHODS, NOISY_PCM_METHODS
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import (
    oldbench,
    live_admm,
)
from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    BASE,
    OUTPUT,
    CATEGORIES,
    frozen_json,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import make_model

PREVIOUS = BASE / "ssdu_pnp_tuning_20261108_v1"
MATCHED = PREVIOUS / "test450/v1/merged/all_450_11_method_metrics.csv"
RAW_OLD = {
    "gs_pnp_oracle": PREVIOUS / "test450/v1/raw/gs_oracle_step6",
    "gs_pnp_ssdu5": PREVIOUS / "test450/v1/raw/gs_ssdu",
    "gs_pnp_expanded5": PREVIOUS / "next_gs_ssdu_sweep_v1/test450/raw",
}
BASICS = ("zero_filled", "spgl1_bp", "spgl1_bpdn", "pnp_admm_oracle", "tv")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def configure():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    return torch.device("cuda:0")


def run(category: str):
    assert category in CATEGORIES
    assert_all_frozen()
    selected = [
        c
        for c in json.loads((OUTPUT / "prepared/test45.json").read_text())["cases"]
        if c["category"] == category
    ]
    assert len(selected) == 9
    with MATCHED.open(newline="") as f:
        benchmark = {(x["category"], int(x["sample_id"])): x for x in csv.DictReader(f)}
    assert len(benchmark) == 450
    device = configure()
    model = make_model("admm", device)
    bp_config = deepcopy(QUICK_METHODS["spgl1"])
    bpdn = SPGL1BPDNReconFn(deepcopy(NOISY_PCM_METHODS["spgl1_bpdn"]), device=device)
    tv = ScalarTVReconFn(
        config=ScalarTVParameter(
            lambda_reg=0.011,
            iters=1024,
            primal_stepsize=0.3,
            dual_stepsize=0.3,
            theta=1.0,
            metric_every=1024,
        )
    )
    base, all_ids = oldbench.config_and_ids()
    for case in selected:
        sid = int(case["sample_id"])
        assert sid in all_ids
        path = oldbench.DATA / category / f"image{sid:04d}.png"
        raw = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        gt = torch.from_numpy((raw.astype(np.float64) / 255.0).astype(np.float32))[
            None, None
        ].to(device)
        truth = gt.detach().cpu()
        dr = float((gt.max() - gt.min()).item())
        acq, indices = oldbench.acquisition(gt, base, category, sid, device)
        y = acq.measurement
        assert sha(y.detach().cpu().numpy()) == case["measurement_sha256"]
        assert sha(np.asarray(indices, dtype=np.int64)) == case["pattern_sha256"]
        old = benchmark[category, sid]
        assert old["measurement_sha256"] == case["measurement_sha256"]
        assert old["pattern_sha256"] == case["pattern_sha256"]
        ri = ReconInput(op=acq.op, y=y, x0=acq.op.pseudo_inv(y), i0=None)
        original_cache = (
            BASE
            / "pcm_utv_fixed20_npl_default_20261103/cases"
            / f"{category}_{sid:04d}.npz"
        )
        cached = {}
        if original_cache.exists():
            with np.load(original_cache, allow_pickle=False) as source:
                if (
                    sha(source["measurement"]) == case["measurement_sha256"]
                    and sha(source["sampled_indices"]) == case["pattern_sha256"]
                ):
                    for key in ("zero_filled", "spgl1_bp", "C"):
                        if key in source:
                            cached[key] = source[key].copy()
        for name in BASICS:
            result = (
                OUTPUT / "test45/baselines/raw" / name / f"{category}_{sid:04d}.npy"
            )
            report = (
                OUTPUT
                / "test45/baselines/provenance"
                / name
                / f"{category}_{sid:04d}.json"
            )
            if result.exists() and report.exists():
                previous = json.loads(report.read_text())
                assert previous["measurement_sha256"] == case["measurement_sha256"]
                continue
            begin = time.perf_counter()
            with torch.no_grad():
                if name == "zero_filled":
                    x = (
                        ri.x0
                        if "zero_filled" not in cached
                        else torch.from_numpy(cached["zero_filled"])[None, None].to(
                            device
                        )
                    )
                elif name == "spgl1_bp":
                    x = (
                        torch.from_numpy(cached["spgl1_bp"])[None, None].to(device)
                        if "spgl1_bp" in cached
                        else spgl1_reconstruction(
                            acq.op, y, bp_config, device=device, shows_prog_bar=False
                        )
                    )
                elif name == "spgl1_bpdn":
                    x = bpdn(ri).reconstruction
                elif name == "pnp_admm_oracle":
                    x = live_admm(acq.op, y, model, rho=0.5, sigma=0.075)["x"]
                elif name == "tv":
                    x = tv(ri).reconstruction
                else:
                    raise RuntimeError("unknown method")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - begin
            img = x.detach().cpu().numpy().squeeze().astype(np.float32)
            assert img.shape == (256, 256) and np.isfinite(img).all()
            scored = oldbench.score(x, truth, dr)
            expected = {
                k: float(old[f"{name}_{k}"]) for k in ("psnr", "ssim", "haarpsi", "mse")
            }
            delta = {k: float(abs(scored[k] - expected[k])) for k in expected}
            # The original SPGL1-BP solver may replay with small numerical
            # differences from the 2026-11-03 stored benchmark. Never change
            # its parameters to match a test metric. Record the difference.
            # All comparison tables use the original frozen benchmark score.
            if name == "spgl1_bp":
                assert delta["psnr"] < 1.0 and delta["ssim"] < 0.08, (
                    "SPGL1-BP replay differs too much; manual source audit required",
                    category,
                    sid,
                    delta,
                )
                if delta["psnr"] > 0.10:
                    print("SPGL1_BP_REPLAY_WARNING", category, sid, delta, flush=True)
            else:
                assert (
                    delta["psnr"] < 0.10
                    and delta["ssim"] < 0.015
                    and delta["haarpsi"] < 0.015
                ), (category, sid, name, delta)
            result.parent.mkdir(parents=True, exist_ok=True)
            with result.with_suffix(".partial").open("xb") as f:
                np.save(f, img, allow_pickle=False)
            result.with_suffix(".partial").replace(result)
            frozen_json(
                report,
                {
                    "method": name,
                    "category": category,
                    "sample_id": sid,
                    "measurement_sha256": case["measurement_sha256"],
                    "pattern_sha256": case["pattern_sha256"],
                    "reconstruction_sha256": sha(img),
                    "archived_metric_diff": delta,
                    "replayed_metrics": scored,
                    "archived_metrics": expected,
                    "elapsed_seconds": duration,
                    "no_test_driven_tuning": True,
                },
            )
            print(
                "BASELINE_REPLAY",
                category,
                sid,
                name,
                "PSNR_DELTA",
                round(delta["psnr"], 5),
                flush=True,
            )
        # Existing GS methods and expanded GS are already stored for all 450 images.
        for name, root in RAW_OLD.items():
            path = root / f"{category}_{sid:04d}.npy"
            assert path.exists(), (name, path)
            im = np.load(path, allow_pickle=False)
            assert im.shape == (256, 256) and np.isfinite(im).all()
            scored = oldbench.score(torch.from_numpy(im)[None, None], truth, dr)
            expected = (
                float(old[f"{name}_psnr"]) if name != "gs_pnp_expanded5" else None
            )
            if expected is not None:
                assert abs(scored["psnr"] - expected) < 0.01
        print("BASELINE_CASE_READY", category, sid, flush=True)
    print("BASELINE_CATEGORY_READY", category, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    args = ap.parse_args()
    run(args.category)
