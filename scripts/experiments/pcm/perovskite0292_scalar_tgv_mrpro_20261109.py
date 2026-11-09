"""Scalar second-order TGV oracle search for frozen Perovskite 0292 PCM.

Uses the existing LION AdaptiveTGVPCMNetwork, its MRPro operator matrix,
and LION's MRPro-backed PDHG. Does not introduce a new TGV solver.

Objective:
  0.5 ||A u-y||^2 + alpha1 ||grad(u)-v||_1 + alpha0 ||E(v)||_1
The chosen implementation has componentwise L1 penalties (not isotropic
Euclidean/Frobenius pixel norms); this detail is important when comparing
parameters with other TGV libraries.

Only Perovskite_0292 is tuned using ground-truth PSNR (oracle illustration).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import load_pcm_sapm_supervised_config_from_path
from LION.pcm.pcm_sapm import make_pcm_forward_data
from LION.utils.metrics import compute_metrics

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1"
BASE = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
CONFIG = ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/requested_config.yaml"
ADMM = ROOT / "bad_images_for_utv_20261109_v3_23/raw_new9/Perovskite_0292.npy"
UTV_BEST = (
    ROOT
    / "perovskite0292_c60_ternary_with_baselines_20261109_v1/ternary_selected_reconstruction.npz"
)
METRIC450 = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
TAU = SIGMA = 0.18
THETA = 1.0
COARSE_A1 = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
COARSE_RATIO = (0.5, 1.0, 2.0, 4.0, 8.0)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def measure(im, gt):
    out = compute_metrics(
        recon=torch.from_numpy(np.asarray(im, dtype=np.float32))[None, None],
        target=torch.from_numpy(gt)[None, None],
        data_range=1.0,
    )
    return {k: float(out[k]) for k in METRICS}


def read_existing(path):
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    assert records
    tmp = path.with_suffix(".pending")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    tmp.replace(path)


def setup():
    import LION.pcm.pcm_config as m

    assert "source_9188b07" in str(Path(m.__file__).resolve()), m.__file__
    torch.set_num_threads(3)
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    with np.load(BASE, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
    with METRIC450.open(newline="") as f:
        row = next(
            x
            for x in csv.DictReader(f)
            if x["category"] == "Perovskite" and int(x["sample_id"]) == 292
        )
    assert (
        gt.shape == (256, 256)
        and y.shape == (1, 1, 13107)
        and indices.shape == (13107,)
    )
    assert sha(y) == row["measurement_sha256"]
    assert sha(indices) == row["pattern_sha256"]
    admm = np.load(ADMM, allow_pickle=False)
    with np.load(UTV_BEST, allow_pickle=False) as z:
        utv = z["reconstruction"].copy()
        rho = float(z["rho"])
        np.testing.assert_array_equal(z["ground_truth"], gt)
        np.testing.assert_array_equal(z["measurement"], y)
        np.testing.assert_array_equal(z["indices"], indices)
    for im, key in [(admm, "pnp_admm_ssdu5")]:
        mse = float(np.mean((im.astype(np.float64) - gt.astype(np.float64)) ** 2))
        np.testing.assert_allclose(mse, float(row[key + "_mse"]), rtol=4e-5, atol=1e-8)
    config = load_pcm_sapm_supervised_config_from_path(CONFIG)
    tgv_cfg = deepcopy(config.model)
    tgv_cfg.pdhg.regularisation = "TGV"
    tgv_cfg.pdhg.primal_stepsize = TAU
    tgv_cfg.pdhg.dual_stepsize = SIGMA
    assert tgv_cfg.pdhg.theta == 1.0
    model = AdaptiveTGVPCMNetwork(config=tgv_cfg, lambda_map_network=None).to(device)
    model.eval()
    pcm_op = PhotocurrentMapOp(
        j_order=8, s_indices=indices, normalized=True, device=device
    )
    data = make_pcm_forward_data(
        config=config,
        op=pcm_op,
        measurement=torch.from_numpy(y).to(device),
        key=0,
        sampling_ratio=0.2,
        coarse_j=6,
        device=device,
    )
    assert bool((data.model_input.initial_image == 0).all())
    # A is a partial orthonormal Hadamard transform, and grad and E each
    # have operator norms bounded by sqrt(8). With K=(A u, grad u-v, E v),
    # the conservative inequality ||K||^2 <= 17 follows from
    # ||a-b||^2 <= 2||a||^2+2||b||^2.
    assert TAU * SIGMA * 17 < 1.0
    OUT.mkdir(parents=True, exist_ok=True)
    return model, data, gt, y, indices, admm, utv, rho, row


def solve(model, data, gt, a1, a0, iters, stage):
    assert 0 < a1 and 0 < a0 and 1 <= iters <= 12000
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        rec = model(
            input_image=None,
            initial_image=data.model_input.initial_image,
            measurement=data.measurement,
            key=0,
            forward_op=data.forward_op,
            lambda_reg=(float(a1), float(a0)),
            max_iterations=iters,
            use_checkpoint=False,
        )
    torch.cuda.synchronize()
    duration = perf_counter() - start
    im = rec.squeeze().detach().cpu().numpy().astype(np.float32)
    assert im.shape == gt.shape and np.isfinite(im).all()
    row = {
        "stage": stage,
        "alpha1": a1,
        "alpha0": a0,
        "ratio": a0 / a1,
        "iterations": iters,
        "runtime_s": duration,
        **measure(im, gt),
    }
    return row, im


def key(a1, a0, iters):
    return (round(float(a1), 12), round(float(a0), 12), int(iters))


def evaluate_items(items, output, stage, model, data, gt, save_images=False):
    path = OUT / output
    records = read_existing(path)
    known = {key(r["alpha1"], r["alpha0"], r["iterations"]) for r in records}
    unique = {}
    for a1, a0, iters in items:
        unique.setdefault(key(a1, a0, iters), (float(a1), float(a0), int(iters)))
    for i, (a1, a0, iters) in enumerate(unique.values(), 1):
        if key(a1, a0, iters) in known:
            continue
        result, im = solve(model, data, gt, a1, a0, iters, stage)
        records.append(result)
        known.add(key(a1, a0, iters))
        write_csv(path, records)
        if save_images:
            folder = OUT / "candidates_4096"
            folder.mkdir(exist_ok=True)
            np.save(folder / f"a1_{a1:.9g}_a0_{a0:.9g}_n{iters}.npy", im)
        print(
            "CANDIDATE",
            stage,
            f"{i}/{len(unique)}",
            "a1",
            f"{a1:.7g}",
            "a0",
            f"{a0:.7g}",
            "PSNR",
            f"{result['psnr']:.5f}",
            "SSIM",
            f"{result['ssim']:.4f}",
            "HaarPSI",
            f"{result['haarpsi']:.4f}",
            "time",
            f"{result['runtime_s']:.2f}",
            flush=True,
        )
    assert len(known) == len(unique)
    rank = sorted(records, key=lambda r: -float(r["psnr"]))
    print(
        "PHASE_COMPLETE",
        stage,
        "count",
        len(records),
        "best",
        {
            x: rank[0][x]
            for x in ("alpha1", "alpha0", "iterations", "psnr", "ssim", "haarpsi")
        },
        flush=True,
    )
    return records


def search(model, data, gt):
    coarse_items = [(a1, a1 * ratio, 512) for a1 in COARSE_A1 for ratio in COARSE_RATIO]
    coarse = evaluate_items(coarse_items, "coarse_512.csv", "coarse", model, data, gt)
    ranked = sorted(coarse, key=lambda x: -float(x["psnr"]))
    seeds = ranked[:6]
    fine_items = []
    for r in seeds:
        a1, a0 = float(r["alpha1"]), float(r["alpha0"])
        for f1 in (2 ** (-0.5), 1.0, 2**0.5):
            for f0 in (2 ** (-0.5), 1.0, 2**0.5):
                fine_items.append((a1 * f1, a0 * f0, 512))
    fine = evaluate_items(fine_items, "fine_512.csv", "fine", model, data, gt)
    all_seen = sorted(coarse + fine, key=lambda r: -float(r["psnr"]))
    top = []
    used = set()
    for r in all_seen:
        k = key(r["alpha1"], r["alpha0"], 512)
        if k in used:
            continue
        top.append(r)
        used.add(k)
        if len(top) == 10:
            break
    (OUT / "top512.json").write_text(json.dumps(top, indent=2) + "\n")
    print("SEARCH_DONE", len(coarse), len(fine), len(top), flush=True)


def validate(model, data, gt):
    top = json.loads((OUT / "top512.json").read_text())
    assert len(top) == 10
    first = [(float(r["alpha1"]), float(r["alpha0"]), 2048) for r in top]
    eval2048 = evaluate_items(
        first, "validate_2048.csv", "validate2048", model, data, gt
    )
    best2048 = sorted(eval2048, key=lambda r: -float(r["psnr"]))[:4]
    second = [(float(r["alpha1"]), float(r["alpha0"]), 4096) for r in best2048]
    eval4096 = evaluate_items(
        second, "validate_4096.csv", "validate4096", model, data, gt, save_images=True
    )
    best4096 = sorted(eval4096, key=lambda r: -float(r["psnr"]))[:2]
    third = [(float(r["alpha1"]), float(r["alpha0"]), 8192) for r in best4096]
    eval8192 = evaluate_items(
        third, "validate_8192.csv", "validate8192", model, data, gt, save_images=False
    )
    winner = sorted(eval8192, key=lambda r: -float(r["psnr"]))[0]
    a1, a0 = float(winner["alpha1"]), float(winner["alpha0"])
    result, image = solve(model, data, gt, a1, a0, 8192, "selected_best")
    np.testing.assert_allclose(result["psnr"], float(winner["psnr"]), atol=2e-5, rtol=0)
    np.save(OUT / "best_tgv.npy", image)
    (OUT / "selected.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "search": "coarse 35 + fine (local around top 6), 512 iters; top10 2048, top4 4096, top2 8192",
                "objective": "ground-truth PSNR, single image oracle",
                "model": "LION AdaptiveTGVPCMNetwork, MRPro operator / prox PDHG",
                "regularisation": "component-wise L1 second-order TGV",
                "fidelity": "0.5 ||Ax-y||_2^2 on normalized PCM operator",
                "tau": TAU,
                "sigma": SIGMA,
                "theta": THETA,
                "selected": result,
                "best512": max(
                    json.loads((OUT / "top512.json").read_text()),
                    key=lambda x: float(x["psnr"]),
                ),
                "best2048": max(eval2048, key=lambda x: float(x["psnr"])),
                "best4096": max(eval4096, key=lambda x: float(x["psnr"])),
                "best8192": winner,
                "frozen_measurement_sha256": sha(
                    data.measurement.detach().cpu().numpy()
                ),
                "torch_version": torch.__version__,
                "method_warning": "parameter chosen using GT; no unbiased generalization claim",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(
        "VALIDATION_DONE",
        "best_alpha1",
        a1,
        "best_alpha0",
        a0,
        "PSNR",
        result["psnr"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("search", "validate"), required=True)
    args = p.parse_args()
    model, data, gt, *_ = setup()
    if args.phase == "search":
        search(model, data, gt)
    else:
        validate(model, data, gt)


if __name__ == "__main__":
    main()
