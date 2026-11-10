"""Perovskite 0292 oracle tuning of scalar TGV-3 using shared LION/MRPro PDHG.

No new PDHG loop. Full-tensor anisotropic L1 TGV-3 with 3 positive weights.
All stages share the EXACT frozen 20% default-NPL PCM acquisition.
Parameter selection maximizes ground-truth PSNR for this one image.

Run:
    python scripts/experiments/pcm/perovskite0292_tgv3_oracle_20261110.py --stage coarse
    python scripts/experiments/pcm/perovskite0292_tgv3_oracle_20261110.py --stage refine
    python scripts/experiments/pcm/perovskite0292_tgv3_oracle_20261110.py --stage validate
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from LION.config.sapm_config import SAPMModelConfig, AdaptivePDHGConfig
from LION.models.AdaptiveTGV3PCMNetwork import AdaptiveTGV3PCMNetwork
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_sapm import PhotocurrentMapMRProOp
from LION.utils.metrics import compute_metrics

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_scalar_tgv3_oracle_20261110_v1"
FROZEN = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
METRIC450 = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
TAU = SIGMA = 0.18
COARSE_ALPHA2 = (0.005, 0.01, 0.02, 0.04, 0.08)
COARSE_ALPHA1 = (0.0025, 0.005, 0.01, 0.02, 0.04)
COARSE_ALPHA0 = (0.0008, 0.002, 0.005, 0.0125, 0.03)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def key(a2, a1, a0, n):
    return (round(float(a2), 12), round(float(a1), 12), round(float(a0), 12), int(n))


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, records):
    assert records
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def quality(pred, gt):
    res = compute_metrics(
        recon=torch.from_numpy(pred)[None, None],
        target=torch.from_numpy(gt)[None, None],
        data_range=1.0,
    )
    return {m: float(res[m]) for m in METRICS}


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "A GPU is required for the experimental sweep."
    with np.load(FROZEN, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
    with METRIC450.open(newline="") as f:
        baseline = next(
            x
            for x in csv.DictReader(f)
            if x["category"] == "Perovskite" and int(x["sample_id"]) == 292
        )
    assert digest(y) == baseline["measurement_sha256"]
    assert digest(indices) == baseline["pattern_sha256"]
    assert y.shape == (1, 1, 13107) and gt.shape == (256, 256)
    cfg = SAPMModelConfig(
        name="Perovskite0292-TGV3-scalar-oracle",
        input_image="zeros",
        initial_image="zeros",
        n_iters=8192,
        backend="new",
        cnn=None,
        pdhg=AdaptivePDHGConfig(
            regularisation="TGV3",
            primal_stepsize=TAU,
            dual_stepsize=SIGMA,
            initial_scale_factor_raw=0.1,
            scale_factor_softplus_beta=1.0,
            lambda_softplus_beta=1.0,
            fd_pad_mode=["zeros", "zeros"],
            fd_dim=[-2, -1],
            theta=1.0,
        ),
    )
    model = AdaptiveTGV3PCMNetwork(cfg, None).to(device).eval()
    forward = PhotocurrentMapMRProOp(
        PhotocurrentMapOp(
            j_order=8,
            s_indices=indices,
            normalized=True,
            device=device,
        )
    )
    measurement = torch.from_numpy(y).to(device)
    initial = torch.zeros((1, 1, 256, 256), dtype=measurement.dtype, device=device)
    print(
        "FROZEN_DATA",
        digest(y),
        digest(indices),
        "GPU",
        torch.cuda.get_device_name(0),
        "params",
        "tau=sigma=0.18",
        flush=True,
    )
    return model, forward, initial, measurement, gt, y, indices


def run(model, forward, initial, measurement, gt, a2, a1, a0, n):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = (
            model(
                input_image=None,
                initial_image=initial,
                measurement=measurement,
                key=0,
                forward_op=forward,
                lambda_reg=(float(a2), float(a1), float(a0)),
                max_iterations=int(n),
                use_checkpoint=False,
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert image.shape == gt.shape and np.isfinite(image).all()
    return {
        "alpha2": float(a2),
        "alpha1": float(a1),
        "alpha0": float(a0),
        "iterations": int(n),
        "seconds": elapsed,
        **quality(image, gt),
    }, image


def evaluate(filename, items, model, forward, initial, measurement, gt):
    path = OUT / filename
    records = read_rows(path)
    known = {
        key(r["alpha2"], r["alpha1"], r["alpha0"], r["iterations"]) for r in records
    }
    todo = {}
    for a2, a1, a0, n in items:
        todo[key(a2, a1, a0, n)] = (a2, a1, a0, n)
    for i, (k, (a2, a1, a0, n)) in enumerate(todo.items(), 1):
        if k in known:
            continue
        r, image = run(model, forward, initial, measurement, gt, a2, a1, a0, n)
        records.append(r)
        known.add(k)
        write_rows(path, records)
        print(
            "EVAL",
            filename,
            f"{len(records)}/{len(todo)}",
            f"a2={a2:.7g}",
            f"a1={a1:.7g}",
            f"a0={a0:.7g}",
            f"n={n}",
            f"PSNR={r['psnr']:.5f}",
            f"SSIM={r['ssim']:.4f}",
            f"HaarPSI={r['haarpsi']:.4f}",
            f"s={r['seconds']:.2f}",
            flush=True,
        )
    assert all(k in known for k in todo)
    ranked = sorted(records, key=lambda r: -float(r["psnr"]))
    best = ranked[0]
    print(
        "PHASE_COMPLETE",
        filename,
        "evaluated",
        len(records),
        "best",
        [(k, best[k]) for k in ("alpha2", "alpha1", "alpha0", "psnr")],
        flush=True,
    )
    return ranked


def top_diverse(rows, n):
    # Log-domain diversity avoids putting every candidate on nearly
    # identical parameter combinations.
    chosen = []
    for r in rows:
        vector = np.log([float(r["alpha2"]), float(r["alpha1"]), float(r["alpha0"])])
        if all(np.linalg.norm(vector - other) > 0.23 for _, other in chosen):
            chosen.append((r, vector))
            if len(chosen) >= n:
                break
    return [r for r, _ in chosen]


def coarse(ctx):
    model, forward, initial, measurement, gt, *_ = ctx
    from itertools import product

    items = [(*v, 512) for v in product(COARSE_ALPHA2, COARSE_ALPHA1, COARSE_ALPHA0)]
    return evaluate("coarse_512.csv", items, model, forward, initial, measurement, gt)


def expand(ctx):
    model, forward, initial, measurement, gt, *_ = ctx
    from itertools import product

    # Coarse PSNR peak reached upper bound alpha0=.03; extend this
    # dimension around the strongest alpha2 and alpha1 values.
    grid = [
        (*v, 512)
        for v in product((0.02, 0.04, 0.08), (0.005, 0.01, 0.02), (0.05, 0.08, 0.14))
    ]
    return evaluate("coarse_512.csv", grid, model, forward, initial, measurement, gt)


def refine(ctx):
    model, forward, initial, measurement, gt, *_ = ctx
    coarse_rows = read_rows(OUT / "coarse_512.csv")
    assert len(coarse_rows) >= 125
    ranked = sorted(coarse_rows, key=lambda r: -float(r["psnr"]))
    seeds = top_diverse(ranked, 12)
    # Force a previously tested baseline into 2048-iteration validation.
    reference = (0.02, 0.01, 0.005)
    if not any(
        key(r["alpha2"], r["alpha1"], r["alpha0"], 512) == key(*reference, 512)
        for r in seeds
    ):
        seeds.append(
            {"alpha2": reference[0], "alpha1": reference[1], "alpha0": reference[2]}
        )
    stage2048 = evaluate(
        "coarse_shortlist_2048.csv",
        [
            (float(r["alpha2"]), float(r["alpha1"]), float(r["alpha0"]), 2048)
            for r in seeds
        ],
        model,
        forward,
        initial,
        measurement,
        gt,
    )
    local_seeds = top_diverse(stage2048, 3)
    factors = (0.8, 1.0, 1.25)
    from itertools import product

    local = [
        (
            float(seed["alpha2"]) * f2,
            float(seed["alpha1"]) * f1,
            float(seed["alpha0"]) * f0,
            1024,
        )
        for seed in local_seeds
        for f2, f1, f0 in product(factors, repeat=3)
    ]
    (OUT / "local_seeds.json").write_text(json.dumps(local_seeds, indent=2) + "\n")
    return evaluate("local_1024.csv", local, model, forward, initial, measurement, gt)


def validate(ctx):
    model, forward, initial, measurement, gt, y, indices = ctx
    local = read_rows(OUT / "local_1024.csv")
    shortlist = read_rows(OUT / "coarse_shortlist_2048.csv")
    assert len(local) >= 27 and len(shortlist) >= 12
    ranked = sorted(local, key=lambda r: -float(r["psnr"]))
    candidate = top_diverse(ranked, 6)
    # Also allow the best 2048 result from the initial grid.
    seed = sorted(shortlist, key=lambda r: -float(r["psnr"]))[0]
    if key(seed["alpha2"], seed["alpha1"], seed["alpha0"], 4096) not in {
        key(r["alpha2"], r["alpha1"], r["alpha0"], 4096) for r in candidate
    }:
        candidate.append(seed)
    validation4096 = evaluate(
        "shortlist_4096.csv",
        [
            (float(r["alpha2"]), float(r["alpha1"]), float(r["alpha0"]), 4096)
            for r in candidate
        ],
        model,
        forward,
        initial,
        measurement,
        gt,
    )
    finalists = top_diverse(validation4096, 3)
    validation8192 = evaluate(
        "finalists_8192.csv",
        [
            (float(r["alpha2"]), float(r["alpha1"]), float(r["alpha0"]), 8192)
            for r in finalists
        ],
        model,
        forward,
        initial,
        measurement,
        gt,
    )
    best = validation8192[0]
    params = (float(best["alpha2"]), float(best["alpha1"]), float(best["alpha0"]))
    row, image = run(model, forward, initial, measurement, gt, *params, 8192)
    np.testing.assert_allclose(row["psnr"], float(best["psnr"]), rtol=0, atol=3e-5)
    np.save(OUT / "tgv3_best_scalar.npy", image)
    summary = {
        "status": "complete",
        "case": "Perovskite_0292",
        "method": "scalar TGV-3, full-tensor anisotropic L1, existing LION MRPro PDHG",
        "data": "frozen 20% multilevel Hadamard, NPL-default PCM noise",
        "measurement_sha256": digest(y),
        "sampled_indices_sha256": digest(indices),
        "objective": "0.5||Au-y||^2 + alpha2||grad u-v||_1 + alpha1||E1 v-w||_1 + alpha0||E2 w||_1",
        "tau": TAU,
        "sigma": SIGMA,
        "theta": 1.0,
        "weights": params,
        "final_metrics": row,
        "counts": {
            "coarse_512": len(read_rows(OUT / "coarse_512.csv")),
            "coarse_shortlist_2048": len(shortlist),
            "local_1024": len(local),
            "shortlist_4096": len(validation4096),
            "finalists_8192": len(validation8192),
        },
        "selection": "ground-truth PSNR on one image, not unbiased or globally certified",
        "caveat": "Existing shared PDHG uses interpolation relaxation, not textbook CP extrapolation.",
        "branch": "feature/pcm-tgv3-mrpro-20261109",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "TGV3_VALIDATION_COMPLETE",
        json.dumps(
            {
                "best": params,
                "PSNR": row["psnr"],
                "SSIM": row["ssim"],
                "HaarPSI": row["haarpsi"],
                "counts": summary["counts"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("coarse", "expand", "refine", "validate")
    )
    args = parser.parse_args()
    ctx = setup()
    {"coarse": coarse, "expand": expand, "refine": refine, "validate": validate}[
        args.stage
    ](ctx)
