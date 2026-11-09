"""Oracle scalar TV parameter sweep for frozen Perovskite_0292.

Use existing LION AdaptiveTVPCMNetwork and its MRPro-backed PDHG solver,
matching scalar TGV's fixed normalized PCM data fidelity, zero initialization
and PDHG step sizes. Tune scalar lambda on GT PSNR for this case only.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import load_pcm_sapm_supervised_config_from_path
from LION.pcm.pcm_sapm import make_pcm_forward_data
from LION.utils.metrics import compute_metrics

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_scalar_tv_oracle_20261109_v1"
FROZEN = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
CONFIG = ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/requested_config.yaml"
BENCH = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
TV_ARCHIVED = (
    ROOT / "tune1_ternary_20261109_v1/test45/baselines/raw/tv/Perovskite_0292.npy"
)
TAU = SIGMA = 0.18
INITIAL_LAMBDAS = (
    0.0001,
    0.0003,
    0.0007,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.007,
    0.01,
    0.011,
    0.013,
    0.016,
    0.02,
    0.025,
    0.03,
    0.04,
    0.06,
    0.08,
    0.12,
    0.2,
)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def quality(x, gt):
    vals = compute_metrics(
        recon=torch.from_numpy(x)[None, None].float(),
        target=torch.from_numpy(gt)[None, None].float(),
        data_range=1.0,
    )
    return {k: float(vals[k]) for k in METRICS}


def model_and_data():
    OUT.mkdir(parents=True, exist_ok=True)
    import LION.pcm.pcm_config as mod

    assert "source_9188b07" in str(Path(mod.__file__).resolve()), mod.__file__
    with np.load(FROZEN, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
    with BENCH.open(newline="") as f:
        old = next(
            r
            for r in csv.DictReader(f)
            if r["category"] == "Perovskite" and int(r["sample_id"]) == 292
        )
    assert digest(y) == old["measurement_sha256"]
    assert digest(indices) == old["pattern_sha256"]
    tv0 = np.load(TV_ARCHIVED, allow_pickle=False)
    assert tv0.shape == gt.shape == (256, 256)
    np.testing.assert_allclose(
        float(np.mean((tv0 - gt) ** 2)), float(old["tv_mse"]), rtol=5e-5, atol=1e-7
    )
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    torch.set_num_threads(3)
    cfg = load_pcm_sapm_supervised_config_from_path(CONFIG)
    c = deepcopy(cfg.model)
    c.pdhg.regularisation = "TV"
    c.pdhg.primal_stepsize = TAU
    c.pdhg.dual_stepsize = SIGMA
    assert c.pdhg.theta == 1.0
    model = AdaptiveTVPCMNetwork(config=c, lambda_map_network=None).to(dev)
    model.eval()
    op = PhotocurrentMapOp(j_order=8, s_indices=indices, normalized=True, device=dev)
    data = make_pcm_forward_data(
        config=cfg,
        op=op,
        measurement=torch.from_numpy(y).to(dev),
        key=0,
        sampling_ratio=0.2,
        coarse_j=6,
        device=dev,
    )
    assert bool((data.model_input.initial_image == 0).all())
    return model, data, gt, y, indices, old


def run_tv(model, data, gt, lam, iters):
    assert lam > 0 and iters > 0
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        recon = model(
            input_image=None,
            initial_image=data.model_input.initial_image,
            measurement=data.measurement,
            key=data.key,
            forward_op=data.forward_op,
            lambda_reg=float(lam),
            max_iterations=int(iters),
            use_checkpoint=False,
        )
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    x = recon.squeeze().detach().cpu().numpy().astype(np.float32)
    assert x.shape == gt.shape and np.isfinite(x).all()
    return {
        "lambda": float(lam),
        "iterations": int(iters),
        "elapsed_s": elapsed,
        **quality(x, gt),
    }, x


def load_records(name):
    f = OUT / name
    if not f.is_file():
        return []
    with f.open(newline="") as fp:
        return list(csv.DictReader(fp))


def evaluate(name, items, model, data, gt):
    records = load_records(name)
    seen = {(round(float(r["lambda"]), 12), int(r["iterations"])) for r in records}
    candidates = {
        (round(float(lam), 12), int(n)): (float(lam), int(n)) for lam, n in items
    }
    for lam, n in candidates.values():
        key = (round(lam, 12), n)
        if key in seen:
            continue
        result, _ = run_tv(model, data, gt, lam, n)
        records.append(result)
        seen.add(key)
        path = OUT / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        print(
            "TV_EVAL",
            name,
            len(records),
            f"lambda={lam:.8f}",
            f"iters={n}",
            f"PSNR={result['psnr']:.6f}",
            f"SSIM={result['ssim']:.6f}",
            f"HaarPSI={result['haarpsi']:.6f}",
            f"runtime={result['elapsed_s']:.2f}",
            flush=True,
        )
    assert len(seen) == len(candidates)
    return sorted(records, key=lambda z: -float(z["psnr"]))


def main():
    model, data, gt, y, indices, bench = model_and_data()
    # First search a broad interval at exactly the C60 iteration count.
    coarse = evaluate(
        "coarse_512.csv", [(lam, 512) for lam in INITIAL_LAMBDAS], model, data, gt
    )
    lbest = float(coarse[0]["lambda"])
    # Dense local search around the highest observed value, including all
    # coarse neighbours and covering roughly a factor of 1.8 on each side.
    refined = sorted(set(float(v) for v in np.geomspace(lbest / 1.8, lbest * 1.8, 33)))
    fine = evaluate("fine_512.csv", [(lam, 512) for lam in refined], model, data, gt)
    merged = {}
    for row in coarse + fine:
        merged[round(float(row["lambda"]), 12)] = row
    ranked = sorted(merged.values(), key=lambda z: -float(z["psnr"]))
    leaders = ranked[:8]
    for n, limit in [(2048, 8), (4096, 4), (8192, 3)]:
        selected = leaders[:limit]
        records = evaluate(
            f"validate_{n}.csv",
            [(float(r["lambda"]), n) for r in selected],
            model,
            data,
            gt,
        )
        if n < 8192:
            # At larger iterations, preserve the best candidates rather
            # than assuming the best 512-iteration ranking remains optimal.
            leaders = records
        else:
            last = records
    best = last[0]
    lam = float(best["lambda"])
    result, x = run_tv(model, data, gt, lam, 8192)
    np.testing.assert_allclose(result["psnr"], float(best["psnr"]), rtol=0, atol=2e-5)
    np.save(OUT / "tv_best_scalar.npy", x)
    protocol = {
        "case": "Perovskite_0292",
        "status": "complete",
        "method": "LION AdaptiveTVPCMNetwork MRPro-backed PDHG",
        "objective": "0.5||Au-y||_2^2 + lambda||grad u||_1, componentwise L1",
        "oracle": "lambda chosen on Perovskite_0292 ground-truth PSNR",
        "fidelity": "same frozen 20% NPL-default noisy differential Hadamard measurement as TGV2",
        "measurement_sha256": digest(y),
        "indices_sha256": digest(indices),
        "initialization": "zero",
        "tau": TAU,
        "sigma": SIGMA,
        "theta": 1.0,
        "stages": {
            "coarse_512": len(coarse),
            "fine_512": len(fine),
            "validate_2048": len(load_records("validate_2048.csv")),
            "validate_4096": len(load_records("validate_4096.csv")),
            "validate_8192": len(load_records("validate_8192.csv")),
        },
        "best": result,
        "archived_val_tuned_TV": {
            "lambda": 0.011,
            "iterations": 1024,
            "psnr": float(bench["tv_psnr"]),
            "ssim": float(bench["tv_ssim"]),
            "haarpsi": float(bench["tv_haarpsi"]),
        },
        "caution": "finite grid + refinement; not mathematically certified global optimum",
    }
    (OUT / "summary.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        "TV_COMPLETE",
        result,
        "vs_previous_validation",
        protocol["archived_val_tuned_TV"],
        flush=True,
    )


if __name__ == "__main__":
    main()
