"""Per-image C60 post-softplus map multiplier ternary search for 23 bad-UTV cases.

Fixed 20% NPL default synthetic PCM acquisitions from the frozen 450-image
test. For cases without an original NPZ archive, deterministic replay must
match both measurement and sampling-mask SHA256 of the frozen benchmark.

Oracle search uses ground-truth PSNR. Ternary search's global correctness
is conditional on PSNR unimodality; width tolerance is in rho, not PSNR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
)
from LION.pcm.pcm_sapm import load_trained_pcm_checkpoint_model, make_pcm_forward_data
from LION.utils.metrics import compute_metrics

from scripts.experiments.pcm.fixed_utv_test_20261103 import benchmark as oldbench

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "bad_utv23_c60_ternary_1e-2_20261109_v1"
MANIFEST = (
    ROOT
    / "bad_images_for_utv_20261109_v3_23/bad_images_for_utv_dataset_manifest_23.json"
)
FROZEN = ROOT / "pcm_utv_fixed20_npl_default_20261103/cases"
ORIGINAL_14 = ROOT / "ten_methods_matched_450_20261105_v1/visual_14_paired_exact_v1/raw"
BASELINE_45 = ROOT / "tune1_ternary_20261109_v1/test45/baselines/raw"
NEW9_ADMM = ROOT / "bad_images_for_utv_20261109_v3_23/raw_new9"
SCORE450 = (
    ROOT / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)
C60_CONFIG = ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/requested_config.yaml"
C60_CHECKPOINT = (
    ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/checkpoints/best_val_epoch60.pt"
)
PINNED = ROOT / "pcm_utv_three_training_selection_20261103/source_9188b07"
LOW, HIGH, RHO_WIDTH = 1e-4, 10.0, 1e-2
DIAGNOSTIC_GRID = (LOW, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0, HIGH)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def score(x: np.ndarray, gt: np.ndarray) -> dict:
    image = torch.from_numpy(np.asarray(x)).float()[None, None]
    reference = torch.from_numpy(gt).float()[None, None]
    result = compute_metrics(recon=image, target=reference, data_range=1.0)
    return {k: float(result[k]) for k in METRICS}


def get_cases_and_table():
    m = json.loads(MANIFEST.read_text())
    cases = m["cases"]
    assert (
        len(cases) == 23 and len({(c["category"], c["sample_id"]) for c in cases}) == 23
    )
    assert {c["category"] for c in cases} == {
        "CIGS",
        "Perovskite",
        "Reference",
        "Silicone",
    }
    with SCORE450.open(newline="") as f:
        rows = list(csv.DictReader(f))
    table = {(r["category"], int(r["sample_id"])): r for r in rows}
    assert len(table) == len(rows) == 450
    assert all((c["category"], c["sample_id"]) in table for c in cases)
    return cases, table


def load_case(c, row, config, device):
    cat, sid = c["category"], int(c["sample_id"])
    png = oldbench.DATA / cat / f"image{sid:04d}.png"
    with Image.open(png) as image:
        gt = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    assert gt.shape == (256, 256) and np.isfinite(gt).all()
    old = FROZEN / f"{cat}_{sid:04d}.npz"
    if old.is_file():
        with np.load(old, allow_pickle=False) as z:
            y = z["measurement"].copy()
            indices = z["sampled_indices"].copy()
            stored_gt = z["ground_truth"].copy()
        np.testing.assert_array_equal(gt, stored_gt)
        acquisition_source = "stored NPZ acquisition"
        assert (ORIGINAL_14 / f"{cat}_{sid:04d}.npz").exists()
    else:
        image = torch.from_numpy(gt)[None, None].to(device)
        acq, indices = oldbench.acquisition(image, config, cat, sid, device)
        y = acq.measurement.detach().cpu().contiguous().numpy().copy()
        indices = np.asarray(indices, dtype=np.int64)
        acquisition_source = "replayed acquisition, exact frozen SHA256"
        assert not (ORIGINAL_14 / f"{cat}_{sid:04d}.npz").exists()

    assert y.dtype == np.float32 and indices.dtype == np.int64
    assert y.shape == (1, 1, 13107) and indices.shape == (13107,)
    assert digest(y) == row["measurement_sha256"], (
        cat,
        sid,
        "measurement hash mismatch",
    )
    assert digest(indices) == row["pattern_sha256"], (cat, sid, "indices hash mismatch")

    if old.is_file():
        with np.load(ORIGINAL_14 / f"{cat}_{sid:04d}.npz", allow_pickle=False) as z:
            admm = z["pnp_tuned"].copy()
            c60 = z["C"].copy()
            np.testing.assert_array_equal(z["ground_truth"], gt)
    else:
        admm = np.load(NEW9_ADMM / f"{cat}_{sid:04d}.npy", allow_pickle=False)
        c60 = np.load(
            BASELINE_45 / "utv_c60" / f"{cat}_{sid:04d}.npy", allow_pickle=False
        )
    for name, array, metric_key in (
        ("PnP-ADMM-tuned-5", admm, "pnp_admm_ssdu5"),
        ("C60-rho1", c60, "utv_c60"),
    ):
        assert array.shape == gt.shape and np.isfinite(array).all(), (cat, sid, name)
        mse = np.mean((array.astype(np.float64) - gt.astype(np.float64)) ** 2)
        np.testing.assert_allclose(
            mse,
            float(row[f"{metric_key}_mse"]),
            rtol=5e-5,
            atol=2e-8,
            err_msg=f"frozen raw array and benchmark metrics disagree: {cat} {sid} {name}",
        )
    return gt, y, indices, admm, c60, acquisition_source


def load_model(device):
    assert torch.cuda.is_available()
    assert PINNED.is_dir()
    import LION.pcm.pcm_sapm as pinned

    assert Path(pinned.__file__).resolve().is_relative_to(PINNED.resolve())
    config = load_pcm_sapm_supervised_config_from_path(C60_CONFIG)
    assert config.model.backend == "new"
    assert config.model.pdhg.regularisation == "TV"
    assert config.model.input_image == "cg"
    assert config.model.initial_image == "zeros"
    assert config.model.n_iters == 512 and config.model.cg.max_iters == 20
    assert config.uses_normalized_op
    assert config.model.cnn.init_filters == 56 and config.model.cnn.n_blocks == 3
    model, epoch, step = load_trained_pcm_checkpoint_model(
        config=config,
        cases=make_supervised_pcm_training_cases(config),
        checkpoint_path=C60_CHECKPOINT,
        device=device,
    )
    assert (epoch, step) == (60, 270000)
    model.eval().requires_grad_(False)
    print("C60_LOADED", torch.cuda.get_device_name(0), "epoch", epoch, flush=True)
    return model, config


def prepare_data(model, config, y, indices, device):
    op = PhotocurrentMapOp(j_order=8, s_indices=indices, normalized=True, device=device)
    data = make_pcm_forward_data(
        config=config,
        op=op,
        measurement=torch.from_numpy(y).to(device),
        key=0,
        sampling_ratio=0.2,
        coarse_j=6,
        device=device,
    )
    with torch.inference_mode():
        lam = model.estimate_lambda_map(data.model_input.input_image, 0).detach()
    assert lam.shape == (1, 1, 256, 256) and bool((lam > 0).all())
    return data, lam


def reconstruct(model, data, lam, rho):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(
            input_image=None,
            initial_image=data.model_input.initial_image,
            measurement=data.measurement,
            key=data.key,
            forward_op=data.forward_op,
            lambda_reg=lam * float(rho),
            max_iterations=512,
            use_checkpoint=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
    assert out.shape == (256, 256) and np.isfinite(out).all()
    return out, elapsed


def run_case(c, row, model, config, device):
    cat, sid = c["category"], int(c["sample_id"])
    key = f"{cat}_{sid:04d}"
    dst = OUT / "cases" / key
    dst.mkdir(parents=True, exist_ok=True)
    completed = dst / "summary.json"
    raw = dst / "reconstructions.npz"
    if completed.exists() and raw.exists():
        data = json.loads(completed.read_text())
        assert data["sample"] == key and data["search_rho_width"] <= RHO_WIDTH
        assert data["measurement_sha256"] == row["measurement_sha256"]
        with np.load(raw, allow_pickle=False) as z:
            assert z["rho_opt"].shape == () and z["optimal"].shape == (256, 256)
        print("CASE_RESUMED", key, data["rho_opt"], data["psnr_opt"], flush=True)
        return data

    gt, y, indices, admm, c60, source = load_case(c, row, config, device)
    data, lam = prepare_data(model, config, y, indices, device)
    records = {}
    steps = []

    def evaluate(rho, role="ternary"):
        rho = float(rho)
        assert LOW <= rho <= HIGH
        keyrho = rho.hex()
        if keyrho in records:
            return records[keyrho]
        image, elapsed = reconstruct(model, data, lam, rho)
        metrics = score(image, gt)
        result = {"rho": rho, "role": role, "seconds": elapsed, **metrics}
        records[keyrho] = result
        return result

    ref = evaluate(1.0, "rho_1_validate")
    recomputed, _ = reconstruct(model, data, lam, 1.0)
    baseline_gap = float(np.max(np.abs(recomputed - c60)))
    assert baseline_gap < 0.002, (key, baseline_gap)
    assert abs(ref["psnr"] - float(row["utv_c60_psnr"])) < 0.025, (key, ref)

    diagnostics = [evaluate(r, "diagnostic") for r in DIAGNOSTIC_GRID]
    lo, hi = LOW, HIGH
    while hi - lo > RHO_WIDTH:
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        a, b = evaluate(m1), evaluate(m2)
        if a["psnr"] < b["psnr"]:
            lo = m1
            decision = "move_left_up"
        elif a["psnr"] > b["psnr"]:
            hi = m2
            decision = "move_right_down"
        else:
            lo, hi = m1, m2
            decision = "equal_psnr"
        steps.append(
            {
                "iteration": len(steps) + 1,
                "m1": m1,
                "m2": m2,
                "psnr1": a["psnr"],
                "psnr2": b["psnr"],
                "lo": lo,
                "hi": hi,
                "width": hi - lo,
                "decision": decision,
            }
        )
        assert len(steps) <= 90
    assert hi - lo <= RHO_WIDTH
    bracket = [lo, lo + 0.25 * (hi - lo), (lo + hi) / 2, lo + 0.75 * (hi - lo), hi]
    probes = [evaluate(x, "final_bracket") for x in bracket]
    bracket_best = sorted(probes, key=lambda d: (-d["psnr"], d["rho"]))[0]
    global_best = sorted(records.values(), key=lambda d: (-d["psnr"], d["rho"]))[0]
    outside = global_best["rho"] < lo or global_best["rho"] > hi

    # Always choose the best PSNR among inspected candidates, even if a
    # nonunimodal objective places it outside the ternary bracket.
    selected = (
        global_best
        if outside and global_best["psnr"] > bracket_best["psnr"]
        else bracket_best
    )
    rho_opt = float(selected["rho"])
    best, _ = reconstruct(model, data, lam, rho_opt)
    opt_metrics = score(best, gt)
    assert abs(opt_metrics["psnr"] - selected["psnr"]) < 2e-5
    assert opt_metrics["psnr"] >= ref["psnr"] - 1e-5, key

    admm_metrics = {name: float(row[f"pnp_admm_ssdu5_{name}"]) for name in METRICS}
    c60_metrics = {name: float(row[f"utv_c60_{name}"]) for name in METRICS}
    summary = {
        "sample": key,
        "category": cat,
        "sample_id": sid,
        "measurement_sha256": digest(y),
        "indices_sha256": digest(indices),
        "acquisition_source": source,
        "rho_opt": rho_opt,
        "rho_bracket": [lo, hi],
        "search_rho_width": hi - lo,
        "rho_lower": LOW,
        "rho_upper": HIGH,
        "target_width": RHO_WIDTH,
        "iterations": len(steps),
        "evaluations": len(records),
        "diagnostic_count": len(DIAGNOSTIC_GRID),
        "best_probe_outside_bracket": outside,
        "selected_outside_bracket": rho_opt < lo or rho_opt > hi,
        "rho_1_recompute_baseline_max_abs_diff": baseline_gap,
        "rho1_metrics": c60_metrics,
        "optimal_metrics": opt_metrics,
        "admm_metrics": admm_metrics,
        "psnr_rho1": c60_metrics["psnr"],
        "psnr_opt": opt_metrics["psnr"],
        "psnr_admm": admm_metrics["psnr"],
        "psnr_gain_vs_rho1": opt_metrics["psnr"] - c60_metrics["psnr"],
        "psnr_gap_to_admm": opt_metrics["psnr"] - admm_metrics["psnr"],
        "map_sha256": digest(lam.squeeze().cpu().numpy()),
        "map_min": float(lam.min()),
        "map_max": float(lam.max()),
        "map_mean": float(lam.mean()),
        "case_observation": c.get("user_observation", ""),
        "diagnostic_best": max(diagnostics, key=lambda r: r["psnr"]),
        "unimodality_warning": "Ternary assumes PSNR unimodal in linear rho; no global optimum proof.",
    }
    # Atomic writes to permit safe resume after an interrupted job.
    with (dst / "evaluations.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(next(iter(records.values()))), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(records.values())
    with (dst / "iterations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(steps)
    np.savez_compressed(
        dst / "reconstructions.npz",
        gt=gt,
        admm=admm,
        c60_rho1=c60,
        optimal=best,
        rho_opt=np.float64(rho_opt),
        lambda_c60=lam.squeeze().cpu().numpy(),
    )
    (dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "CASE_DONE",
        key,
        f"rho={rho_opt:.6f}",
        f"PSNR={opt_metrics['psnr']:.5f}",
        f"gain={summary['psnr_gain_vs_rho1']:+.4f}",
        "width",
        f"{hi-lo:.5g}",
        "evals",
        len(records),
        "outside_bracket",
        summary["selected_outside_bracket"],
        flush=True,
    )
    return summary


def run_shard(shard, count):
    OUT.mkdir(parents=True, exist_ok=True)
    cases, table = get_cases_and_table()
    config_bench, ids = oldbench.config_and_ids()
    assert all(c["sample_id"] in ids for c in cases)
    torch.set_num_threads(3)
    device = torch.device("cuda:0")
    model, cfg = load_model(device)
    outcomes = []
    for index, case in enumerate(cases):
        if index % count != shard:
            continue
        key = (case["category"], case["sample_id"])
        summary = run_case(case, table[key], model, config_bench, device)
        outcomes.append(summary)
        print("PROGRESS", f"shard={shard}/{count}", f"{len(outcomes)}", flush=True)
    (OUT / f"shard_{shard}_of_{count}_completed.json").write_text(
        json.dumps(
            {
                "shard": shard,
                "count": count,
                "completed": [x["sample"] for x in outcomes],
            },
            indent=2,
        )
        + "\n"
    )
    print("SHARD_COMPLETE", shard, "case_count", len(outcomes), flush=True)


def preflight():
    cases, table = get_cases_and_table()
    cfg, ids = oldbench.config_and_ids()
    assert all(c["sample_id"] in ids for c in cases)
    dev = torch.device("cuda:0")
    for case in cases:
        cat, sid = case["category"], case["sample_id"]
        load_case(case, table[cat, sid], cfg, dev)
        print("PREFLIGHT_OK", f"{cat}_{sid:04d}", flush=True)
    print("PREFLIGHT_COMPLETE", len(cases), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        assert args.num_shards >= 1 and 0 <= args.shard < args.num_shards
        run_shard(args.shard, args.num_shards)
