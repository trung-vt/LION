"""Replay five preselected PCM 20%-NPL test reconstructions, then render 9-method comparison.

Case selection was fixed before inspecting reconstruction metrics, in
test450/v1/visual5/selection.json. Original 450 acquisitions are reused bitwise.
The final plot shows one ground-truth panel and nine method labels including
ADMM's numerically identical two configurations. All metrics are raw-image scores.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.reconstruct import ReconInput, ScalarTVReconFn, ScalarTVParameter
from LION.pcm.spgl1_bpdn import SPGL1BPDNReconFn
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.experiments.pcm_presets.presets import QUICK_METHODS, NOISY_PCM_METHODS
from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import (
    live_admm,
    oldbench,
)

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
BASE = ROOT / "pcm_utv_fixed20_npl_default_20261103"
SSDU = ROOT / "ssdu_pnp_tuning_20261108_v1"
OUT = SSDU / "test450/v1/visual5"
RAW = SSDU / "test450/v1/raw"
CSV = SSDU / "test450/v1/merged/all_450_11_method_metrics.csv"
SOURCE = OUT / "selection.json"
COLS = (
    ("target", "Target"),
    ("zero_filled", "Zero-filled"),
    ("spgl1_bp", "SPGL1-BP"),
    ("spgl1_bpdn", "SPGL1-BPDN"),
    ("pnp_admm_oracle", "PnP-ADMM oracle"),
    ("gs_pnp_oracle", "GS-PnP oracle (Step 6)"),
    ("tv", "Scalar TV"),
    ("utv_c60", "UTV-C60"),
    ("pnp_admm_ssdu5", "PnP-ADMM tuned-5"),
    ("gs_pnp_ssdu5", "GS-PnP tuned-5"),
)
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_selection():
    s = json.loads(SOURCE.read_text())
    c = [(v["category"], int(v["sample_id"])) for v in s["chosen_cases"]]
    assert len(c) == 5 and len(set(c)) == 5
    assert [x[0] for x in c] == [
        "CIGS",
        "Organic",
        "Perovskite",
        "Reference",
        "Silicone",
    ]
    assert c[0] == ("CIGS", 279)
    return c, s


def read_metrics():
    with CSV.open(newline="") as f:
        v = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    assert len(v) == 450
    return v


def load_cached_case(cat, sid, merged):
    p = BASE / "cases" / f"{cat}_{sid:04d}.npz"
    assert p.is_file(), p
    with np.load(p, allow_pickle=False) as a:
        required = [
            "ground_truth",
            "measurement",
            "sampled_indices",
            "zero_filled",
            "spgl1_bp",
            "C",
        ]
        old = {k: a[k].copy() for k in required}
    assert old["ground_truth"].shape == (256, 256)
    ref = merged[(cat, sid)]
    assert digest(old["measurement"]) == ref["measurement_sha256"], (
        cat,
        sid,
        "measurement mismatch",
    )
    assert digest(old["sampled_indices"]) == ref["pattern_sha256"], (
        cat,
        sid,
        "mask mismatch",
    )
    return old


def initial_gpu():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    assert torch.cuda.is_available()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    return torch.device("cuda:0")


def calc_scores(x, gt):
    tensor = torch.from_numpy(np.ascontiguousarray(x))[None, None]
    truth = torch.from_numpy(np.ascontiguousarray(gt))[None, None]
    dr = float(gt.max() - gt.min())
    return oldbench.score(tensor, truth, dr)


def replay():
    device = initial_gpu()
    cases, selection = load_selection()
    rows = read_metrics()
    drunet = (
        build_grayscale_denoiser(
            deepcopy(QUICK_METHODS["pnp_admm_drunet"].denoiser), device=device
        )
        .eval()
        .requires_grad_(False)
    )
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
    overall = []
    for cat, sid in cases:
        dest = OUT / "arrays" / f"{cat}_{sid:04d}.npz"
        audit = OUT / "audits" / f"{cat}_{sid:04d}.json"
        if dest.exists() and audit.exists():
            print("REUSE_VISUAL5", cat, sid, flush=True)
            continue
        old = load_cached_case(cat, sid, rows)
        gt = old["ground_truth"]
        y = torch.as_tensor(old["measurement"], device=device)
        op = PhotocurrentMapOp(
            j_order=8, s_indices=old["sampled_indices"], normalized=True, device=device
        )
        ri = ReconInput(op=op, y=y, x0=op.pseudo_inv(y), i0=None)
        arrays = {
            "target": gt.copy(),
            "zero_filled": old["zero_filled"].copy(),
            "spgl1_bp": old["spgl1_bp"].copy(),
            "utv_c60": old["C"].copy(),
        }
        timing = {}
        with torch.no_grad():
            t0 = time.perf_counter()
            bv = bpdn(ri).reconstruction
            arrays["spgl1_bpdn"] = (
                bv.detach().cpu().numpy().squeeze().astype(np.float32)
            )
            timing["spgl1_bpdn"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ad = live_admm(op, y, drunet)
            arrays["pnp_admm_oracle"] = (
                ad["x"].detach().cpu().numpy().squeeze().astype(np.float32)
            )
            timing["pnp_admm_oracle"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            tvv = tv(ri).reconstruction
            arrays["tv"] = tvv.detach().cpu().numpy().squeeze().astype(np.float32)
            timing["tv"] = time.perf_counter() - t0
        arrays["pnp_admm_ssdu5"] = arrays["pnp_admm_oracle"].copy()
        for key, m in [
            ("gs_pnp_oracle", "gs_oracle_step6"),
            ("gs_pnp_ssdu5", "gs_ssdu"),
        ]:
            p = RAW / m / f"{cat}_{sid:04d}.npy"
            assert p.is_file(), p
            arrays[key] = np.load(p, allow_pickle=False).copy()
        checked = {}
        for key, label in COLS:
            arr = arrays[key]
            assert arr.shape == (256, 256) and np.isfinite(arr).all()
            if key == "target":
                checked[key] = {"score": "ground truth"}
                continue
            expected = {met: float(rows[(cat, sid)][f"{key}_{met}"]) for met in METRICS}
            actual = calc_scores(arr, gt)
            delta = {met: abs(actual[met] - expected[met]) for met in METRICS}
            assert (
                delta["psnr"] < 0.12
                and delta["ssim"] < 0.012
                and delta["haarpsi"] < 0.015
            ), (cat, sid, key, delta)
            checked[key] = {
                "target_metrics": expected,
                "replayed_metrics": actual,
                "difference": delta,
            }
        dest.parent.mkdir(parents=True, exist_ok=True)
        audit.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        with dest.open("xb") as f:
            np.savez_compressed(f, **arrays)
        summary = {
            "case": {"category": cat, "sample_id": sid},
            "match_benchmark_measurement_sha256": digest(old["measurement"]),
            "match_benchmark_mask_sha256": digest(old["sampled_indices"]),
            "initial_random_select_seed": selection["seed"],
            "methods": checked,
            "replay_runtime_seconds": timing,
            "afmhot_cmap_max": 0.8,
            "all_metrics_unclipped": True,
        }
        audit.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(
            "REPLAY_VISUAL5",
            cat,
            sid,
            "PSNR_CHECKED",
            {
                k: round(v["difference"]["psnr"], 4)
                for k, v in checked.items()
                if k != "target"
            },
            flush=True,
        )
        overall.append({"category": cat, "sample_id": sid, "status": "verified"})
    print("REPLAY_VISUAL5_COMPLETE", overall, flush=True)


def render():
    cases, sel = load_selection()
    metrics = read_metrics()
    cmap = make_colormap("afmhot", 0.8)
    pdf = OUT / "PCM_5_Cases_9_Methods_with_Metrics.pdf"
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10.5, "pdf.fonttype": 42}
    )
    with PdfPages(
        pdf,
        metadata={"Title": "PCM 5 paired test cases: nine reconstructions and targets"},
    ) as out:
        for page, (cat, sid) in enumerate(cases, 1):
            array = OUT / "arrays" / f"{cat}_{sid:04d}.npz"
            assert array.exists(), array
            with np.load(array, allow_pickle=False) as a:
                images = {k: a[k].copy() for k, _ in COLS}
            row = metrics[(cat, sid)]
            fig, axs = plt.subplots(
                2, 5, figsize=(19.4, 10.8), facecolor="white", layout="constrained"
            )
            fig.suptitle(
                f"{cat}  |  image {sid:04d}  |  20% NPL-noisy PCM  |  identical acquisition for all methods",
                fontsize=17,
                weight="bold",
            )
            for (key, name), ax in zip(COLS, axs.flat):
                ax.imshow(
                    images[key], cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if key == "target":
                    subtitle = "PSNR inf | SSIM 1.0000\nHaarPSI 1.0000 | MSE 0"
                else:
                    psnr = float(row[f"{key}_psnr"])
                    ssim = float(row[f"{key}_ssim"])
                    hp = float(row[f"{key}_haarpsi"])
                    mse = float(row[f"{key}_mse"])
                    subtitle = (
                        f"PSNR {psnr:.2f} dB | SSIM {ssim:.4f}\n"
                        f"HaarPSI {hp:.4f} | MSE {mse:.2e}"
                    )
                ax.set_title(
                    name + "\n" + subtitle, fontsize=10.1, weight="medium", pad=6
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d4d4d8")
                    spine.set_linewidth(0.65)
            fig.text(
                0.5,
                0.018,
                f"{page}/5  |  AFM-hot capped at 0.8; fixed intensity [0,1]. "
                "The ADMM oracle and tuned-5 panels are identical by configuration.",
                fontsize=9.1,
                color="#52525b",
                ha="center",
            )
            out.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("RENDER_VISUAL", page, "/", len(cases), cat, sid, flush=True)
    print("VISUAL5_PDF_COMPLETE", pdf, pdf.stat().st_size, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["reconstruct", "render"])
    args = ap.parse_args()
    if args.stage == "reconstruct":
        replay()
    else:
        render()


if __name__ == "__main__":
    main()
