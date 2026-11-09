"""Perovskite 0031: C60 learned TV-map global multiplier, frozen 20% NPL measurement.

Reuses archived source snapshot, checkpoint, solver settings, GT, and measurement.
Rho multiplies the *post-softplus* predicted map; nothing else changes.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
)
from LION.pcm.pcm_sapm import load_trained_pcm_checkpoint_model, make_pcm_forward_data
from LION.utils.metrics import compute_metrics
from LION.visualize.plotting_functions import make_colormap

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
BASE = ROOT / "pcm_utv_fixed20_npl_default_20261103/cases/Perovskite_0031.npz"
BENCHMARK_CSV = ROOT / "pcm_utv_fixed20_npl_default_20261103/all_450_metrics.csv"
CHECKPOINT = (
    ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/checkpoints/best_val_epoch60.pt"
)
CONFIG = ROOT / "pcm_utv_4090_3x56_winner_snapshot_20261103/requested_config.yaml"
SNAPSHOT = ROOT / "pcm_utv_three_training_selection_20261103/source_9188b07"
OUT = ROOT / "perovskite0031_utv_c60_rho_sweep_20261109_v2"
RHO = (0.01, 0.05, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 0.9, 0.95, 1.0)
METRICS = ("psnr", "ssim", "haarpsi", "mse")
MODEL_ITERATIONS = 512


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def digest_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quality(image, target):
    reconstruction = torch.from_numpy(np.asarray(image)).float()[None, None]
    reference = torch.from_numpy(np.asarray(target)).float()[None, None]
    r = compute_metrics(recon=reconstruction, target=reference, data_range=1.0)
    return {key: float(r[key]) for key in METRICS}


def validate_frozen():
    assert SNAPSHOT.is_dir(), SNAPSHOT
    import LION.pcm.pcm_sapm as pinned

    assert (
        Path(pinned.__file__).resolve().is_relative_to(SNAPSHOT.resolve())
    ), pinned.__file__
    with np.load(BASE, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        y = z["measurement"].copy()
        indices = z["sampled_indices"].copy()
        reference = z["C"].copy()
        zero = z["zero_filled"].copy()
    assert gt.shape == reference.shape == zero.shape == (256, 256)
    assert y.shape == (1, 1, 13107) and indices.shape == (13107,)
    assert float(gt.min()) == 0.0 and float(gt.max()) == 1.0
    with BENCHMARK_CSV.open(newline="") as f:
        row = next(
            x
            for x in csv.DictReader(f)
            if x["category"] == "Perovskite" and int(x["sample_id"]) == 31
        )
    assert digest(y) == row["measurement_sha256"], "Frozen acquisition mismatch"
    assert digest(indices) == row["pattern_sha256"], "Frozen mask mismatch"
    assert abs(quality(reference, gt)["psnr"] - float(row["C_psnr"])) < 0.002
    return gt, y, indices, reference, zero, row


def setup():
    assert torch.cuda.is_available()
    assert "4090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    gt, y, indices, reference, zero, row = validate_frozen()
    cfg = load_pcm_sapm_supervised_config_from_path(CONFIG)
    assert cfg.model.backend == "new" and cfg.model.pdhg.regularisation == "TV"
    assert cfg.model.input_image == "cg" and cfg.model.initial_image == "zeros"
    assert cfg.model.n_iters == MODEL_ITERATIONS and cfg.model.cg.max_iters == 20
    assert (
        cfg.uses_normalized_op
        and cfg.model.cnn.init_filters == 56
        and cfg.model.cnn.n_blocks == 3
    )
    device = torch.device("cuda:0")
    model, epoch, step = load_trained_pcm_checkpoint_model(
        config=cfg,
        cases=make_supervised_pcm_training_cases(cfg),
        checkpoint_path=CHECKPOINT,
        device=device,
    )
    assert epoch == 60 and step == 270000
    model.eval().requires_grad_(False)
    op = PhotocurrentMapOp(j_order=8, s_indices=indices, normalized=True, device=device)
    data = make_pcm_forward_data(
        config=cfg,
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
    return model, data, lam, gt, y, indices, reference, zero, row


def reconstruct(model, data, lam, rho):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        rec = model(
            input_image=None,
            initial_image=data.model_input.initial_image,
            measurement=data.measurement,
            key=data.key,
            forward_op=data.forward_op,
            lambda_reg=lam * rho,
            max_iterations=MODEL_ITERATIONS,
            use_checkpoint=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = rec.squeeze().cpu().numpy().astype(np.float32)
    assert out.shape == (256, 256) and np.isfinite(out).all()
    return out, elapsed


def save_figure(gt, images, rows):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}
    )
    cmap = make_colormap("afmhot", 0.8)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12.7), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.035, right=0.965, top=0.795, bottom=0.073, wspace=0.095, hspace=0.30
    )
    cells = [("GT", gt, None)] + [
        (f"rho = {rho:g}", images[rho], m) for rho, m in zip(RHO, rows)
    ]
    for ax, (name, im, m) in zip(axes.flat, cells):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if m is None:
            label = "Ground truth"
        else:
            label = (
                f"{name}\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.3f}\n"
                f"HaarPSI {m['haarpsi']:.3f}"
            )
        ax.set_title(
            label,
            fontsize=12 if m is None else 10.5,
            pad=7,
            fontweight="bold" if name in ("GT", "rho = 1") else "normal",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
    ax = axes.flat[13]
    ax.axis("off")
    table_rows = sorted(rows, key=lambda x: -x["psnr"])
    best = table_rows[0]
    anchor = next(x for x in rows if x["rho"] == 1.0)
    ax.text(
        0.03,
        0.92,
        "PSNR ranking",
        weight="bold",
        size=14,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(0.03, 0.7, f"Best rho: {best['rho']:g}", size=13, transform=ax.transAxes)
    ax.text(
        0.03, 0.57, f"Best PSNR: {best['psnr']:.3f} dB", size=12, transform=ax.transAxes
    )
    ax.text(
        0.03,
        0.44,
        f"rho=1 PSNR: {anchor['psnr']:.3f} dB",
        size=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.31,
        f"Change: {best['psnr']-anchor['psnr']:+.3f} dB",
        size=12,
        transform=ax.transAxes,
    )
    ax = axes.flat[14]
    ax.plot(
        [r["rho"] for r in rows],
        [r["psnr"] for r in rows],
        marker="o",
        linewidth=1.5,
        markersize=5,
    )
    ax.axvline(1, color="gray", linewidth=1, linestyle="--")
    ax.set_title("PSNR vs. map multiplier", size=11, pad=7)
    ax.set_xlabel("rho")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.23)
    ax.tick_params(labelsize=9)
    fig.suptitle(
        "Perovskite 0031 | UTV (C60) regularisation-map multiplier sweep",
        y=0.977,
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.922,
        "Same archived 20% Hadamard measurement with NPL-default noise | 512 PDHG iterations | C60 epoch 60",
        ha="center",
        size=11,
    )
    fig.text(
        0.5,
        0.025,
        "Only lambda(x) is changed: lambda_rho(x) = rho x lambda_C60(x). "
        "Metrics use original reconstructions; displayed intensities fixed to [0,1]. "
        "This is a per-image GT-guided sweep.",
        ha="center",
        size=9.7,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "Perovskite_0031_UTV_C60_Rho_Sweep.pdf"
    png = OUT / "Perovskite_0031_UTV_C60_Rho_Sweep.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf, png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, data, lam, gt, y, indices, reference, zero, row = setup()
    print(
        "CHECKPOINT_LOADED",
        {
            "epoch": 60,
            "measurement_sha256": digest(y),
            "indices_sha256": digest(indices),
            "lambda_min": float(lam.min()),
            "lambda_max": float(lam.max()),
            "lambda_mean": float(lam.mean()),
        },
        flush=True,
    )
    img_ref, elapsed = reconstruct(model, data, lam, 1.0)
    baseline_diff = float(np.max(np.abs(img_ref - reference)))
    baseline_metrics = quality(img_ref, gt)
    assert baseline_diff < 0.002, (
        "rho=1 does not reproduce the archived C60 reconstruction",
        baseline_diff,
    )
    assert abs(baseline_metrics["psnr"] - float(row["C_psnr"])) < 0.025
    print(
        "BASELINE_VALIDATED",
        baseline_diff,
        baseline_metrics,
        "time",
        elapsed,
        flush=True,
    )
    if args.baseline_only:
        return
    images = {}
    rows = []
    for rho in RHO:
        if rho == 1.0:
            rec = img_ref.copy()
            secs = elapsed
        else:
            rec, secs = reconstruct(model, data, lam, rho)
        images[rho] = rec
        stats = quality(rec, gt)
        rows.append({"rho": rho, **stats, "elapsed_seconds": round(secs, 4)})
        print("RHO_DONE", rho, stats, "time", round(secs, 3), flush=True)
    # Save unmodified per-rho reconstruction images to permit later inspection.
    np.savez_compressed(
        OUT / "reconstructions.npz",
        ground_truth=gt,
        zero_filled=zero,
        measurement=y,
        indices=indices,
        original_lambda_map=lam.squeeze().cpu().numpy(),
        **{f"rho_{rho:g}": v for rho, v in images.items()},
    )
    with (OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    protocol = {
        "case": "Perovskite_0031",
        "dataset": "frozen 450-image held-out test subset",
        "observation": "stored 20% multi-level Hadamard acquisition with NPL-default noise",
        "baseline_file": str(BASE),
        "baseline_file_sha256": digest_file(BASE),
        "measurement_sha256": digest(y),
        "sampled_indices_sha256": digest(indices),
        "checkpoint_file": str(CHECKPOINT),
        "checkpoint_sha256": digest_file(CHECKPOINT),
        "config_file": str(CONFIG),
        "config_sha256": digest_file(CONFIG),
        "snapshot_source": str(SNAPSHOT),
        "lambda_map_min": float(lam.min()),
        "lambda_map_max": float(lam.max()),
        "lambda_map_mean": float(lam.mean()),
        "lambda_map_sha256": digest(lam.squeeze().cpu().numpy()),
        "lambda_map_scaling": "post-softplus lambda estimated once from fixed CG input; scaled by rho",
        "rho_grid": list(RHO),
        "iterations": MODEL_ITERATIONS,
        "fixed": [
            "measurement",
            "pattern mask",
            "checkpoint",
            "CG input",
            "unscaled lambda",
            "initial image",
            "step sizes",
            "PDHG iterations",
        ],
        "baseline_max_abs_diff": baseline_diff,
        "baseline_metrics": baseline_metrics,
        "per_image_oracle_analysis": True,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    pdf, png = save_figure(gt, images, rows)
    assert pdf.stat().st_size > 10000 and png.stat().st_size > 10000
    print("SWEEP_COMPLETE", str(pdf), str(png), flush=True)


if __name__ == "__main__":
    main()
