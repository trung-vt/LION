"""Reconstruct original frozen epoch-60 UTV-C60 on test45 if uncached.

Use the old pinned 9188b07 source and epoch60 checkpoint. The test is
blocked until all 50 tune-1 selections are frozen; no test-based tuning.
"""
from __future__ import annotations

import sys
from pathlib import Path

PIN = Path(
    "/store/LION/ttv22/pcm/results/pcm/pcm_utv_three_training_selection_20261103/source_9188b07"
)
sys.path.insert(0, str(PIN))

import argparse
import csv
import hashlib
import json
import os
import numpy as np
import torch
from PIL import Image

from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
)
from LION.pcm.pcm_sapm import (
    load_trained_pcm_checkpoint_model,
    make_pcm_forward_data,
    forward_func_sapm_pcm,
)
from LION.pcm import pcm_sapm
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import oldbench
from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    BASE,
    OUTPUT,
    CATEGORIES,
    frozen_json,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen

R = BASE / "pcm_utv_4090_3x56_winner_snapshot_20261103"
CFG = R / "requested_config.yaml"
CKPT = R / "checkpoints/best_val_epoch60.pt"
PREV = (
    BASE / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
)


def main(category):
    assert category in CATEGORIES
    assert_all_frozen()
    assert CFG.exists() and CKPT.exists()
    assert Path(pcm_sapm.__file__).resolve().is_relative_to(PIN.resolve())
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    cf = load_pcm_sapm_supervised_config_from_path(CFG)
    assert (
        cf.model.n_iters == 512
        and cf.model.cnn.init_filters == 56
        and cf.model.cnn.n_blocks == 3
    )
    model, epoch, step = load_trained_pcm_checkpoint_model(
        config=cf,
        cases=make_supervised_pcm_training_cases(cf),
        checkpoint_path=CKPT,
        device=device,
    )
    assert epoch == 60
    model.eval().requires_grad_(False)
    test = [
        x
        for x in json.loads((OUTPUT / "prepared/test45.json").read_text())["cases"]
        if x["category"] == category
    ]
    assert len(test) == 9
    with PREV.open(newline="") as f:
        # Evaluate only after tuning is frozen.
        rows = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    config, ids = oldbench.config_and_ids()
    for row in test:
        sid = int(row["sample_id"])
        assert sid in ids
        path = OUTPUT / "test45/baselines/raw/utv_c60" / f"{category}_{sid:04d}.npy"
        report = (
            OUTPUT
            / "test45/baselines/provenance/utv_c60"
            / f"{category}_{sid:04d}.json"
        )
        if path.exists() and report.exists():
            assert json.loads(report.read_text())["model_checkpoint_sha256"] == digest(
                CKPT
            )
            continue
        png = oldbench.DATA / category / f"image{sid:04d}.png"
        x0 = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
        gt = torch.from_numpy((x0.astype(np.float64) / 255).astype(np.float32))[
            None, None
        ].to(device)
        truth = gt.detach().cpu()
        dr = float((gt.max() - gt.min()).item())
        acq, indices = oldbench.acquisition(gt, config, category, sid, device)
        assert (
            hashlib.sha256(acq.measurement.detach().cpu().numpy().tobytes()).hexdigest()
            == row["measurement_sha256"]
        )
        assert (
            hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
            == row["pattern_sha256"]
        )
        data = make_pcm_forward_data(
            config=cf,
            op=acq.op,
            measurement=acq.measurement,
            key=0,
            sampling_ratio=0.2,
            coarse_j=6,
            device=device,
        )
        with torch.no_grad():
            rec = forward_func_sapm_pcm(
                model=model, data=data, config=cf, skip_grad_checkpoint=True
            ).reconstruction
        image = rec.detach().cpu().numpy().squeeze().astype(np.float32).copy()
        assert image.shape == (256, 256) and np.isfinite(image).all()
        metric = oldbench.score(rec, truth, dr)
        oldscore = float(rows[category, sid]["utv_c60_psnr"])
        assert abs(metric["psnr"] - oldscore) < 0.10, (category, sid, metric, oldscore)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".partial").open("xb") as f:
            np.save(f, image, allow_pickle=False)
        path.with_suffix(".partial").replace(path)
        frozen_json(
            report,
            {
                "category": category,
                "sample_id": sid,
                "metric": metric,
                "model_checkpoint_sha256": digest(CKPT),
                "model_config_sha256": digest(CFG),
                "pinned_source_sha256": digest(Path(pcm_sapm.__file__)),
                "measurement_sha256": row["measurement_sha256"],
                "pattern_sha256": row["pattern_sha256"],
                "replay_psnr_difference": float(metric["psnr"] - oldscore),
                "test_not_used_to_choose_parameters": True,
            },
        )
        print("C60_REPLAY", category, sid, metric["psnr"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    args = ap.parse_args()
    main(args.category)
