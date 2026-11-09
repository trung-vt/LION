"""One-time GPU parity audit for original GS-PnP on expanded-test hardware.

Does not select/modify any parameter, purely tests whether old RTX4070 scores
replay on the RTX4090 used for new expanded GS-PnP.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import oldbench
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct_gs,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs_test450 import (
    device_setup,
    validate,
    TEST,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import ROOT
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_reconstruct import sha


def main():
    _, archive, cfg, ids = validate()
    old = json.loads((ROOT.parent / "pilot5/gs/frozen_selection.json").read_text())
    assert old["winner"]["candidate_id"] == "gs_018"
    assert old["winner"]["parameters"]["lambda"] == 0.1
    assert old["winner"]["parameters"]["sigma"] == 0.075
    assert old["winner"]["parameters"]["tau0"] == 100 / 3
    param = old["winner"]["parameters"]
    with (ROOT.parent / "test450/v1/merged/all_450_11_method_metrics.csv").open(
        newline=""
    ) as f:
        table = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    device = device_setup()
    model = make_model("gs", device)
    selected = [
        ("Silicone", 807),
        ("Organic", 623),
        ("Organic", 727),
        ("Silicone", 188),
        ("Perovskite", 527),
        ("CIGS", 279),
        ("Reference", 301),
        ("Organic", 331),
    ]
    results = []
    for cat, sid in selected:
        assert sid in ids
        key = (cat, sid)
        path = oldbench.DATA / cat / f"image{sid:04d}.png"
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        gt = torch.from_numpy((img.astype(np.float64) / 255).astype(np.float32))[
            None, None
        ].to(device)
        target = gt.cpu()
        dr = float((gt.max() - gt.min()).item())
        acq, indices = oldbench.acquisition(gt, cfg, cat, sid, device)
        obs = acq.measurement
        assert sha(obs.detach().cpu().numpy()) == table[key]["measurement_sha256"]
        assert sha(np.asarray(indices, dtype=np.int64)) == table[key]["pattern_sha256"]
        with torch.no_grad():
            recon, diagnostics = reconstruct_gs(acq.op, obs, model, param)
        score = oldbench.score(recon, target, dr)
        original_score = float(table[key]["gs_pnp_ssdu5_psnr"])
        delta = float(score["psnr"] - original_score)
        item = {
            "category": cat,
            "sample_id": sid,
            "same_gpu_original_psnr": float(score["psnr"]),
            "old_rtx4070_archived_psnr": original_score,
            "difference_db": delta,
            "accepted_iterations": diagnostics["accepted_iterations"],
            "last_tau": diagnostics["last_tau"],
        }
        results.append(item)
        print(
            "GPU_PARITY",
            cat,
            sid,
            "new",
            score["psnr"],
            "previous",
            original_score,
            "delta",
            delta,
            flush=True,
        )
    OUT = TEST / "same_gpu_probe.json"
    OUT.write_text(
        json.dumps(
            {
                "note": "Original frozen gs_018 on RTX4090 vs archived GS SSDU on RTX4070. NOT a tuning run.",
                "cases": results,
                "no_test_driven_tuning": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("SAME_GPU_PROBE_SAVED", OUT, flush=True)


if __name__ == "__main__":
    main()
