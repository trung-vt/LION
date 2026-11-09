"""Retrospective 450-image PCM test of FROZEN expanded-GS SSDU configuration.

This module refuses to open any test dataset until the expanded 5-image
SSDU sweep is fully finished and its selection is frozen. It never chooses
parameters using test scores. Saves exact paired test acquisitions and raw
reconstructions for independent analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import oldbench
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import (
    ROOT,
    PREPARED,
    GRID_FILE,
    sha_file,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct_gs,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_reconstruct import (
    BASE,
    BENCH,
    validate_archive,
    sha,
    write_json,
)

TEST = ROOT / "test450"
SELECTION = ROOT / "frozen_selection.json"
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")


def frozen_selection():
    assert SELECTION.is_file(), "Expanded SSDU selection must be frozen before test!"
    record = json.loads(SELECTION.read_text())
    assert record["status"] == "frozen_before_any_additional_test_evaluation"
    assert record["no_test_metrics_used"] and record["no_ground_truth_used"]
    assert record["tested_candidate_count"] == 27
    assert record["grid_sha256"] == sha_file(GRID_FILE)
    assert record["preflight_manifest_sha256"] == sha_file(PREPARED / "manifest.json")
    cfg = record["winner"]["parameters"]
    assert cfg["output"] == "paper_final_z"
    assert (cfg["lambda"], cfg["sigma"], cfg["tau0"]) in {
        (lam, sig, tau)
        for lam in (0.1, 0.2, 0.4)
        for sig in (0.075, 0.1, 0.15)
        for tau in (100 / 3, 50.0, 75.0)
    }
    return record, cfg


def validate():
    record, candidate = frozen_selection()
    archive, config, ids, _ = validate_archive()
    assert len(archive) == 450 and len(ids) == 10
    assert len(set(ids)) == 10
    proto = {
        "protocol": "GS-PnP-SSDU-expanded-5 on 450 fixed 20%-sampling NPL-noisy PCM test images",
        "selection_status": "frozen_before_test",
        "method": "GS-PnP-SSDU-expanded-5",
        "number_cases": 450,
        "categories": list(CATEGORIES),
        "ids": ids,
        "selection_file_sha256": sha_file(SELECTION),
        "candidate_grid_sha256": sha_file(GRID_FILE),
        "selection_id": record["winner"]["candidate_id"],
        "selection_parameters": candidate,
        "fidelity": "normalised PCM Hadamard J=8 coarse J=6 multilevel 20% NPL-default fixed",
        "measurement_fingerprint_source": str(BENCH),
        "measurement_fingerprint_csv_sha256": sha_file(BENCH),
        "metric_definition": "oldbench.score raw unclipped images; per-image true target dynamic range",
        "stage": "retrospective test only after validation-only parameter freezing",
        "no_test_driven_parameter_selection": True,
    }
    write_json(TEST / "protocol.json", proto)
    print(
        "TEST450_PREPARED_FROZEN",
        proto["selection_id"],
        candidate,
        "cases",
        len(archive),
        flush=True,
    )
    return candidate, archive, config, ids


def device_setup():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-"):
        raise RuntimeError("Select one GPU by UUID")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic CUDA workspace missing")
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda:0")


def run(first_category=0, last_category=5):
    candidate, baseline, config, ids = validate()
    device = device_setup()
    model = make_model("gs", device)
    counts = {}
    for category in CATEGORIES[first_category:last_category]:
        counts[category] = 0
        for sid in ids:
            key = (category, sid)
            dest = TEST / "records" / category / f"{sid:04d}.json"
            raw = TEST / "raw" / f"{category}_{sid:04d}.npy"
            if dest.exists():
                old = json.loads(dest.read_text())
                assert old["measurement_sha256"] == baseline[key]["measurement_sha256"]
                assert old["pattern_sha256"] == baseline[key]["pattern_sha256"]
                assert old["selection_file_sha256"] == sha_file(SELECTION)
                assert raw.exists()
                counts[category] += 1
                continue
            path = oldbench.DATA / category / f"image{sid:04d}.png"
            image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            gt = torch.from_numpy(
                (image.astype(np.float64) / 255.0).astype(np.float32)
            )[None, None].to(device)
            target = gt.detach().cpu()
            data_range = float((gt.max() - gt.min()).item())
            assert 0 < data_range <= 1
            acq, indices = oldbench.acquisition(gt, config, category, sid, device)
            y = acq.measurement
            meas_sha = sha(y.detach().cpu().numpy())
            ids_sha = sha(np.asarray(indices, dtype=np.int64))
            assert meas_sha == baseline[key]["measurement_sha256"], (key, "measurement")
            assert ids_sha == baseline[key]["pattern_sha256"], (key, "pattern")
            with torch.no_grad():
                predicted, diagnostics = reconstruct_gs(acq.op, y, model, candidate)
            scores = oldbench.score(predicted, target, data_range)
            assert all(np.isfinite(float(v)) for v in scores.values())
            x = predicted.detach().cpu().numpy().squeeze().astype(np.float32).copy()
            assert x.shape == (256, 256) and np.isfinite(x).all()
            raw.parent.mkdir(parents=True, exist_ok=True)
            with raw.with_suffix(".partial").open("xb") as f:
                np.save(f, x, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            raw.with_suffix(".partial").replace(raw)
            record = {
                "category": category,
                "sample_id": sid,
                "method": "GS-PnP-SSDU-expanded-5",
                "selection_file_sha256": sha_file(SELECTION),
                "selected_candidate_id": json.loads(SELECTION.read_text())["winner"][
                    "candidate_id"
                ],
                "measurement_sha256": meas_sha,
                "pattern_sha256": ids_sha,
                "source_png_sha256": sha_file(path),
                "reconstruction_sha256": sha(x),
                "metrics": scores,
                "diagnostics": diagnostics,
                "data_range": data_range,
                "run_status": "frozen before test",
                "no_test_tuning": True,
            }
            write_json(dest, record)
            counts[category] += 1
            if counts[category] % 10 == 0 or counts[category] == 1:
                print(
                    "EXPANDED_GS_TEST",
                    category,
                    counts[category],
                    "/",
                    len(ids),
                    "PSNR",
                    round(scores["psnr"], 3),
                    "accepted",
                    diagnostics["accepted_iterations"],
                    "backtracking_reject",
                    diagnostics["rejections"],
                    flush=True,
                )
            del gt, target, acq, y, predicted, x
        print("CATEGORY_DONE", category, counts[category], flush=True)
    print("TEST450_GS_RUN_FINISHED", counts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run"])
    parser.add_argument("--first-category", type=int, default=0)
    parser.add_argument("--last-category", type=int, default=5)
    a = parser.parse_args()
    if a.action == "preflight":
        validate()
    else:
        run(a.first_category, a.last_category)


if __name__ == "__main__":
    main()
