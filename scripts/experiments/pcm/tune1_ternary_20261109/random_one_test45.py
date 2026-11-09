"""Apply one randomly selected frozen Tune-1 calibration per category to test45.

The selection is frozen using validation acquisitions before ANY test45
ground-truth reconstruction or test score can be read. Each calibration
configuration is evaluated on nine independent held-out test images of
its own category, all using original 20% NPL-default acquisitions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import hashlib

import numpy as np
import torch
from PIL import Image

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    BASE,
    frozen_json,
    digest,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct,
)
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import oldbench
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.test450_reconstruct import (
    sha as hash_array,
)

METHS = ("admm", "gs")


def assert_all_frozen():
    calib = json.loads((OUTPUT / "prepared/validation_calibrations.json").read_text())[
        "cases"
    ]
    test = json.loads((OUTPUT / "prepared/test45.json").read_text())["cases"]
    assert len(calib) == 25 and len(test) == 45
    allcfg = {}
    for meth in METHS:
        for item in calib:
            cat = item["category"]
            sid = int(item["sample_id"])
            p = OUTPUT / "tuning" / meth / cat / f"{sid:04d}" / "frozen_selection.json"
            if not p.exists():
                raise RuntimeError(
                    f"Do not open test45 before all 50 tune-1 searches are frozen: {p}"
                )
            c = json.loads(p.read_text())
            assert c["stage"] == "validation_selection_frozen_before_test"
            assert c["no_image_ground_truth_used"] and c["no_test_metrics_used"]
            assert c["category"] == cat and c["calibration_sample_id"] == sid
            assert c["validation_source_file_sha256"] == item["source_file_sha256"]
            allcfg[(meth, cat, sid)] = c
    assert len(allcfg) == 50
    return calib, test, allcfg


def init_device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.cuda.is_available()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    torch.set_num_threads(4)
    return torch.device("cuda:0")


def replay_category(method: str, category: str):
    assert method in METHS and category in CATEGORIES
    calibration, test, configs = assert_all_frozen()
    choices_path = OUTPUT / "test45/random_one_per_category_v1/selection.json"
    manifest = json.loads(choices_path.read_text())
    assert manifest["status"] == "frozen_before_test_access"
    assert manifest["using_test_metrics_for_selection"] is False
    assert manifest["n_unique_test_images"] == 45
    assert manifest["test45_file_sha256"] == digest(OUTPUT / "prepared/test45.json")
    chosen = manifest["selected_calibrations"][method][category]
    local = [
        x
        for x in calibration
        if x["category"] == category
        and int(x["sample_id"]) == chosen["calibration_sample_id"]
    ]
    subset = [x for x in test if x["category"] == category]
    assert len(local) == 1 and len(subset) == 9
    config_file = (
        OUTPUT
        / "tuning"
        / method
        / category
        / f'{chosen["calibration_sample_id"]:04d}'
        / "frozen_selection.json"
    )
    assert chosen["selection_file_sha256"] == digest(config_file)
    assert (
        chosen["frozen_config"]
        == configs[(method, category, chosen["calibration_sample_id"])]["winner_config"]
    )
    lock = {
        "method": method,
        "category": category,
        "calibration_manifest_sha256": digest(
            OUTPUT / "prepared/validation_calibrations.json"
        ),
        "test45_manifest_sha256": digest(OUTPUT / "prepared/test45.json"),
        "configs_sha256": {
            str(c["sample_id"]): digest(
                OUTPUT
                / "tuning"
                / method
                / category
                / f'{int(c["sample_id"]):04d}'
                / "frozen_selection.json"
            )
            for c in local
        },
        "selected_test_case_keys": [
            f"{x['category']}_{x['sample_id']:04d}" for x in subset
        ],
        "uses_only_frozen_validation_configs": True,
        "no_test_reparameterisation": True,
    }
    lock["choice_manifest_sha256"] = digest(choices_path)
    lock["random_chosen_calibration_id"] = int(chosen["calibration_sample_id"])
    out = OUTPUT / "test45/random_one_per_category_v1/recon" / method / category
    frozen_json(out / "test_lock.json", lock)
    device = init_device()
    model = make_model(method, device)
    # Cached test acquisitions and test image GT may now be used for evaluation.
    base_config, all_ids = oldbench.config_and_ids()
    assert len(all_ids) == 90
    assert all(x["sample_id"] in all_ids for x in subset)
    for calib in local:
        csid = int(calib["sample_id"])
        selected = configs[(method, category, csid)]
        candidate = selected["winner_config"]
        for testrow in subset:
            tid = int(testrow["sample_id"])
            p = out / "records" / f"{csid:04d}" / f"{tid:04d}.json"
            imgpath = out / "raw" / f"{csid:04d}" / f"{tid:04d}.npy"
            if p.exists():
                old = json.loads(p.read_text())
                assert (
                    old["frozen_selection_sha256"] == lock["configs_sha256"][str(csid)]
                )
                assert old["test_measurement_sha256"] == testrow["measurement_sha256"]
                assert old["test_pattern_sha256"] == testrow["pattern_sha256"]
                assert imgpath.exists()
                print("REUSE_RANDOM_ONE_TEST", method, category, csid, tid, flush=True)
                continue
            png = oldbench.DATA / category / f"image{tid:04d}.png"
            gt = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
            truth = torch.from_numpy(
                (gt.astype(np.float64) / 255.0).astype(np.float32)
            )[None, None].to(device)
            target_cpu = truth.detach().cpu()
            dr = float((truth.max() - truth.min()).item())
            assert 0 < dr <= 1
            acq, indices = oldbench.acquisition(
                truth, base_config, category, tid, device
            )
            y = acq.measurement
            assert hash_array(y.detach().cpu().numpy()) == testrow["measurement_sha256"]
            assert (
                hash_array(np.asarray(indices, dtype=np.int64))
                == testrow["pattern_sha256"]
            )
            t = time.perf_counter()
            with torch.no_grad():
                x, detail = reconstruct(method, acq.op, y, model, candidate)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t
            metrics = oldbench.score(x, target_cpu, dr)
            assert all(np.isfinite(v) for v in metrics.values())
            raw = x.detach().cpu().numpy().squeeze().astype(np.float32)
            assert raw.shape == (256, 256) and np.isfinite(raw).all()
            imgpath.parent.mkdir(parents=True, exist_ok=True)
            with imgpath.with_suffix(".partial").open("xb") as f:
                np.save(f, raw, allow_pickle=False)
            imgpath.with_suffix(".partial").replace(imgpath)
            frozen_json(
                p,
                {
                    "schema": "pcm-tune1-test45-pair-v1",
                    "method": method,
                    "category": category,
                    "calibration_id": csid,
                    "test_sample_id": tid,
                    "frozen_selection_sha256": lock["configs_sha256"][str(csid)],
                    "test_measurement_sha256": testrow["measurement_sha256"],
                    "test_pattern_sha256": testrow["pattern_sha256"],
                    "test_source_image_sha256": digest(png),
                    "reconstruction_sha256": hash_array(raw),
                    "frozen_config": candidate,
                    "metrics": metrics,
                    "solver": detail,
                    "wall_seconds": elapsed,
                    "no_test_driven_parameter_tuning": True,
                },
            )
            print(
                "RANDOM_ONE_TEST",
                method,
                category,
                csid,
                tid,
                "PSNR",
                round(metrics["psnr"], 3),
                "seconds",
                round(elapsed, 3),
                flush=True,
            )
    print(
        "CATEGORY_RANDOM_ONE_FINISHED",
        method,
        category,
        "chosen_calibration",
        chosen["calibration_sample_id"],
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHS, required=True)
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    a = ap.parse_args()
    replay_category(a.method, a.category)


if __name__ == "__main__":
    main()
