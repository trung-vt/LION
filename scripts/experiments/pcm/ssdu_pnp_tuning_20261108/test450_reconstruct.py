"""Blind-frozen 450-image PCM test for SSDU GS and Step-6 oracle GS-PnP.

No SSDU hyperparameter may be changed using test metrics.
The two SSDU GS variants select identical config, so one unique execution
provides both labels, while retaining separately frozen selection records.
Archived ADMM, SPGL1, TV, ZF and C60 results are reused after hash audits.
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
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct_gs,
)
from scripts.experiments.pcm.gs_pnp_gs_drunet_tuned_20261104.step6_frozen_reconstructor import (
    load_frozen_selection,
    reconstruct_frozen,
)

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
SSDU = ROOT / "ssdu_pnp_tuning_20261108_v1"
OUT = SSDU / "test450" / "v1"
BENCH = ROOT / "ten_methods_matched_450_20261105_v1/all_ten_methods_450_metrics.csv"
BASE = ROOT / "pcm_utv_fixed20_npl_default_20261103"
PREV_ADMM = (
    ROOT / "pnp_admm_tuning_20261103/step3_final_heldout_v1/step3_per_image_scores.csv"
)
PREV_TV = ROOT / "tv_lambda0p011_fixed20_npl_450_test_20261104_v1/tv_450_metrics.csv"
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def sha(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def shafile(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(newline="") as f:
        records = {(r["category"], int(r["sample_id"])): r for r in csv.DictReader(f)}
    if len(records) != 450 or set(r["category"] for r in records.values()) != set(
        CATEGORIES
    ):
        raise RuntimeError("Expected 450 unique category/sample records")
    return records


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != b:
            raise RuntimeError("Existing record drift: " + str(path))
        return
    tmp = path.with_suffix(".partial")
    with tmp.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_frozen():
    assert (
        SSDU / "full50/frozen_pair.json"
    ).exists(), "SSDU must be frozen before test"
    options = {}
    for name in ("admm", "gs"):
        a = json.loads((SSDU / "pilot5" / name / "frozen_selection.json").read_text())
        b = json.loads((SSDU / "full50" / name / "frozen_selection.json").read_text())
        assert a["winner"]["parameters"] == b["winner"]["parameters"]
        assert all(
            not x["accessed_test_set"] and not x["used_ground_truth"] for x in (a, b)
        )
        options[name] = a["winner"]["parameters"]
    assert options["admm"]["rho"] == 0.5 and options["admm"]["sigma"] == 0.075
    assert (
        options["admm"]["min_iterations"] == 20
        and options["admm"]["max_iterations"] == 100
    )
    assert options["admm"]["consecutive_passes"] == 5
    assert options["gs"]["output"] == "paper_final_z"
    oracle = load_frozen_selection()
    assert oracle["config"]["lambda"] == 0.03
    assert oracle["config"]["sigma"] == 0.05
    return options, oracle


def validate_archive():
    baseline = rows(BENCH)
    original = rows(BASE / "all_450_metrics.csv")
    admm = rows(PREV_ADMM)
    tv = rows(PREV_TV)
    assert set(baseline) == set(original) == set(admm) == set(tv)
    for key in baseline:
        base = baseline[key]
        assert (
            base["pattern_sha256"]
            == original[key]["pattern_sha256"]
            == admm[key]["pattern_sha256"]
            == tv[key]["pattern_sha256"]
        ), key
        assert (
            base["measurement_sha256"]
            == original[key]["measurement_sha256"]
            == admm[key]["measurement_sha256"]
            == tv[key]["measurement_sha256"]
        ), key
        for field in METRICS:
            assert (
                abs(
                    float(base["pnp_tuned_" + field]) - float(admm[key]["pnp_" + field])
                )
                < 1e-8
            ), key
            assert abs(float(base["tv_" + field]) - float(tv[key][field])) < 1e-8, key
    frozen = json.loads(
        (
            ROOT
            / "pnp_admm_tuning_20261103/step3_final_heldout_v1/frozen_protocol.json"
        ).read_text()
    )
    assert frozen["rho"] == 0.5 and frozen["sigma_d"] == 0.075
    assert frozen["stop_rule"]["consecutive_passes"] == 5
    assert frozen["stop_rule"]["max_iters"] == 100
    assert frozen["stop_rule"]["min_iters"] == 20
    assert frozen["test_images_total"] == 450
    cfg, ids = oldbench.config_and_ids()
    assert sorted(ids) == sorted({key[1] for key in baseline})
    p = json.loads((BASE / "protocol.json").read_text())
    assert p["test_ids"] == ids and p["categories"] == list(CATEGORIES)
    assert p["sampling_ratio"] == 0.2 and p["J"] == 8 and p["coarse_J"] == 6
    methods = {
        "zero_filled": {"source": "matched450", "csv_prefix": "zero_filled"},
        "spgl1_bp": {"source": "matched450", "csv_prefix": "spgl1_bp"},
        "spgl1_bpdn": {"source": "matched450", "csv_prefix": "bpdn"},
        "pnp_admm_oracle": {
            "source": "matched450+step3_provenance",
            "csv_prefix": "pnp_tuned",
        },
        "gs_pnp_oracle": {"source": "new_full450_step6_paper_final_z"},
        "tv": {"source": "matched450+tv_provenance", "csv_prefix": "tv"},
        "utv_c60": {"source": "matched450", "csv_prefix": "C"},
        "pnp_admm_ssdu5": {
            "source": "same_frozen_config_as_pnp_admm_oracle",
            "csv_prefix": "pnp_tuned",
        },
        "pnp_admm_ssdu50": {
            "source": "same_frozen_config_as_pnp_admm_oracle",
            "csv_prefix": "pnp_tuned",
        },
        "gs_pnp_ssdu5": {
            "source": "new_full450_gs_ssdu",
            "reuse_shared_execution": True,
        },
        "gs_pnp_ssdu50": {
            "source": "new_full450_gs_ssdu",
            "reuse_shared_execution": True,
        },
    }
    assert len(methods) == 11
    return baseline, cfg, ids, methods


def prepare():
    frozen, oracle = read_frozen()
    baseline, cfg, ids, methods = validate_archive()
    protocol = {
        "study": "SSDU PnP frozen 5/50 matched PCM test, 450 cases",
        "status": "frozen_input_before_test_run",
        "image_shape": [256, 256],
        "sampling_ratio": 0.2,
        "J": 8,
        "coarse_J": 6,
        "NPL_noise": "fixed default, same per-image seeds as prior matched baseline",
        "categories": list(CATEGORIES),
        "ids": ids,
        "frozen_ssdu_admm": frozen["admm"],
        "frozen_ssdu_gs": frozen["gs"],
        "frozen_gs_oracle_step6": oracle["config"],
        "frozen_gs_oracle_stop": oracle["stopping"],
        "baseline_csv_sha256": shafile(BENCH),
        "baseline_protocol_sha256": shafile(BASE / "protocol.json"),
        "existing_admm_csv_sha256": shafile(PREV_ADMM),
        "existing_tv_csv_sha256": shafile(PREV_TV),
        "gs_oracle_selection_sha256": shafile(
            ROOT
            / "gs_pnp_gs_drunet_step6_full_validation_20261104_v1/selected_config.json"
        ),
        "ssdu5_admm_selection_sha256": shafile(
            SSDU / "pilot5/admm/frozen_selection.json"
        ),
        "ssdu50_admm_selection_sha256": shafile(
            SSDU / "full50/admm/frozen_selection.json"
        ),
        "ssdu5_gs_selection_sha256": shafile(SSDU / "pilot5/gs/frozen_selection.json"),
        "ssdu50_gs_selection_sha256": shafile(SSDU / "full50/gs/frozen_selection.json"),
        "all_11_methods": methods,
        "num_unique_new_gs_reconstructions_per_case": 2,
        "reuse_admm_note": "Both SSDU ADMM configurations exactly coincide with the former validation-supervised ADMM frozen config and stopping rule; historical test computations are reused with all 450 hashes independently checked.",
        "ssdu_duplication_note": "GS-SSDU-5 and GS-SSDU-50 have identical hyperparameters and solver; one GS execution is shared, but both frozen selection identities retained.",
        "no_test_tuning": True,
        "test_metric_scale": "raw reconstruction arrays, per-image target dynamic range, no clipping",
        "colormap": "afmhot with cmap_max=0.8; image intensity vmin=0 vmax=1",
    }
    write_json(OUT / "protocol.json", protocol)
    print(
        "PREPARED_450_TEST",
        len(baseline),
        "two new GS variants, three ADMM labels reused",
        flush=True,
    )
    return frozen, oracle, baseline, cfg, ids


def configure_gpu():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    return torch.device("cuda:0")


def run(
    method: str,
    start_category: int = 0,
    stop_category: int = 5,
    limit_per_category: int = 90,
):
    selected, oracle, baseline, cfg, ids = prepare()
    device = configure_gpu()
    model = make_model("gs", device)
    selection = selected["gs"]
    paths = []
    for category in CATEGORIES[start_category:stop_category]:
        cat_count = 0
        for j, sid in enumerate(ids):
            key = (category, sid)
            target = OUT / "records" / method / category / f"{sid:04d}.json"
            if target.exists():
                old = json.loads(target.read_text())
                assert old["measurement_sha256"] == baseline[key]["measurement_sha256"]
                assert old["pattern_sha256"] == baseline[key]["pattern_sha256"]
                assert (
                    old["method"] == method
                    and old["frozen_source"] == "validation_only"
                )
                cat_count += 1
                continue
            png = oldbench.DATA / category / f"image{sid:04d}.png"
            image = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
            gt = torch.from_numpy((image.astype(np.float64) / 255).astype(np.float32))[
                None, None
            ].to(device)
            truth = gt.detach().cpu()
            data_range = float((gt.max() - gt.min()).item())
            if not 0 < data_range <= 1:
                raise ValueError("Unexpected ground truth range")
            acq, indices = oldbench.acquisition(gt, cfg, category, sid, device)
            y = acq.measurement
            msha = sha(y.detach().cpu().numpy())
            isha = sha(np.asarray(indices, dtype=np.int64))
            assert msha == baseline[key]["measurement_sha256"], (
                method,
                key,
                "measurement hash mismatch",
            )
            assert isha == baseline[key]["pattern_sha256"], (
                method,
                key,
                "mask hash mismatch",
            )
            if method == "gs_ssdu":
                with torch.no_grad():
                    recon, diag = reconstruct_gs(acq.op, y, model, selection)
                expected_conf = "ssdu_frozen5_and_frozen50_identical"
            elif method == "gs_oracle_step6":
                with torch.no_grad():
                    recon, diag = reconstruct_frozen(acq.op, y, model, oracle)
                expected_conf = "step6_supervised_validation_oracle_frozen"
            else:
                raise ValueError("Unknown GS comparison method")
            scores = oldbench.score(recon, truth, data_range)
            assert all(np.isfinite(v) for v in scores.values())
            img = recon.detach().cpu().numpy().squeeze().astype(np.float32, copy=True)
            assert img.shape == (256, 256) and np.isfinite(img).all()
            record = {
                "category": category,
                "sample_id": sid,
                "permutation_position": 911 + j,
                "method": method,
                "frozen_source": "validation_only",
                "configuration": expected_conf,
                "measurement_sha256": msha,
                "pattern_sha256": isha,
                "source_png_sha256": shafile(png),
                "reconstruction_sha256": sha(img),
                "metrics": scores,
                "diagnostics": diag,
                "data_range": data_range,
                "no_test_retraining": True,
            }
            rec_path = OUT / "raw" / method / f"{category}_{sid:04d}.npy"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = rec_path.with_suffix(".npy.partial")
            with tmp.open("xb") as f:
                np.save(f, img, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, rec_path)
            write_json(target, record)
            cat_count += 1
            if cat_count % 10 == 0 or cat_count == 1:
                print(
                    "CASE_PROGRESS",
                    method,
                    category,
                    cat_count,
                    "/",
                    limit_per_category,
                    "PSNR",
                    round(scores["psnr"], 3),
                    "SSIM",
                    round(scores["ssim"], 4),
                    "accepted",
                    diag.get("accepted_iterations"),
                    "time",
                    round(float(diag.get("wall_seconds_with_final_gradient", 0.0)), 3),
                    flush=True,
                )
            del acq, recon, gt, truth, y, img
            if cat_count >= limit_per_category:
                break
        paths.append((category, cat_count))
        print("CATEGORY_FINISHED", method, category, cat_count, flush=True)
    print("METHOD_RUN_FINISHED", method, paths, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=("prepare", "gs_ssdu", "gs_oracle_step6"), required=True
    )
    ap.add_argument("--first-category", type=int, default=0)
    ap.add_argument("--last-category", type=int, default=5)
    ap.add_argument("--limit-per-category", type=int, default=90)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run(args.mode, args.first_category, args.last_category, args.limit_per_category)


if __name__ == "__main__":
    main()
