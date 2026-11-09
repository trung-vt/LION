"""Measurement-only tune-1 search for each of 25 PCM validation acquisitions.

Selects hyperparameters independently for each acquired noisy calibration datum.
Never loads ground-truth images or test image metrics.
Run with --method admm/gs --category CIGS/Organic/Perovskite/Reference/Silicone.
Each invocation uses one isolated GPU; it can resume cached candidate scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    BASE,
    OUTPUT,
    SCHEMA,
    CATEGORIES,
    frozen_json,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.ternary import search
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.score_grid import (
    score_single_case,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import make_model

HERE = Path(__file__).resolve().parent
PREVIOUS = BASE / "ssdu_pnp_tuning_20261108_v1"
METHODS = ("admm", "gs")


def local_context():
    sel = json.loads((OUTPUT / "prepared/validation_calibrations.json").read_text())
    protocol = json.loads((OUTPUT / "prepared/protocol.json").read_text())
    parameters = json.loads((OUTPUT / "prepared/ranges.json").read_text())
    assert sel["schema"] == protocol["schema"] == parameters["schema"] == SCHEMA
    assert len(sel["cases"]) == 25
    assert sel["selection_without_gt_or_test_metrics"] is True
    assert protocol["status"] == "prepared_no_gpu_parameter_search"
    return sel, parameters


def configure():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert uuid.startswith(
        "GPU-"
    ), "Select a single GPU with CUDA_DEVICE_ORDER=PCI_BUS_ID"
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


def code_checksums():
    files = [
        HERE / "prepare.py",
        HERE / "ternary.py",
        HERE / "run_tune1.py",
        HERE.parent / "ssdu_pnp_tuning_20261108/solver_adapter.py",
        HERE.parent / "ssdu_pnp_tuning_20261108/score_grid.py",
        HERE.parent / "ssdu_pnp_tuning_20261108/ssdu_risk.py",
    ]
    return {str(x): digest(x) for x in files}


def candidate_config(method, values, identifier):
    assert set(values) == (
        {"rho", "sigma"} if method == "admm" else {"lambda", "sigma", "tau0"}
    )
    if method == "admm":
        return {
            "candidate_id": identifier,
            "rho": values["rho"],
            "sigma": values["sigma"],
            "min_iterations": 20,
            "max_iterations": 100,
            "eps_abs": 1e-4,
            "eps_rel": 1e-3,
            "consecutive_passes": 5,
        }
    return {
        "candidate_id": identifier,
        "lambda": values["lambda"],
        "sigma": values["sigma"],
        "tau0": values["tau0"],
        "max_accepted_iterations": 100,
        "max_backtracking_attempts": 20,
        "gamma_backtracking": 0.1,
        "eta_backtracking": 0.9,
        "stopping_policy": "deepinv_relative_objective",
        "stopping_eps": 1e-6,
        "output": "paper_final_z",
    }


def run(method, category):
    assert method in METHODS and category in CATEGORIES
    selected, params = local_context()
    data = [x for x in selected["cases"] if x["category"] == category]
    assert len(data) == 5
    device = configure()
    checkpoint_hash = {
        "schema": SCHEMA,
        "method": method,
        "selection_sha256": digest(OUTPUT / "prepared/validation_calibrations.json"),
        "ranges_sha256": digest(OUTPUT / "prepared/ranges.json"),
        "numerical_source_checksums": code_checksums(),
        "sampling_rate": 0.2,
        "noise": "NPL fixed default",
        "n_folds": 4,
        "no_test_metrics_used": True,
        "no_gt_image_used": True,
    }
    local = OUTPUT / "tuning" / method / category
    frozen_json(local / "source_lock.json", checkpoint_hash)
    model = make_model(method, device)
    for index, case in enumerate(data, 1):
        sample = int(case["sample_id"])
        folder = local / f"{sample:04d}"
        frozen = folder / "frozen_selection.json"
        if frozen.exists():
            recorded = json.loads(frozen.read_text())
            assert recorded["source_lock_sha256"] == digest(local / "source_lock.json")
            print("REUSED_FROZEN", method, category, sample, flush=True)
            continue
        evaluation_root = folder / "evaluations"
        parameter_ranges = params["ranges"][method]
        orders = params["coordinate_order"][method]
        evaluated = 0

        def evaluate(values):
            nonlocal evaluated
            raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
            ident = "t1_" + method + "_" + hashlib.sha256(raw).hexdigest()[:16]
            candidate = candidate_config(method, values, ident)
            score = evaluation_root / (ident + ".json")
            error = evaluation_root / (ident + ".failure.json")
            if score.exists():
                v = json.loads(score.read_text())
                assert v["parameters"] == candidate
                assert v["source_file_sha256"] == case["source_file_sha256"]
                assert v["fold_file_sha256"] == case["folds_sha256"]
                assert v["no_image_ground_truth_used"] is True
                return float(v["mean_heldout_mse"])
            if error.exists():
                return float("inf")
            evaluated += 1
            try:
                payload = score_single_case(
                    case,
                    method,
                    candidate,
                    device,
                    model,
                    checkpoint_hash["ranges_sha256"],
                )
                assert payload["uses_ground_truth"] is False
                assert payload["test_set_access"] is False
                assert len(payload["folds"]) == 4
                payload["no_image_ground_truth_used"] = True
                payload["parameter_source"] = "bounded_log_ternary_search"
                frozen_json(score, payload)
                loss = float(payload["mean_heldout_mse"])
                assert math.isfinite(loss) and loss >= 0
                print(
                    "PROBE", method, category, sample, ident, round(loss, 8), flush=True
                )
                return loss
            except Exception as exc:
                frozen_json(
                    error,
                    {
                        "schema": SCHEMA,
                        "method": method,
                        "category": category,
                        "sample_id": sample,
                        "candidate": candidate,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "no_test_metrics_used": True,
                        "no_image_ground_truth_used": True,
                    },
                )
                print(
                    "INVALID_PROBE",
                    method,
                    category,
                    sample,
                    ident,
                    type(exc).__name__,
                    str(exc),
                    flush=True,
                )
                return float("inf")

        print(
            "TUNE_ONE_START",
            method,
            category,
            sample,
            "calibration",
            index,
            "of",
            len(data),
            flush=True,
        )
        result = search(
            evaluate,
            parameter_ranges,
            orders,
            contractions=params["ternary_contractions_per_coordinate"],
        )
        assert result["evaluations"] <= (
            params["algorithmic_budget_per_calibration"][
                method + "_max_candidate_evaluations"
            ]
        )
        cfg = candidate_config(method, result["winner"], "frozen_" + method)
        # Freeze only in validation space. Nothing here reads the test45 CSV.
        selection = {
            "schema": SCHEMA,
            "stage": "validation_selection_frozen_before_test",
            "method": method,
            "category": category,
            "calibration_sample_id": sample,
            "validation_source_file_sha256": case["source_file_sha256"],
            "fold_file_sha256": case["folds_sha256"],
            "source_lock_sha256": digest(local / "source_lock.json"),
            "winner_config": cfg,
            "winner_params": result["winner"],
            "winner_ssdu_mse": result["winner_measurement_mse"],
            "probes": result["evaluations"],
            "invalid_probes": result["invalid_evaluations"],
            "unimodality_assumed_not_proven": True,
            "no_image_ground_truth_used": True,
            "no_test_metrics_used": True,
        }
        frozen_json(folder / "search_trace.json", result)
        frozen_json(frozen, selection)
        print(
            "TUNE_ONE_FROZEN",
            method,
            category,
            sample,
            result["winner"],
            result["winner_measurement_mse"],
            "evaluated",
            result["evaluations"],
            flush=True,
        )
    print("CATEGORY_TUNING_FINISHED", method, category, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    args = ap.parse_args()
    run(args.method, args.category)


if __name__ == "__main__":
    main()
