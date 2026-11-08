"""Expanded GS-PnP SSDU five-image sweep.

Preparation, resumable GPU scoring, frozen selection, read-only status.
This is a separate experiment. No ground-truth file or test-case file is read.

Usage after preparing:
    python -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs prepare
    python -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs status
    python -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs run
    python -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs freeze

Do not run "run" or "freeze" during preparation of the experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    OUTPUT as ORIGINAL_PREPARED,
    CATEGORIES,
    sha_file,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.score_grid import (
    atomic_json,
    score_single_case,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
    aggregate_case_losses,
    read_blind_acquisition,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import make_model

MODULE_DIR = Path(__file__).resolve().parent
ROOT = ORIGINAL_PREPARED.parent / "next_gs_ssdu_sweep_v1"
PILOT = ORIGINAL_PREPARED.parent / "pilot5" / "gs"
ORIGINAL_CONTROL = PILOT / "candidates" / "gs_018"
GRID_FILE = ROOT / "candidate_grid.json"
PREPARED = ROOT / "prepared"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
CODE_FILES = ("expanded_gs.py", "solver_adapter.py", "score_grid.py", "ssdu_risk.py")
CONTROL_ID = "gs_exp_001"
NO_TEST_DATA = True
TOL = 1e-10


def record_sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def get_grid() -> tuple[list[dict], str]:
    d = json.loads(GRID_FILE.read_text())
    assert d["status"] == "frozen_proposed_grid_not_run"
    assert d["uses_test_metrics_for_selection"] is False
    source = d["candidates"]
    assert len(source) == 27
    grid = []
    for i, c in enumerate(source, 1):
        assert c["id"] == f"gs_exp_{i:03d}"
        cand = copy.deepcopy(c)
        cand["candidate_id"] = cand.pop("id")
        assert set(cand) == {
            "candidate_id",
            "lambda",
            "sigma",
            "tau0",
            "max_accepted_iterations",
            "max_backtracking_attempts",
            "gamma_backtracking",
            "eta_backtracking",
            "stopping_eps",
            "stopping_policy",
            "output",
        }
        assert cand["max_accepted_iterations"] == 100
        assert cand["max_backtracking_attempts"] == 20
        assert cand["gamma_backtracking"] == 0.1
        assert cand["eta_backtracking"] == 0.9
        assert cand["stopping_eps"] == 1e-6
        assert cand["stopping_policy"] == "deepinv_relative_objective"
        assert cand["output"] == "paper_final_z"
        grid.append(cand)
    space = {
        (float(lam), float(sigma), float(tau))
        for lam in (0.1, 0.2, 0.4)
        for sigma in (0.075, 0.1, 0.15)
        for tau in (100.0 / 3.0, 50.0, 75.0)
    }
    assert {(x["lambda"], x["sigma"], x["tau0"]) for x in grid} == space
    assert len({x["candidate_id"] for x in grid}) == 27
    return grid, sha_file(GRID_FILE)


def original_control():
    old = json.loads((PILOT / "frozen_selection.json").read_text())
    assert old["method"] == "gs" and old["num_images"] == 5
    assert old["status"] == "frozen_before_any_new_test_evaluation"
    assert old["used_ground_truth"] is False
    assert old["winner"]["candidate_id"] == "gs_018"
    ctrl = old["winner"]["parameters"]
    assert ctrl["lambda"] == 0.1
    assert ctrl["sigma"] == 0.075
    assert ctrl["tau0"] == 100.0 / 3.0
    return old


def read_cases() -> list[dict]:
    old = json.loads((ORIGINAL_PREPARED / "manifest.json").read_text())
    assert old["status"] == "prepared_no_solver_tuning"
    assert old["no_target_access"] and old["test_access"] is False
    assert len(old["validation_cases"]) == 50 and len(old["pilot5"]) == 5
    pilot = {(x["category"], int(x["sample_id"])) for x in old["pilot5"]}
    assert len(pilot) == 5
    cases = [
        c
        for c in old["validation_cases"]
        if (c["category"], int(c["sample_id"])) in pilot
    ]
    assert len(cases) == 5
    assert Counter(x["category"] for x in cases) == Counter(CATEGORIES)
    for case in cases:
        f = Path(case["source_file"])
        assert f.is_file() and "/validation/" in str(f)
        assert "/test/" not in str(f)
        assert sha_file(f) == case["source_file_sha256"]
        measurement, inds = read_blind_acquisition(f)
        assert record_sha(measurement) == case["measurement_sha256"]
        # Original archived indices are uint16 in the hash manifest.
        assert record_sha(inds.astype(np.uint16)) == case["sampled_indices_sha256"]
        fp = Path(case["fold_file"])
        assert fp.is_file() and sha_file(fp) == case["folds_sha256"]
        with np.load(fp, allow_pickle=False) as folds:
            hidden = [folds[f"heldout_{i}"].copy() for i in range(4)]
        assert sum(len(x) for x in hidden) == 9011
        assert np.array_equal(
            np.sort(np.concatenate(hidden).astype(np.int64)),
            np.arange(4096, 13107, dtype=np.int64),
        )
        assert max(len(x) for x in hidden) < 3000
    return cases


def prepared_record() -> dict:
    assert (PREPARED / "manifest.json").exists(), "Run prepare first"
    record = json.loads((PREPARED / "manifest.json").read_text())
    assert record["status"] == "prepared_not_run"
    assert record["candidate_grid_sha256"] == sha_file(GRID_FILE)
    assert record["source_validation_manifest_sha256"] == sha_file(
        ORIGINAL_PREPARED / "manifest.json"
    )
    assert record["old_control_frozen_selection_sha256"] == sha_file(
        PILOT / "frozen_selection.json"
    )
    assert record["code_sha256"] == {
        x: sha_file(MODULE_DIR / x) for x in CODE_FILES
    }, "Code changed after preflight; inspect and version protocol"
    assert record["has_ground_truth"] is False
    assert record["test_set_used_for_selection"] is False
    assert len(record["five_cases"]) == 5
    return record


def prepare() -> dict:
    grid, grid_sha = get_grid()
    cases = read_cases()
    old = original_control()
    assert (ORIGINAL_PREPARED / "fidelity_audit.json").exists()
    assert (ORIGINAL_PREPARED / "adapter_equivalence.json").exists()
    assert (
        json.loads((ORIGINAL_PREPARED / "fidelity_audit.json").read_text())["status"]
        == "passed"
    )
    assert (
        json.loads((ORIGINAL_PREPARED / "adapter_equivalence.json").read_text())[
            "status"
        ]
        == "passed"
    )
    assert (grid[0]["lambda"], grid[0]["sigma"], grid[0]["tau0"]) == (
        old["winner"]["parameters"]["lambda"],
        old["winner"]["parameters"]["sigma"],
        old["winner"]["parameters"]["tau0"],
    )
    orig_score = json.loads((ORIGINAL_CONTROL / "summary.json").read_text())
    assert orig_score["candidate"]["candidate_id"] == "gs_018"
    assert math.isclose(orig_score["loss"], old["winner"]["loss"], abs_tol=1e-14)
    provenance = {
        "schema": "pcm-expanded-gs-ssdu-5-preflight-v1",
        "status": "prepared_not_run",
        "purpose": "Only the blind SSDU validation MSE determines hyperparameters",
        "candidate_count": len(grid),
        "new_candidate_count": len(grid) - 1,
        "candidate_grid_sha256": grid_sha,
        "source_validation_manifest_sha256": sha_file(
            ORIGINAL_PREPARED / "manifest.json"
        ),
        "old_control_frozen_selection_sha256": sha_file(
            PILOT / "frozen_selection.json"
        ),
        "original_control_score": orig_score["loss"],
        "original_control_score_sha256": sha_file(ORIGINAL_CONTROL / "summary.json"),
        "control_params": grid[0],
        "five_cases": [
            {
                "category": c["category"],
                "sample_id": int(c["sample_id"]),
                "source_file_sha256": c["source_file_sha256"],
                "fold_file_sha256": c["folds_sha256"],
            }
            for c in cases
        ],
        "source_pretrained_model": "Original frozen GS-DRUNet, paper-final-z",
        "code_sha256": {x: sha_file(MODULE_DIR / x) for x in CODE_FILES},
        "git_commit_at_prepare": subprocess.check_output(
            ["git", "-C", str(MODULE_DIR), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "repeat_controls": False,
        "control_reuse_note": "Existing 20 fold scores from gs_018 reused with original SHA audit",
        "planned_unique_new_fold_reconstructions": 26 * 5 * 4,
        "total_fold_records_across_27_candidates": 27 * 5 * 4,
        "noise_independence_caveat": "PCM time-correlated noise can bias held-out risk",
        "fidelity_convention": "normalized Hadamard operator; summed half-squared l2",
        "failure_policy": (
            "Any failed fold invalidates its candidate. Exclude invalid candidates; "
            "if more than 9 of 27 are invalid, withhold selection and investigate "
            "numerical failure using validation data only."
        ),
        "tie_tolerance_absolute": TOL,
        "has_ground_truth": False,
        "test_set_used_for_selection": False,
    }
    atomic_json(PREPARED / "manifest.json", provenance)
    atomic_json(
        PREPARED / "candidate_grid_normalized.json",
        {
            "grid": grid,
            "source_grid_sha256": grid_sha,
        },
    )
    print(
        "PREFLIGHT_PASS",
        len(cases),
        "blind_validation_cases",
        len(grid),
        "candidates",
        26 * 5 * 4,
        "new_fold_calls",
        flush=True,
    )
    return provenance


def install_control(grid: list[dict], grid_sha: str, cases: list[dict]):
    control = grid[0]
    assert control["candidate_id"] == CONTROL_ID
    out = RESULTS / "candidates" / CONTROL_ID
    for case in cases:
        key = f'{case["category"]}_{int(case["sample_id"]):04d}'
        old_file = ORIGINAL_CONTROL / f"{key}.json"
        assert old_file.exists()
        old = json.loads(old_file.read_text())
        assert old["candidate_id"] == "gs_018"
        assert old["source_file_sha256"] == case["source_file_sha256"]
        assert old["fold_file_sha256"] == case["folds_sha256"]
        assert old["parameters"]["tau0"] == control["tau0"]
        assert len(old["fold_losses"]) == 4
        new = copy.deepcopy(old)
        new.update(
            {
                "candidate_id": CONTROL_ID,
                "parameters": control,
                "grid_sha256": grid_sha,
                "control_reused": True,
                "source_control_sha256": sha_file(old_file),
                "source_control_relative_path": str(old_file),
            }
        )
        atomic_json(out / f"{key}.json", new)
    rec = [
        json.loads(
            (out / f'{c["category"]}_{int(c["sample_id"]):04d}.json').read_text()
        )
        for c in cases
    ]
    loss = aggregate_case_losses(rec)
    orig = json.loads((ORIGINAL_CONTROL / "summary.json").read_text())
    assert abs(loss["loss"] - orig["loss"]) < 1e-14
    atomic_json(
        out / "summary.json",
        {
            "schema": "gs-expanded-ssdu-5-candidate-summary-v1",
            "candidate": control,
            "grid_sha256": grid_sha,
            **loss,
            "control_reused_from": str(ORIGINAL_CONTROL),
            "control_source_sha256": sha_file(ORIGINAL_CONTROL / "summary.json"),
            "uses_ground_truth": False,
            "test_set_access": False,
        },
    )
    print(
        "REUSE_ORIGINAL_CONTROL", CONTROL_ID, format(loss["loss"], ".12g"), flush=True
    )


def gpu_setup() -> torch.device:
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-"):
        raise RuntimeError("Set isolated CUDA_VISIBLE_DEVICES=GPU-... first")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic CuBLAS requires :4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    return torch.device("cuda:0")


def run(first: int, last: int, only: str | None):
    prepared_record()
    grid, grid_sha = get_grid()
    cases = read_cases()
    if not 1 <= first <= last <= len(grid):
        raise ValueError("Candidate index outside 1 to 27")
    if only is not None:
        if only not in {x["candidate_id"] for x in grid}:
            raise ValueError("Unknown candidate ID")
        selected = [x for x in grid if x["candidate_id"] == only]
    else:
        selected = grid[first - 1 : last]
    install_control(grid, grid_sha, cases)
    # Source/control reuse is allowed without GPU. No other candidates are.
    if selected == [grid[0]]:
        print("CONTROL_ONLY_COMPLETE_NO_NEW_GPU_WORK", flush=True)
        return
    device = gpu_setup()
    model = make_model("gs", device)
    print(
        "RUN_GS_EXPANDED",
        torch.cuda.get_device_name(0),
        "selected",
        len(selected),
        "cases",
        len(cases),
        flush=True,
    )
    for c in selected:
        label = c["candidate_id"]
        if label == CONTROL_ID:
            continue
        out = RESULTS / "candidates" / label
        if (out / "summary.json").exists():
            previous = json.loads((out / "summary.json").read_text())
            assert previous["grid_sha256"] == grid_sha
            assert previous["candidate"] == c
            assert previous["num_cases"] == 5
            print("REUSE_COMPLETE", label, flush=True)
            continue
        print("CANDIDATE_START", label, c, flush=True)
        for case in cases:
            key = f'{case["category"]}_{int(case["sample_id"]):04d}'
            existing = out / f"{key}.json"
            failure = out / f"{key}.failure.json"
            if existing.exists():
                obj = json.loads(existing.read_text())
                assert obj["candidate_id"] == label and obj["grid_sha256"] == grid_sha
                print("REUSE_CASE", label, key, flush=True)
                continue
            if failure.exists():
                print("SKIP_PREDECLARED_INVALID_CASE", label, key, flush=True)
                continue
            try:
                obj = score_single_case(case, "gs", c, device, model, grid_sha)
                assert obj["uses_ground_truth"] is False
                assert len(obj["folds"]) == 4
                assert all(
                    math.isfinite(float(x["heldout_mse"]))
                    and x["heldout_mse"] >= 0
                    and x["solver"]["accepted_iterations"] >= 1
                    for x in obj["folds"]
                )
                atomic_json(existing, obj)
                print("CASE_SCORED", label, key, obj["mean_heldout_mse"], flush=True)
            except Exception as ex:
                atomic_json(
                    failure,
                    {
                        "candidate": c,
                        "case": key,
                        "error_type": type(ex).__name__,
                        "error": str(ex),
                        "invalid_candidate": True,
                        "grid_sha256": grid_sha,
                        "no_test_metrics": True,
                    },
                )
                print(
                    "CASE_INVALID", label, key, type(ex).__name__, str(ex), flush=True
                )
        scores = []
        errors = []
        for case in cases:
            key = f'{case["category"]}_{int(case["sample_id"]):04d}'
            f = out / f"{key}.json"
            e = out / f"{key}.failure.json"
            if f.exists():
                scores.append(json.loads(f.read_text()))
            if e.exists():
                errors.append(str(e))
        if errors:
            atomic_json(
                out / "invalid_candidate.json",
                {
                    "candidate": c,
                    "grid_sha256": grid_sha,
                    "failed_case_files": errors,
                    "invalid_candidate": True,
                    "successful_cases": len(scores),
                    "no_test_metrics": True,
                },
            )
            print("CANDIDATE_INVALID", label, len(errors), "cases", flush=True)
            continue
        if len(scores) == 5:
            a = aggregate_case_losses(scores)
            atomic_json(
                out / "summary.json",
                {
                    "schema": "gs-expanded-ssdu-5-candidate-summary-v1",
                    "candidate": c,
                    "grid_sha256": grid_sha,
                    **a,
                    "case_sha256": {
                        f'{x["category"]}_{int(x["sample_id"]):04d}': sha_file(
                            out / f'{x["category"]}_{int(x["sample_id"]):04d}.json'
                        )
                        for x in scores
                    },
                    "uses_ground_truth": False,
                    "test_set_access": False,
                },
            )
            print("CANDIDATE_FINISHED", label, a["loss"], flush=True)
        else:
            print("CANDIDATE_INCOMPLETE", label, len(scores), flush=True)
    print("RUN_SHARD_COMPLETE", len(selected), flush=True)


def status() -> dict:
    grid, _ = get_grid()
    result = {
        "ready": (PREPARED / "manifest.json").exists(),
        "valid": [],
        "invalid": [],
        "incomplete": [],
        "frozen": (ROOT / "frozen_selection.json").exists(),
    }
    for c in grid:
        p = RESULTS / "candidates" / c["candidate_id"]
        if (p / "summary.json").exists():
            d = json.loads((p / "summary.json").read_text())
            result["valid"].append([c["candidate_id"], d["loss"]])
        elif (p / "invalid_candidate.json").exists():
            result["invalid"].append(c["candidate_id"])
        else:
            result["incomplete"].append(c["candidate_id"])
    print(
        "STATUS_EXPANDED_GS",
        len(result["valid"]),
        "valid",
        len(result["invalid"]),
        "invalid",
        len(result["incomplete"]),
        "incomplete",
        "frozen",
        result["frozen"],
        flush=True,
    )
    if result["valid"]:
        print("BEST_VALID_SO_FAR", sorted(result["valid"], key=lambda x: x[1])[:5])
    return result


def freeze() -> dict:
    pre = prepared_record()
    grid, gridsha = get_grid()
    check = status()
    if check["incomplete"]:
        raise RuntimeError(f"Search incomplete: {len(check['incomplete'])} candidates")
    if len(check["invalid"]) > 9:
        raise RuntimeError(
            "Over 9 invalid candidates; withhold selection, examine validation-only stability"
        )
    if not check["valid"]:
        raise RuntimeError("No usable candidate")
    rows = []
    for rank, c in enumerate(grid):
        path = RESULTS / "candidates" / c["candidate_id"] / "summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        assert d["grid_sha256"] == gridsha and d["candidate"] == c
        assert d["num_cases"] == 5 and d["folds"] == 4
        assert d["uses_ground_truth"] is False and d["test_set_access"] is False
        rows.append(
            {
                "candidate_id": c["candidate_id"],
                "grid_position": rank,
                "parameters": c,
                "SSDU_MSE": float(d["loss"]),
                "category_losses": d["category_means"],
                "summary_sha256": sha_file(path),
            }
        )
    rows.sort(key=lambda x: (x["SSDU_MSE"], x["grid_position"]))
    best = rows[0]
    close = [x for x in rows if abs(x["SSDU_MSE"] - best["SSDU_MSE"]) <= TOL]
    winner = min(close, key=lambda x: x["grid_position"])
    record = {
        "schema": "GS-PnP-SSDU-expanded-five-frozen-v1",
        "status": "frozen_before_any_additional_test_evaluation",
        "selection_set": "same_five_validation_cases",
        "selection_loss": "mean four-fold held-out squared prediction risk, equal-category weights",
        "grid_sha256": gridsha,
        "preflight_manifest_sha256": sha_file(PREPARED / "manifest.json"),
        "winner": winner,
        "ranked_valid_candidates": rows,
        "invalid_candidates": check["invalid"],
        "tested_candidate_count": len(grid),
        "no_test_metrics_used": True,
        "no_ground_truth_used": True,
        "boundary_flag": {
            "lambda_upper": winner["parameters"]["lambda"] == 0.4,
            "sigma_upper": winner["parameters"]["sigma"] == 0.15,
            "tau0_upper": winner["parameters"]["tau0"] == 75.0,
        },
    }
    atomic_json(ROOT / "frozen_selection.json", record)
    print(
        "FROZEN_EXPANDED_GS",
        winner["candidate_id"],
        winner["parameters"],
        "SSDU_MSE",
        winner["SSDU_MSE"],
        flush=True,
    )
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "run", "freeze", "status"))
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=27)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.first, args.last, args.only)
    elif args.action == "freeze":
        freeze()
    else:
        status()


if __name__ == "__main__":
    main()
