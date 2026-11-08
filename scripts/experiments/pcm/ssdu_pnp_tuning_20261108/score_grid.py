"""Resumable blind measurement-only SSDU parameter sweep on frozen PCM validation.

Run separate processes on one isolated GPU per PnP method.
No image GT, image quality metrics, or held-out TEST case are ever accessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    OUTPUT,
    CATEGORIES,
    sha_file,
    immutable_json,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
    read_blind_acquisition,
    make_fold_inputs,
    heldout_mse,
    aggregate_case_losses,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct,
)

PROJECT = OUTPUT.parent
KEYS = ("admm", "gs")


def load_frozen_context(stage: str, method: str):
    if stage not in ("pilot5", "full50") or method not in KEYS:
        raise ValueError("Unknown stage/method")
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert manifest["status"] == "prepared_no_solver_tuning"
    assert manifest["no_target_access"] and not manifest["test_access"]
    assert len(manifest["validation_cases"]) == 50
    assert (OUTPUT / "fidelity_audit.json").is_file()
    assert (
        json.loads((OUTPUT / "fidelity_audit.json").read_text())["status"] == "passed"
    )
    grid_path = OUTPUT / "candidate_grid.json"
    grid = json.loads(grid_path.read_text())
    assert grid["uses_ground_truth"] is False
    if stage == "full50":
        for meth in KEYS:
            p = PROJECT / "pilot5" / meth / "frozen_selection.json"
            if not p.exists():
                raise RuntimeError(
                    "Full50 cannot start before both Pilot5 methods are frozen"
                )
    records = manifest["validation_cases"]
    if stage == "pilot5":
        pilot = {(r["category"], r["sample_id"]) for r in manifest["pilot5"]}
        records = [r for r in records if (r["category"], r["sample_id"]) in pilot]
    assert len(records) == (5 if stage == "pilot5" else 50)
    expected = 1 if stage == "pilot5" else 10
    assert all(
        sum(r["category"] == cat for r in records) == expected for cat in CATEGORIES
    )
    return records, grid[method], sha_file(grid_path)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("Existing candidate score differs: " + str(path))
    else:
        tmp = path.with_suffix(path.suffix + ".partial")
        with tmp.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def score_single_case(
    case: dict,
    method: str,
    candidate: dict,
    device: torch.device,
    network: torch.nn.Module,
    grid_sha: str,
) -> dict:
    assert case["source_file"].startswith(
        str(
            PROJECT.parent
            / "pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final/validation"
        )
    )
    source = Path(case["source_file"])
    assert sha_file(source) == case["source_file_sha256"]
    y, idx = read_blind_acquisition(source)
    assert hashlib.sha256(y.tobytes()).hexdigest() == case["measurement_sha256"]
    assert (
        hashlib.sha256(idx.astype(np.uint16).tobytes()).hexdigest()
        == case["sampled_indices_sha256"]
    )
    fold_path = Path(case["fold_file"])
    assert sha_file(fold_path) == case["folds_sha256"]
    folds = []
    with np.load(fold_path, allow_pickle=False) as f:
        assert len(f.files) == 4
        for k in range(4):
            held = f[f"heldout_{k}"].copy()
            op, yy, opheld, yheld = make_fold_inputs(y, idx, held, device)
            t0 = time.perf_counter()
            image, diag = reconstruct(method, op, yy, network, candidate)
            assert image.shape == (1, 1, 256, 256)
            loss = heldout_mse(image, opheld, yheld)
            if not np.isfinite(loss) or loss < 0:
                raise FloatingPointError("Invalid SSDU heldout loss")
            folds.append(
                {
                    "fold": k,
                    "fit_rows": int(yy.numel()),
                    "heldout_rows": int(yheld.numel()),
                    "heldout_mse": float(loss),
                    "wall_seconds_with_operators": time.perf_counter() - t0,
                    "solver": diag,
                }
            )
            del op, yy, opheld, yheld, image
    return {
        "schema": "ssdu-pnp-blind-case-score-v1",
        "stage_independent_case": True,
        "candidate_id": candidate["candidate_id"],
        "parameters": candidate,
        "method": method,
        "category": case["category"],
        "sample_id": case["sample_id"],
        "permutation_position_1based": case["permutation_position_1based"],
        "source_file_sha256": case["source_file_sha256"],
        "fold_file_sha256": case["folds_sha256"],
        "grid_sha256": grid_sha,
        "fold_losses": [r["heldout_mse"] for r in folds],
        "folds": folds,
        "mean_heldout_mse": float(np.mean([r["heldout_mse"] for r in folds])),
        "uses_ground_truth": False,
        "test_set_access": False,
    }


def run(
    stage: str,
    method: str,
    only_candidate: str | None = None,
    first: int = 1,
    last: int | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith(
        "GPU-"
    ), "Isolate GPU by UUID"
    torch.set_num_threads(4)
    assert (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    ), "Require deterministic CuBLAS"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    device = torch.device("cuda:0")
    cases, grid, grid_sha = load_frozen_context(stage, method)
    if only_candidate is not None:
        assert first == 1 and last is None
        assert only_candidate in {x["candidate_id"] for x in grid}
        grid = [x for x in grid if x["candidate_id"] == only_candidate]
    else:
        upper = len(grid) if last is None else last
        assert 1 <= first <= upper <= len(grid), (first, upper, len(grid))
        grid = grid[first - 1 : upper]
    destination = PROJECT / stage / method
    network = make_model(method, device)
    print(
        "STAGE",
        stage,
        "METHOD",
        method,
        "DEVICE",
        torch.cuda.get_device_name(0),
        "CANDIDATES",
        len(grid),
        "CASES",
        len(cases),
        "FOLDS",
        len(cases) * 4,
        flush=True,
    )
    for ix, candidate in enumerate(grid, 1):
        label = candidate["candidate_id"]
        output = destination / "candidates" / label
        finished = output / "summary.json"
        if finished.exists():
            prior = json.loads(finished.read_text())
            assert prior["candidate"] == candidate and prior["grid_sha256"] == grid_sha
            assert prior["num_cases"] == len(cases)
            print(
                "REUSE_COMPLETE_CANDIDATE",
                stage,
                method,
                label,
                prior["loss"],
                flush=True,
            )
            continue
        print(
            "CANDIDATE_START",
            stage,
            method,
            ix,
            len(grid),
            label,
            {k: v for k, v in candidate.items() if k not in ("candidate_id",)},
            flush=True,
        )
        started = time.perf_counter()
        for j, case in enumerate(cases, 1):
            key = f'{case["category"]}_{int(case["sample_id"]):04d}'
            target = output / f"{key}.json"
            if target.exists():
                existing = json.loads(target.read_text())
                assert existing["candidate_id"] == label
                assert existing["source_file_sha256"] == case["source_file_sha256"]
                assert existing["fold_file_sha256"] == case["folds_sha256"]
                assert existing["grid_sha256"] == grid_sha
                print("REUSE", stage, method, label, key, flush=True)
                continue
            if stage == "full50":
                pilot_file = (
                    PROJECT / "pilot5" / method / "candidates" / label / f"{key}.json"
                )
                if pilot_file.exists():
                    saved = json.loads(pilot_file.read_text())
                    assert saved["grid_sha256"] == grid_sha
                    assert saved["source_file_sha256"] == case["source_file_sha256"]
                    assert saved["fold_file_sha256"] == case["folds_sha256"]
                    assert len(saved["fold_losses"]) == 4
                    atomic_json(target, saved)
                    print("REUSE_PILOT_FOLDS", stage, method, label, key, flush=True)
                    continue
            try:
                result = score_single_case(
                    case, method, candidate, device, network, grid_sha
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                error_file = output / f"{key}.failure.json"
                atomic_json(
                    error_file,
                    {
                        "candidate_id": label,
                        "category": case["category"],
                        "sample_id": case["sample_id"],
                        "error": message,
                        "grid_sha256": grid_sha,
                        "uses_ground_truth": False,
                        "test_set_access": False,
                    },
                )
                print(
                    "CANDIDATE_CASE_FAILED",
                    stage,
                    method,
                    label,
                    key,
                    message,
                    flush=True,
                )
                continue
            atomic_json(target, result)
            print(
                "CASE",
                stage,
                method,
                label,
                j,
                len(cases),
                key,
                "SSDU_MSE",
                format(result["mean_heldout_mse"], ".9g"),
                flush=True,
            )
        scores = []
        for case in cases:
            f = output / f'{case["category"]}_{int(case["sample_id"]):04d}.json'
            if f.exists():
                scores.append(json.loads(f.read_text()))
        if len(scores) == len(cases):
            agg = aggregate_case_losses(scores, num_folds=4)
            summary = {
                "schema": "ssdu-pnp-candidate-summary-v1",
                "stage": stage,
                "method": method,
                "candidate": candidate,
                "grid_sha256": grid_sha,
                **agg,
                "uses_ground_truth": False,
                "test_set_access": False,
                "wall_seconds_current_process": time.perf_counter() - started,
                "result_case_file_sha256": {
                    f'{x["category"]}_{x["sample_id"]:04d}': sha_file(
                        output / f'{x["category"]}_{int(x["sample_id"]):04d}.json'
                    )
                    for x in scores
                },
            }
            atomic_json(output / "summary.json", summary)
            print(
                "CANDIDATE_COMPLETE",
                stage,
                method,
                label,
                "loss",
                format(agg["loss"], ".9g"),
                "seconds",
                round(time.perf_counter() - started, 3),
                flush=True,
            )
        else:
            print(
                "CANDIDATE_INCOMPLETE",
                stage,
                method,
                label,
                len(scores),
                len(cases),
                flush=True,
            )
    print(
        "METHOD_STAGE_COMPLETE",
        stage,
        method,
        "FINISHED",
        sum(
            (destination / "candidates" / x["candidate_id"] / "summary.json").exists()
            for x in grid
        ),
        "/",
        len(grid),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot5", "full50"), required=True)
    parser.add_argument("--method", choices=KEYS, required=True)
    parser.add_argument("--only-candidate", default=None)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=None)
    args = parser.parse_args()
    run(args.stage, args.method, args.only_candidate, args.first, args.last)


if __name__ == "__main__":
    main()
