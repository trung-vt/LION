"""Freeze SSDU-selected hyperparameters from measurement-only validation scores.

The tuner is intentionally not allowed to see reference images or test scores.
The separate 5- and 50-image selections use the exact same candidate grid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    OUTPUT,
    sha_file,
    immutable_json,
    CATEGORIES,
)


def frozen_stage(stage: str) -> dict:
    if stage not in ("pilot5", "full50"):
        raise ValueError("Unknown SSDU stage")
    root = OUTPUT.parent
    if stage == "full50":
        for method in ("admm", "gs"):
            if not (root / "pilot5" / method / "frozen_selection.json").is_file():
                raise RuntimeError(
                    "Both Pilot5 selections must be frozen before Full50"
                )
    grid_path = OUTPUT / "candidate_grid.json"
    grid_sha = sha_file(grid_path)
    configs = json.loads(grid_path.read_text())
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert manifest["status"] == "prepared_no_solver_tuning"
    assert manifest["no_target_access"] and not manifest["test_access"]
    size = 5 if stage == "pilot5" else 50
    winners = {}
    for method in ("admm", "gs"):
        candidates = configs[method]
        assert len(candidates) == (15 if method == "admm" else 18)
        all_scores = []
        for rank, candidate in enumerate(candidates):
            path = (
                root
                / stage
                / method
                / "candidates"
                / candidate["candidate_id"]
                / "summary.json"
            )
            if not path.exists():
                raise RuntimeError(f"Cannot freeze incomplete {stage}/{method}: {path}")
            r = json.loads(path.read_text())
            if r["grid_sha256"] != grid_sha or r["candidate"] != candidate:
                raise RuntimeError("Grid/parameters changed since SSDU score")
            if (r["num_cases"], r["folds"]) != (size, 4):
                raise RuntimeError("Missing SSDU validation folds/cases")
            assert r["uses_ground_truth"] is False and r["test_set_access"] is False
            assert set(r["category_means"]) == set(CATEGORIES)
            assert r["cases_per_category"] == {c: size // 5 for c in CATEGORIES}
            all_scores.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "original_grid_order": rank,
                    "parameters": candidate,
                    "loss": r["loss"],
                    "per_category_ssdu_mse": r["category_means"],
                    "candidate_summary_sha256": sha_file(path),
                }
            )
        all_scores.sort(key=lambda x: (x["loss"], x["original_grid_order"]))
        best = all_scores[0]
        # Tiny tolerance, predeclared. Ties favor the original grid ordering.
        within = [r for r in all_scores if abs(r["loss"] - best["loss"]) <= 1e-10]
        selected = min(within, key=lambda x: x["original_grid_order"])
        record = {
            "schema": "pcm-ssdu-blind-global-pnp-selection-v1",
            "status": "frozen_before_any_new_test_evaluation",
            "method_id": f"{'PnP-ADMM' if method=='admm' else 'GS-PnP'}-tuned-{'5' if stage=='pilot5' else '50'}",
            "selection_set": stage,
            "method": method,
            "num_images": size,
            "num_folds_per_image": 4,
            "validation_archive_manifest_sha256": manifest[
                "validation_archive_manifest_sha256"
            ],
            "preparation_manifest_sha256": sha_file(OUTPUT / "manifest.json"),
            "candidate_grid_sha256": grid_sha,
            "source_pnp_numerical_model": "existing historical PnP/GS DeepInv source via SSDU blind adapter",
            "selection_basis": "minimum equal-category mean of four-fold heldout measurement MSE",
            "tolerance_absolute_for_tie": 1e-10,
            "winner": selected,
            "ranked_candidates": all_scores,
            "all_candidates_evaluated": len(candidates),
            "used_ground_truth": False,
            "accessed_test_set": False,
            "no_test_tuning_permitted": True,
        }
        immutable_json(root / stage / method / "frozen_selection.json", record)
        winners[method] = {
            "method_id": record["method_id"],
            "parameters": selected["parameters"],
            "SSDU_MSE": selected["loss"],
            "winner_path": str(root / stage / method / "frozen_selection.json"),
        }
        print(
            "FROZEN",
            stage,
            method,
            "WINNER",
            selected["candidate_id"],
            "SSDU_MSE",
            selected["loss"],
            "RANK_2",
            all_scores[1]["loss"],
            flush=True,
        )
    immutable_json(
        root / stage / "frozen_pair.json",
        {
            "schema": "pcm-ssdu-blind-frozen-two-method-selections-v1",
            "status": "frozen_before_new_test_evaluation",
            "validation_image_count": size,
            "full_candidate_grid_both_methods": True,
            "winners": winners,
            "test_set_not_read": True,
            "ground_truth_not_read": True,
        },
    )
    return winners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot5", "full50"], required=True)
    args = parser.parse_args()
    frozen_stage(args.stage)


if __name__ == "__main__":
    main()
