"""Freeze tune-1 PCM validation calibration cases and fixed test45 cases.

Five independently tuned validation images per category (25 tunings per solver).
Nine distinct test cases per category (45 unique test cases).
Selection never accesses ground truth or any test image quality metric.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/store/LION/ttv22/pcm/results/pcm")
PREVIOUS = BASE / "ssdu_pnp_tuning_20261108_v1"
OUTPUT = BASE / "tune1_ternary_20261109_v1"
SOURCE_VAL = PREVIOUS / "prepared/manifest.json"
SOURCE_450 = PREVIOUS / "test450/v1/merged/all_450_11_method_metrics.csv"
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
SCHEMA = "PCM-SSDU-TUNE1-TERNARY-20261109-v1"
ROOT_SEED = 2026110901
CALIBRATION_PER_CAT = 5
TEST_PER_CAT = 9


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rng_for(cat: str, split: str) -> np.random.Generator:
    value = hashlib.sha256(
        f"{SCHEMA}|{ROOT_SEED}|{cat}|{split}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(value[:8], "little"))


def frozen_json(path: Path, obj) -> None:
    data = (json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, "Frozen file has drifted: " + str(path)
    else:
        with path.open("xb") as f:
            f.write(data)


def read_test_ids_and_hashes():
    """Read only four existing baseline identity columns, not reconstruction scores."""
    data = {}
    with SOURCE_450.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        needed = ["category", "sample_id", "measurement_sha256", "pattern_sha256"]
        locations = [header.index(x) for x in needed]
        for row in reader:
            c, sid, measurement_sha256, mask_sha256 = (row[i] for i in locations)
            assert c in CATEGORIES
            key = (c, int(sid))
            assert key not in data
            assert len(measurement_sha256) == len(mask_sha256) == 64
            data[key] = {
                "category": c,
                "sample_id": int(sid),
                "measurement_sha256": measurement_sha256,
                "pattern_sha256": mask_sha256,
            }
    assert len(data) == 450
    assert Counter(x[0] for x in data) == Counter({c: 90 for c in CATEGORIES})
    return data


def parameter_ranges():
    # Search logarithms, not a fixed 15/18-point Cartesian grid.
    # Ranges are broad, dimensionless only under existing normalised PCM conventions.
    # Two coordinate passes, 3 ternary contractions per coordinate per pass.
    return {
        "schema": SCHEMA,
        "search_method": "bounded_log_coordinate_ternary_search",
        "coordinate_passes": 2,
        "ternary_contractions_per_coordinate": 3,
        "all_holdout_folds_per_candidate": 4,
        "initialization": "geometric midpoint of each range",
        "coordinate_order": {
            "admm": [["rho", "sigma"], ["sigma", "rho"]],
            "gs": [["lambda", "sigma", "tau0"], ["tau0", "sigma", "lambda"]],
        },
        "ranges": {
            "admm": {
                "rho": [0.03, 4.0],
                "sigma": [0.015, 0.18],
            },
            "gs": {
                "lambda": [0.005, 0.40],
                "sigma": [0.015, 0.15],
                "tau0": [3.0, 75.0],
            },
        },
        "algorithmic_budget_per_calibration": {
            "admm_max_candidate_evaluations": 26,
            "gs_max_candidate_evaluations": 38,
            "admm_max_fold_calls": 104,
            "gs_max_fold_calls": 152,
            "detail": "one midpoint, 2*3*2=12 coordinate probes per ADMM pass pair? ADMM 2*3*2*2=24 plus final -> 26; GS 3*3*2*2=36 plus midpoint and final -> 38",
        },
        "initialization_is_not_from_test_metrics": True,
        "failure_policy": {
            "invalid_fold": "candidate invalid, record error; never assign a finite score",
            "one_invalid_in_ternary_pair": "keep the bracket containing valid probe",
            "both_invalid": "stop shrinking that coordinate and retain best previously valid candidate",
            "all_invalid": "do not freeze selection; require validation-only numerical investigation",
            "selection": "best finite measurement-only SSDU loss among evaluated candidates",
            "tie_tolerance": 1e-10,
        },
        "fixed_method_config": {
            "admm": {
                "min_iterations": 20,
                "max_iterations": 100,
                "eps_abs": 1e-4,
                "eps_rel": 1e-3,
                "consecutive_passes": 5,
                "output": "final_x",
            },
            "gs": {
                "max_accepted_iterations": 100,
                "max_backtracking_attempts": 20,
                "gamma_backtracking": 0.1,
                "eta_backtracking": 0.9,
                "stopping_eps": 1e-6,
                "stopping_policy": "deepinv_relative_objective",
                "output": "paper_final_z",
            },
        },
        "reuse_old_pretrained_denoiser_weights": True,
        "selection_metric": "equal average of four held-out normalised Hadamard measurement MSE",
        "test_image_ground_truth_used_to_choose_parameters": False,
        "no_test_driven_refinement": True,
        "status": "prepared_not_executed",
    }


def prepare():
    assert SOURCE_VAL.is_file() and SOURCE_450.is_file()
    original = json.loads(SOURCE_VAL.read_text())
    assert original["num_cases"] == 50 and original["no_target_access"]
    assert original["test_access"] is False
    assert len(original["validation_cases"]) == 50
    val = original["validation_cases"]
    baseline = read_test_ids_and_hashes()
    calibration = []
    test45 = []
    by_cat = {}
    for cat in CATEGORIES:
        pool = sorted(
            (r for r in val if r["category"] == cat), key=lambda x: int(x["sample_id"])
        )
        assert len(pool) == 10 and len({int(x["sample_id"]) for x in pool}) == 10
        chosen = rng_for(cat, "validation_calibrations").choice(
            len(pool), size=CALIBRATION_PER_CAT, replace=False
        )
        calib = sorted(
            [pool[int(i)] for i in chosen], key=lambda x: int(x["sample_id"])
        )
        all_test = sorted(
            (x for (c, sid), x in baseline.items() if c == cat),
            key=lambda x: x["sample_id"],
        )
        assert len(all_test) == 90
        draw = rng_for(cat, "test45_fixed").choice(
            len(all_test), size=TEST_PER_CAT, replace=False
        )
        subset = sorted([all_test[int(i)] for i in draw], key=lambda x: x["sample_id"])
        assert not (
            {int(v["sample_id"]) for v in calib} & {int(v["sample_id"]) for v in subset}
        )
        for item in calib:
            original_path = Path(item["source_file"])
            fold_path = Path(item["fold_file"])
            assert "/validation/" in str(original_path) and "/test/" not in str(
                original_path
            )
            assert digest(original_path) == item["source_file_sha256"]
            assert digest(fold_path) == item["folds_sha256"]
            # The archive itself has no ground-truth/target key; verify without loading pixels.
            with np.load(original_path, allow_pickle=False) as z:
                assert "target" not in z.files and "ground_truth" not in z.files
                assert "measurement" in z.files and "sampled_indices" in z.files
            calibration.append(
                {
                    key: item[key]
                    for key in (
                        "category",
                        "sample_id",
                        "source_file",
                        "source_file_sha256",
                        "measurement_sha256",
                        "sampled_indices_sha256",
                        "fold_file",
                        "folds_sha256",
                        "heldout_counts",
                        "permutation_position_1based",
                    )
                }
            )
        test45.extend(subset)
        by_cat[cat] = {
            "calibration": [int(x["sample_id"]) for x in calib],
            "test": [int(x["sample_id"]) for x in subset],
        }
    assert len(calibration) == 25 and len(test45) == 45
    assert len({(x["category"], x["sample_id"]) for x in calibration}) == 25
    assert len({(x["category"], x["sample_id"]) for x in test45}) == 45
    frozen_json(OUTPUT / "prepared/ranges.json", parameter_ranges())
    frozen_json(
        OUTPUT / "prepared/validation_calibrations.json",
        {
            "schema": SCHEMA,
            "seed": ROOT_SEED,
            "source_sha256": digest(SOURCE_VAL),
            "source_val_manifest": str(SOURCE_VAL),
            "cases": calibration,
            "selection_by_category": by_cat,
            "number_independent_tunings_per_method": 25,
            "number_independent_tunings_per_category": 5,
            "selection_without_gt_or_test_metrics": True,
        },
    )
    frozen_json(
        OUTPUT / "prepared/test45.json",
        {
            "schema": SCHEMA,
            "seed": ROOT_SEED,
            "source_test450_csv": str(SOURCE_450),
            "test450_csv_sha256": digest(SOURCE_450),
            "case_count": 45,
            "per_category": 9,
            "cases": test45,
            "no_test_metrics_used_to_select_cases": True,
        },
    )
    frozen_json(
        OUTPUT / "prepared/protocol.json",
        {
            "schema": SCHEMA,
            "date": "2026-11-09",
            "status": "prepared_no_gpu_parameter_search",
            "test_set_450": "Original fixed 450-image test, 90 per category",
            "test_set_45": "Frozen stratified 45-image subset, 9 per category",
            "tune1_per_category": 5,
            "number_category_experiments": 5,
            "number_independent_searches_per_algorithm": 25,
            "number_algorithms": 2,
            "validation_noise": "one original fixed NPL-default acquisition at 20% sampling",
            "test_noise": "exact archived 20%-sampling NPL-default acquisition",
            "operator": "normalised 256x256 Hadamard J=8, coarse_J=6",
            "tuning_folds": 4,
            "no_image_gt_for_parameter_selection": True,
            "no_test_quality_metrics_for_tuning": True,
            "source_450_sha256": digest(SOURCE_450),
            "source_validation_manifest_sha256": digest(SOURCE_VAL),
            "referenced_previous_ssdu_report": str(
                PREVIOUS / "report/SSDU_PnP_Tuning_Report.pdf"
            ),
            "new_result_methods": ["pnp_admm_tune1", "gs_pnp_tune1"],
            "comparisons": "seven old baselines + original SSDU-tuned5 + expanded GS-SSDU + two new tune1 methods; include oracle GS Step6; preserve old 11 labels archive separately",
            "outputs": "per-calibration parameters, case-level 4-fold SSDU risks, 5x9 per category paired test reconstructions, PSNR/SSIM/HaarPSI/MSE, runtimes, visuals, report",
        },
    )
    print("PREPARED", OUTPUT, flush=True)
    print("CASE_COUNTS validation_calibrations=25 test45=45", flush=True)
    for cat in CATEGORIES:
        print(
            cat,
            "CALIBRATE",
            by_cat[cat]["calibration"],
            "TEST",
            by_cat[cat]["test"],
            flush=True,
        )
    print("NO_TUNING_OR_TEST_RECONSTRUCTION_EXECUTED", flush=True)


if __name__ == "__main__":
    prepare()
