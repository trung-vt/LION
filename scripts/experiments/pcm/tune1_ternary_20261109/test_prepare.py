"""CPU-only protocol and bounded ternary search tests (no PnP GPU runs)."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    SCHEMA,
    parameter_ranges,
    read_test_ids_and_hashes,
)
from scripts.experiments.pcm.tune1_ternary_20261109.ternary import search
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen


def test_frozen_random_splits_exact():
    val = json.loads((OUTPUT / "prepared/validation_calibrations.json").read_text())
    test = json.loads((OUTPUT / "prepared/test45.json").read_text())
    assert val["schema"] == test["schema"] == SCHEMA
    assert len(val["cases"]) == 25 and len(test["cases"]) == 45
    assert Counter(x["category"] for x in val["cases"]) == Counter(
        {c: 5 for c in CATEGORIES}
    )
    assert Counter(x["category"] for x in test["cases"]) == Counter(
        {c: 9 for c in CATEGORIES}
    )
    for c in CATEGORIES:
        a = {x["sample_id"] for x in val["cases"] if x["category"] == c}
        b = {x["sample_id"] for x in test["cases"] if x["category"] == c}
        assert len(a) == 5 and len(b) == 9 and a.isdisjoint(b)
    original = read_test_ids_and_hashes()
    assert all((x["category"], x["sample_id"]) in original for x in test["cases"])


def test_ranges_and_budgets():
    x = parameter_ranges()
    assert len(x["ranges"]["admm"]) == 2
    assert len(x["ranges"]["gs"]) == 3
    assert x["algorithmic_budget_per_calibration"]["admm_max_fold_calls"] == 104
    assert x["algorithmic_budget_per_calibration"]["gs_max_fold_calls"] == 152
    assert x["coordinate_passes"] == 2
    assert x["ternary_contractions_per_coordinate"] == 3


def test_ternary_finds_unimodal_log_minimum():
    ranges = {"rho": [0.01, 10.0], "sigma": [0.005, 0.5]}
    wanted = {"rho": 0.35, "sigma": 0.06}
    seen = []

    def f(x):
        seen.append(tuple(sorted(x.items())))
        return sum((math.log10(x[k] / wanted[k])) ** 2 for k in wanted)

    result = search(f, ranges, [["rho", "sigma"], ["sigma", "rho"]], contractions=3)
    assert result["no_image_ground_truth_used"]
    assert result["winner_measurement_mse"] < 0.02
    assert abs(math.log10(result["winner"]["rho"] / wanted["rho"])) < 0.15
    assert abs(math.log10(result["winner"]["sigma"] / wanted["sigma"])) < 0.15
    assert result["evaluations"] == len(seen) == len(set(seen))
    assert result["evaluations"] <= 25


def test_ternary_gs_three_params_and_invalid_probe():
    ranges = {"lambda": [0.005, 0.4], "sigma": [0.015, 0.15], "tau0": [3.0, 75.0]}
    targets = {"lambda": 0.08, "sigma": 0.05, "tau0": 18.0}

    def f(x):
        if x["lambda"] > 0.35 and x["tau0"] > 30:
            return float("inf")
        return sum(math.log10(x[k] / targets[k]) ** 2 for k in targets)

    result = search(
        f, ranges, [["lambda", "sigma", "tau0"], ["tau0", "sigma", "lambda"]]
    )
    assert result["winner_measurement_mse"] < 0.045
    assert result["evaluations"] <= 37
    assert result["valid_evaluations"] >= 1


def test_tuning_does_not_open_test_images_before_freezing():
    # This is a deliberately blocking gate; full run should not be called now.
    with pytest.raises(RuntimeError, match="all 50 tune-1 searches"):
        assert_all_frozen()


def test_reproducible_frozen_preparation():
    from scripts.experiments.pcm.tune1_ternary_20261109.prepare import prepare

    prepare()
