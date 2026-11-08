"""CPU-only preparation tests. No GT or test-case loads."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    make_folds,
    candidate_grids,
    COARSE_COUNT,
    MEASUREMENT_COUNT,
)


def _fake_mask():
    coarse = np.arange(COARSE_COUNT, dtype=np.int64)
    rest = np.arange(COARSE_COUNT, 4**8, dtype=np.int64)
    selected = np.random.default_rng(810).choice(
        rest, MEASUREMENT_COUNT - COARSE_COUNT, replace=False
    )
    return np.sort(np.concatenate([coarse, selected]))


def test_fold_partition_and_leakage():
    idx = _fake_mask()
    folds = make_folds(idx, category="CIGS", sample_id=733)
    assert len(folds) == 4
    fine = set(range(COARSE_COUNT, MEASUREMENT_COUNT))
    assert set(np.concatenate(folds).tolist()) == fine
    for h in folds:
        assert not set(h.tolist()).intersection(range(COARSE_COUNT))
        fit = np.setdiff1d(np.arange(MEASUREMENT_COUNT), h)
        assert len(fit) + len(h) == MEASUREMENT_COUNT
        assert np.array_equal(fit[:COARSE_COUNT], np.arange(COARSE_COUNT))


def test_folds_deterministic():
    idx = _fake_mask()
    a = make_folds(idx, category="Reference", sample_id=768)
    b = make_folds(idx, category="Reference", sample_id=768)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_grid_independent_and_equal_budget():
    d = candidate_grids()
    assert len(d["admm"]) == 15 and len(d["gs"]) == 18
    assert len(set(x["candidate_id"] for x in d["admm"])) == 15
    assert len(set(x["candidate_id"] for x in d["gs"])) == 18
    combinations = {(x["lambda"], x["sigma"], x["tau0"]) for x in d["gs"]}
    assert len(combinations) == 18
    assert d["pilot_and_full_use_same_grid"] is True
    assert d["uses_ground_truth"] is False


def test_no_target_dependency_in_preparation_source():
    p = Path(__file__).with_name("prepare.py").read_text()
    assert 'a["ground_truth"]' not in p
    assert 'a["target"]' not in p
    assert "test/" not in p


def test_frozen_ssdu_risk_from_measurements_only(tmp_path):
    import torch
    from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
        read_blind_acquisition,
        heldout_mse,
    )

    m = np.full((1, 1, MEASUREMENT_COUNT), 1.0, dtype=np.float64)
    idx = _fake_mask().astype(np.uint16)
    filename = tmp_path / "case.npz"
    np.savez_compressed(filename, measurement=m, sampled_indices=idx)
    measurement, indices = read_blind_acquisition(filename)
    assert np.array_equal(measurement, m)
    assert np.array_equal(indices, idx)
    predicted = torch.tensor([[[1.0, 2.0, 3.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float64)
    assert abs(heldout_mse(predicted, lambda x: x, target) - 5 / 3) < 1e-12
    contaminated = tmp_path / "invalid.npz"
    np.savez_compressed(
        contaminated, measurement=m, sampled_indices=idx, target=np.zeros((256, 256))
    )
    import pytest

    with pytest.raises(ValueError, match="must not contain"):
        read_blind_acquisition(contaminated)


def test_equal_category_selection_scores():
    from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
        aggregate_case_losses,
    )

    categories = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
    rows = [
        {"category": c, "fold_losses": [float(i)] * 4}
        for i, c in enumerate(categories, 1)
    ]
    s = aggregate_case_losses(rows)
    assert s["loss"] == 3
    assert s["num_cases"] == 5
    doubled = aggregate_case_losses(rows + rows)
    assert doubled["loss"] == s["loss"]
