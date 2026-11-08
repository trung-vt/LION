"""CPU-only tests for expanded GS-PnP SSDU preparation."""
import json
import pytest
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108 import expanded_gs as e


def test_candidate_grid():
    grid, digest = e.get_grid()
    assert len(grid) == 27 and len(digest) == 64
    assert grid[0]["candidate_id"] == "gs_exp_001"
    assert len({(a["lambda"], a["sigma"], a["tau0"]) for a in grid}) == 27
    assert len({a["candidate_id"] for a in grid}) == 27
    assert grid[0]["lambda"] == 0.1
    assert grid[0]["sigma"] == 0.075
    assert grid[0]["tau0"] == 100 / 3
    assert all(a["output"] == "paper_final_z" for a in grid)


def test_validation_only_cases():
    original = e.original_control()
    cases = e.read_cases()
    assert len(cases) == 5
    assert {x["category"] for x in cases} == set(e.CATEGORIES)
    assert original["winner"]["parameters"]["lambda"] == 0.1
    assert all("/validation/" in x["source_file"] for x in cases)


def test_no_freezing_before_completed_sweep():
    if not (e.ROOT / "frozen_selection.json").exists():
        with pytest.raises((RuntimeError, AssertionError)):
            e.freeze()


def test_prepared_manifest():
    if (e.PREPARED / "manifest.json").exists():
        prepared = e.prepared_record()
        assert prepared["candidate_count"] == 27
        assert prepared["planned_unique_new_fold_reconstructions"] == 520
        assert not prepared["has_ground_truth"]
        assert not prepared["test_set_used_for_selection"]
