"""Freeze exact validation-only SSDU data splits and candidate grids.

This module reads archived noisy measurements and sampled indices only.
No image ground truth and no test acquisitions are used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
SOURCE = ROOT / "pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final"
OUTPUT = ROOT / "ssdu_pnp_tuning_20261108_v1/prepared"
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
PILOT_POSITIONS = (901, 903, 905, 907, 909)
J = 8
COARSE_J = 6
COARSE_COUNT = 4**COARSE_J
MEASUREMENT_COUNT = 13107
BLOCK = 128
FOLDS = 4
SCHEMA = "pcm-ssdu-pnp-5-vs-50-prepared-v1"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seed(*parts: str) -> int:
    blob = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "little") % (
        2**32
    )


def immutable_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if path.exists():
        assert path.read_bytes() == payload, "Frozen file drift: " + str(path)
    else:
        with path.open("xb") as f:
            f.write(payload)


def candidate_grids() -> dict:
    admm = []
    for rho, sigma in itertools.product(
        (0.1, 0.25, 0.5, 1.0, 2.0), (0.03, 0.05, 0.075)
    ):
        admm.append(
            {
                "candidate_id": f"admm_{len(admm)+1:03d}",
                "rho": rho,
                "sigma": sigma,
                "min_iterations": 20,
                "max_iterations": 100,
                "eps_abs": 1e-4,
                "eps_rel": 1e-3,
                "consecutive_passes": 5,
            }
        )
    gs = []
    for lam, sigma, tau0 in itertools.product(
        (0.01, 0.03, 0.1), (0.03, 0.05, 0.075), (10.0, 100.0 / 3.0)
    ):
        gs.append(
            {
                "candidate_id": f"gs_{len(gs)+1:03d}",
                "lambda": lam,
                "sigma": sigma,
                "tau0": tau0,
                "max_accepted_iterations": 100,
                "max_backtracking_attempts": 20,
                "gamma_backtracking": 0.1,
                "eta_backtracking": 0.9,
                "stopping_policy": "deepinv_relative_objective",
                "stopping_eps": 1e-6,
                "output": "paper_final_z",
            }
        )
    assert len(admm) == 15 and len(gs) == 18
    return {
        "schema": SCHEMA,
        "admm": admm,
        "gs": gs,
        "selection_loss": "mean held-out normalized-operator squared residual; equal category weights",
        "pilot_and_full_use_same_grid": True,
        "uses_ground_truth": False,
        "fidelity_scaling_status": "UNRESOLVED: audit unscaled summed L2 fold-to-full transfer BEFORE first sweep",
    }


def make_folds(
    indices: np.ndarray,
    *,
    category: str,
    sample_id: int,
    block_size: int = BLOCK,
    n_folds: int = FOLDS,
) -> tuple[np.ndarray, ...]:
    """Split acquired fine-scale rows by level-stratified temporal blocks."""
    assert category in CATEGORIES and sample_id >= 1
    idx = np.asarray(indices, dtype=np.int64)
    assert idx.shape == (MEASUREMENT_COUNT,)
    assert np.array_equal(idx[:COARSE_COUNT], np.arange(COARSE_COUNT))
    assert np.array_equal(idx, np.sort(idx)) and len(np.unique(idx)) == len(idx)
    assert idx[COARSE_COUNT] >= COARSE_COUNT and idx[-1] < 4**J
    folds = [[] for _ in range(n_folds)]
    for level in (7, 8):
        positions = np.flatnonzero((idx >= 4 ** (level - 1)) & (idx < 4**level))
        chunks = [
            positions[k : k + block_size] for k in range(0, len(positions), block_size)
        ]
        order = np.random.default_rng(
            seed(SCHEMA, category, str(sample_id), str(level))
        ).permutation(len(chunks))
        for j, chunk_index in enumerate(order):
            folds[j % n_folds].append(chunks[int(chunk_index)])
    result = tuple(np.sort(np.concatenate(parts).astype(np.uint16)) for parts in folds)
    all_held = np.concatenate(result).astype(np.int64)
    assert np.array_equal(np.sort(all_held), np.arange(COARSE_COUNT, len(idx)))
    for h in result:
        assert h.size > 0 and h[0] >= COARSE_COUNT
        assert 1600 < h.size < 3000, (category, sample_id, h.size)
        fit = np.setdiff1d(np.arange(len(idx)), h, assume_unique=True)
        assert len(fit) + len(h) == len(idx)
        assert np.array_equal(fit[:COARSE_COUNT], np.arange(COARSE_COUNT))
        assert np.intersect1d(fit, h).size == 0
    return result


def prepare(output: Path = OUTPUT) -> dict:
    mf = SOURCE / "manifest.json"
    sp = SOURCE / "protocol.json"
    assert mf.is_file() and sp.is_file()
    manifest = json.loads(mf.read_text())
    assert (
        manifest["status"]
        in ("complete", "completed", "reviewed_final", "frozen", "final")
        or manifest["frozen"] is True
    )
    assert manifest["validation_count"] == 50 and manifest["test_count"] == 0
    records = [
        r for r in manifest["acquisitions"] if r["file"].startswith("validation/")
    ]
    assert len(records) == 50
    assert Counter(r["category"] for r in records) == Counter(
        {c: 10 for c in CATEGORIES}
    )
    by_pos = {
        (r["category"], int(r["permutation_position_1based"])): r for r in records
    }
    assert len(by_pos) == 50 and set(by_pos) == {
        (c, p) for c in CATEGORIES for p in range(901, 911)
    }
    groups = []
    for cat in CATEGORIES:
        for p in range(901, 911):
            r = by_pos[cat, p]
            f = SOURCE / r["file"]
            assert f.is_file() and sha_file(f) == r["sha256"], f
            with np.load(f, allow_pickle=False) as a:
                assert "ground_truth" not in a.files and "target" not in a.files
                assert "measurement" in a.files and "sampled_indices" in a.files
                y = np.array(a["measurement"], copy=True)
                idx = np.array(a["sampled_indices"], copy=True)
            assert y.shape == (1, 1, MEASUREMENT_COUNT) and y.dtype == np.float64
            assert idx.shape == (MEASUREMENT_COUNT,)
            sid = int(r["sample_id"])
            holdouts = make_folds(idx, category=cat, sample_id=sid)
            dst = output / "folds" / f"{cat}_{sid:04d}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            arrays = {f"heldout_{j}": v for j, v in enumerate(holdouts)}
            if dst.exists():
                with np.load(dst, allow_pickle=False) as old:
                    assert set(old.files) == set(arrays)
                    assert all(
                        np.array_equal(old[k], v) for k, v in arrays.items()
                    ), "Fold drift " + str(dst)
            else:
                with dst.open("xb") as f2:
                    np.savez_compressed(f2, **arrays)
            groups.append(
                {
                    "category": cat,
                    "sample_id": sid,
                    "permutation_position_1based": p,
                    "source_file": str(f),
                    "source_file_sha256": r["sha256"],
                    "measurement_sha256": digest_bytes(y.tobytes()),
                    "sampled_indices_sha256": digest_bytes(idx.tobytes()),
                    "fold_file": str(dst),
                    "heldout_counts": [int(len(h)) for h in holdouts],
                    "folds_sha256": sha_file(dst),
                }
            )
    pilot = []
    for cat, p in zip(CATEGORIES, PILOT_POSITIONS, strict=True):
        case = next(
            c
            for c in groups
            if c["category"] == cat and c["permutation_position_1based"] == p
        )
        pilot.append(
            {
                "category": cat,
                "sample_id": case["sample_id"],
                "permutation_position_1based": p,
            }
        )
    assert len(pilot) == 5 and len({x["sample_id"] for x in pilot}) == 5
    frozen = {
        "schema": SCHEMA,
        "status": "prepared_no_solver_tuning",
        "validation_archive_path": str(SOURCE),
        "validation_archive_manifest_sha256": sha_file(mf),
        "validation_archive_protocol_sha256": sha_file(sp),
        "source_precision": "float64 archived measurement; fit solver may cast to float32",
        "num_cases": 50,
        "categories": list(CATEGORIES),
        "num_per_category": 10,
        "pilot5": pilot,
        "folds": FOLDS,
        "block_size": BLOCK,
        "coarse_patterns_preserved": COARSE_COUNT,
        "acquired_patterns_per_image": MEASUREMENT_COUNT,
        "fine_patterns_heldout_once": MEASUREMENT_COUNT - COARSE_COUNT,
        "sampling_ratio": 0.2,
        "noise": "NPL default; one fixed acquisition per image",
        "pattern": "normalized J8 multilevel, coarse_J6",
        "no_target_access": True,
        "test_access": False,
        "status_next": "Step 1: numerical adapter/fidelity scale and noise-correlation checks",
        "paths": {
            "pilot": str(output.parent / "pilot5"),
            "full50": str(output.parent / "final50"),
            "test450": str(output.parent / "test450"),
        },
        "validation_cases": groups,
    }
    immutable_json(output / "candidate_grid.json", candidate_grids())
    immutable_json(output / "manifest.json", frozen)
    immutable_json(output / "pilot5.json", pilot)
    print("PREPARED", output, flush=True)
    print(
        "CASES", len(groups), "PILOT", pilot, "FOLDS", len(groups) * FOLDS, flush=True
    )
    print("CANDIDATES_ADMM", 15, "GS", 18, "STATUS", "no_tuning_started", flush=True)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
