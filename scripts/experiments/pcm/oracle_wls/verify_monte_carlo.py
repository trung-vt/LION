"""Verify frozen Step 4 independent Monte Carlo evidence and its Step 2/3 bindings.

Checks archive integrity, case identity, exact expected Step 3 oracle values,
statistical summaries, and optionally replays independently seeded noise draws.
Does not edit any acquisition, weight, or Monte Carlo record.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.oracle_wls.build_oracle_weights import verify_weights
from scripts.experiments.pcm.oracle_wls.float64_acquisition import _sha256
from scripts.experiments.pcm.oracle_wls.monte_carlo_validate import (
    COMPONENT_MODES,
    SCHEMA,
    _locked_noise,
    frozen_case,
    independent_mc_seed,
    noise_parameters_for_mode,
    numerical_diagnostics,
    one_realization,
    streaming_moments,
)


def verify_saved_monte_carlo(
    acquisition_root: Path,
    weight_root: Path,
    output: Path,
    *,
    expected_acquisition_manifest_sha256: str,
    expected_weight_manifest_sha256: str,
    expected_monte_carlo_manifest_sha256: str,
    replay: bool = False,
) -> dict:
    """Verify every Step 4 case without accepting unlisted or modified files."""
    acq, weights, root = map(Path, (acquisition_root, weight_root, output))
    verify_weights(
        acq,
        weights,
        expected_acquisition_manifest_sha256=expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=expected_weight_manifest_sha256,
    )
    if _sha256(root / "manifest.json") != expected_monte_carlo_manifest_sha256:
        raise ValueError("Independent Monte Carlo manifest SHA256 differs")
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if manifest["schema"] != protocol["schema"] or manifest["schema"] != SCHEMA:
        raise ValueError("Unexpected Monte Carlo archive schema")
    if (
        protocol["step2_manifest_sha256"] != expected_acquisition_manifest_sha256
        or protocol["step3_manifest_sha256"] != expected_weight_manifest_sha256
        or _sha256(root / "protocol.json") != manifest["source_protocol_sha256"]
    ):
        raise ValueError("Step 2/3 provenance or MC protocol identity mismatch")

    entries = manifest["records"]
    expected_paths = {"protocol.json", "manifest.json", "summary.csv"} | {
        record["file"] for record in entries
    }
    seen = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        if relative in seen or path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe or duplicate Monte Carlo checksum entry")
        seen[relative] = digest
        if _sha256(root / path) != digest:
            raise ValueError("Monte Carlo SHA256SUMS mismatch")
    if set(seen) != expected_paths:
        raise ValueError("Missing or extra MC manifest checksum entries")
    actual_npz = {
        str(path.relative_to(root)) for path in (root / "validation").rglob("*.npz")
    }
    if actual_npz != {r["file"] for r in entries}:
        raise ValueError("Unlisted or missing Monte Carlo results")

    with (root / "summary.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != len(entries):
        raise ValueError("Summary CSV has unexpected number of cases")
    source = json.loads((acq / "manifest.json").read_text())
    oracle = json.loads((weights / "manifest.json").read_text())
    source_lookup = {
        (r["category"], int(r["sample_id"])): r for r in source["acquisitions"]
    }
    weight_lookup = {(r["category"], int(r["sample_id"])): r for r in oracle["files"]}
    source_protocol = json.loads((acq / "protocol.json").read_text())
    noise_default = _locked_noise(source_protocol["noise"]).noise_params
    replay_count = 0
    for index, record in enumerate(entries):
        location = root / record["file"]
        if _sha256(location) != record["sha256"]:
            raise ValueError("Monte Carlo per-case checksum differs")
        with np.load(location, allow_pickle=False) as f:
            target = f["predicted_variance"].copy()
            empirical = f["empirical_variance"].copy()
            mean = f["empirical_mean"].copy()
            indices = f["sampled_indices"].copy()
            meta = json.loads(str(f["metadata_json"]))
        if (
            any(
                a.dtype != np.float64 or a.shape != (13107,)
                for a in (target, empirical, mean)
            )
            or indices.dtype != np.uint16
            or indices.shape != (13107,)
        ):
            raise ValueError("Invalid Monte Carlo vector shapes or dtypes")
        if meta["mc_seed_decimal"] != record["mc_seed_decimal"]:
            raise ValueError("Recorded Monte Carlo seed differs")
        if (
            meta["category"] != record["category"]
            or int(meta["sample_id"]) != int(record["sample_id"])
            or meta["mode"] != record["mode"]
            or int(meta["repetitions"]) != record["sample_count"]
            or meta["schema"] != SCHEMA
        ):
            raise ValueError("Monte Carlo case metadata mismatch")
        key = (record["category"], int(record["sample_id"]))
        source_record = source_lookup[key]
        weight_record = weight_lookup[key]
        if (
            meta["step2_acquisition_sha256"] != source_record["sha256"]
            or meta["step3_weight_sha256"] != weight_record["sha256"]
        ):
            raise ValueError("MC case is bound to different source or weight files")
        with np.load(weights / weight_record["file"], allow_pickle=False) as f:
            expected_v = f[
                "variance"
                if record["mode"] == "full"
                else COMPONENT_MODES[record["mode"]]
            ]
        if not np.array_equal(expected_v, target):
            raise ValueError("Predicted variance was changed after Step 3")
        diagnostics = numerical_diagnostics(
            mean,
            empirical,
            target,
            sample_count=record["sample_count"],
            mode=record["mode"],
        )
        for name, value in diagnostics.items():
            recorded = record[name]
            if isinstance(value, (int, bool, str, list)):
                if recorded != value:
                    raise ValueError(f"Changed Monte Carlo summary: {name}")
            elif not np.isclose(recorded, value, rtol=0, atol=1e-14):
                raise ValueError(f"Changed Monte Carlo numeric summary: {name}")
        if csv_rows[index]["file"] != record["file"]:
            raise ValueError("CSV row order differs from the signed manifest")
        if replay:
            acquired, x, p, q, clean, _ = frozen_case(
                acq,
                weights,
                source_record,
                weight_record,
                source_protocol,
            )
            if not np.array_equal(indices, acquired.sampled_indices):
                raise ValueError("MC results were generated using different rows")
            seed = independent_mc_seed(*key, record["mode"])
            if seed != int(record["mc_seed_decimal"]):
                raise ValueError("MC RNG stream definition has changed")
            selected_params = noise_parameters_for_mode(
                noise_default,
                record["mode"],
            )
            rng = np.random.default_rng(seed)
            n = record["sample_count"]
            replay_mean, replay_var = streaming_moments(
                (
                    one_realization(
                        p,
                        q,
                        float(acquired.i0.reshape(-1)[0]),
                        selected_params,
                        rng,
                    )
                    for _ in range(n)
                ),
                n=n,
            )
            if not (
                np.array_equal(replay_mean, mean)
                and np.array_equal(replay_var, empirical)
            ):
                raise AssertionError("Monte Carlo seeded replay differs")
            replay_count += 1
    return {
        "status": "PASS",
        "verified_files": len(entries),
        "replayed_files": replay_count,
        "manifest_sha256": expected_monte_carlo_manifest_sha256,
        "full_checks_pass": manifest["full_checks_pass"],
        "full_checks_total": manifest["full_checks_total"],
        "control_checks_pass": manifest["control_checks_pass"],
        "control_checks_total": manifest["control_checks_total"],
        "predeclared_checks_all_passed": manifest["predeclared_checks_all_passed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-acquisition-manifest-sha256", required=True)
    parser.add_argument("--expected-weight-manifest-sha256", required=True)
    parser.add_argument("--expected-monte-carlo-manifest-sha256", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify_saved_monte_carlo(
        args.acquisitions,
        args.weights,
        args.output,
        expected_acquisition_manifest_sha256=args.expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=args.expected_weight_manifest_sha256,
        expected_monte_carlo_manifest_sha256=args.expected_monte_carlo_manifest_sha256,
        replay=args.replay,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
