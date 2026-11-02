"""Reviewer follow-up to Step 6: repeat matched validation at opt_tol=1e-6.

Preserves the original 50-pair opt_tol=1e-3 archive unchanged. This run
uses the same frozen A,y, oracle weights, calibrated epsilon, original
periodized db4 wavelet and CPU SPGL1 with a larger iteration budget.
It reports SPGL1's native relative duality gap for every solve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.experiments.pcm.oracle_wls.compare_ols_wls_step6 import (
    CALIBRATION_MANIFEST_SHA256,
    ORDERED_CATEGORIES,
    _write_manifest,
    paired_case,
    summarize_results,
)
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import validate_frozen_archives

ITERATIONS = 1500
OPT_TOL = 1e-6


def run(acquisitions, weights, calibration, output):
    source, oracle, calib, dest = map(
        Path, (acquisitions, weights, calibration, output)
    )
    if dest.exists():
        raise FileExistsError("Tighter-tolerance experiment never overwrites results")
    validate_frozen_archives(source, oracle)
    if _sha256(calib / "manifest.json") != CALIBRATION_MANIFEST_SHA256:
        raise ValueError("Step 5 calibration hash differs")
    base = json.loads((source / "manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    cases = base["acquisitions"]
    if len(cases) != 50 or base["test_count"] != 0:
        raise ValueError("The frozen 50 validation images are required")
    pilot, rest = [], []
    for category in ORDERED_CATEGORIES:
        group = sorted(
            [r for r in cases if r["category"] == category],
            key=lambda r: int(r["permutation_position_1based"]),
        )
        if len(group) != 10:
            raise ValueError("Incomplete validation category")
        pilot.extend(group[:2])
        rest.extend(group[2:])
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records = []
    for stage, arr in (("pilot", pilot), ("full", rest)):
        for r in arr:
            records.append(
                paired_case(
                    source,
                    oracle,
                    calib,
                    CALIBRATION_MANIFEST_SHA256,
                    r,
                    protocol,
                    dest,
                    iters=ITERATIONS,
                    opt_tol=OPT_TOL,
                    stage=stage,
                )
            )
        if stage == "pilot":
            _, summary = summarize_results(records)
            write_json_new(dest / "pilot_summary.json", summary)
            print(
                "TIGHT_PILOT_PAIRS",
                summary["ALL"]["fully_successful"],
                "/10",
                flush=True,
            )
    _, summary = summarize_results(records)
    manifest = _write_manifest(
        dest,
        records,
        summary,
        iters=ITERATIONS,
        opt_tol=OPT_TOL,
        calibration_sha=CALIBRATION_MANIFEST_SHA256,
        stage_status="tighter_tolerance_reviewer_followup",
    )
    root = summary["ALL"]
    gaps = {}
    for mode in ("ols", "wls"):
        current = [
            r["methods"][mode].get("solver", {}).get("relative_duality_gap")
            for r in records
        ]
        gaps[mode] = {
            "missing": sum(x is None for x in current),
            "max": max(x for x in current if x is not None),
            "median": sorted(x for x in current if x is not None)[len(current) // 2],
        }
    print(
        json.dumps(
            {
                "status": "FINISHED",
                "cases": len(records),
                "fully_successful": root["fully_successful"],
                "mean_delta_psnr": root["metric_differences"]["psnr"],
                "gap_summaries": gaps,
                "manifest_sha256": _sha256(dest / "manifest.json"),
                "source_code_sha256": _sha256(Path(__file__)),
            },
            indent=2,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run(a.acquisitions, a.weights, a.calibration, a.output)


if __name__ == "__main__":
    main()
