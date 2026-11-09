"""Ternary search for the best global C60 TV-map multiplier, Perovskite 0031.

The frozen measurement, sampling mask, model, predicted positive parameter map,
initial image, PDHG step sizes and iteration count are unchanged. This is an
oracle single-image analysis, not an unbiased held-out parameter selection.
An interval-width tolerance of 1e-4 is conditional on PSNR unimodality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm import perovskite0031_c60_rho_sweep_20261109_v2 as base

ROOT = base.ROOT
OUT = ROOT / "perovskite0031_c60_ternary_20261109_v1"
LO = 0.0001
HI = 10.0
TOL_RHO = 1e-4
METRIC_KEYS = ("psnr", "ssim", "haarpsi", "mse")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, data, lam, gt, y, indices, baseline, _zero, benchmark = base.setup()
    with np.load(
        ROOT / "perovskite0031_utv_c60_rho_sweep_20261109_v2/reconstructions.npz"
    ) as previous:
        np.testing.assert_array_equal(gt, previous["ground_truth"])
        np.testing.assert_array_equal(y, previous["measurement"])
        np.testing.assert_array_equal(indices, previous["indices"])
        np.testing.assert_allclose(
            lam.squeeze().cpu().numpy(),
            previous["original_lambda_map"],
            rtol=0,
            atol=0,
        )
    print("FROZEN_INPUT_VALIDATED", base.digest(y), base.digest(indices), flush=True)

    evaluations: dict[str, dict] = {}
    search_steps: list[dict] = []

    def evaluate(rho: float, note: str = "ternary") -> dict:
        rho = float(rho)
        assert LO <= rho <= HI, rho
        key = rho.hex()
        if key in evaluations:
            return evaluations[key]
        image, runtime = base.reconstruct(model, data, lam, rho)
        score = base.quality(image, gt)
        record = {
            "rho": rho,
            "rho_hex": key,
            "role": note,
            "elapsed_seconds": runtime,
            **score,
        }
        evaluations[key] = record
        print(
            "EVAL",
            len(evaluations),
            f"rho={rho:.12f}",
            f"PSNR={score['psnr']:.9f}",
            f"SSIM={score['ssim']:.7f}",
            f"HaarPSI={score['haarpsi']:.7f}",
            flush=True,
        )
        return record

    # Fixed benchmark validation. rho=1 must reproduce the stored C60 baseline.
    ref = evaluate(1.0, "archived_baseline")
    image_ref, _ = base.reconstruct(model, data, lam, 1.0)
    np.testing.assert_array_equal(image_ref, baseline)
    assert abs(ref["psnr"] - float(benchmark["C_psnr"])) < 1e-5

    # A small diagnostic sweep checks whether there are obvious additional
    # peaks outside the previously explored 0.01-1 range.
    diagnostic_rhos = [
        LO,
        0.001,
        0.01,
        0.05,
        0.1,
        0.3,
        0.45,
        0.7,
        1.0,
        1.5,
        2.5,
        5.0,
        HI,
    ]
    diagnostics = [evaluate(r, "diagnostic") for r in diagnostic_rhos]
    peak_diag = max(diagnostics, key=lambda d: d["psnr"])
    print("DIAGNOSTIC_PEAK", peak_diag["rho"], peak_diag["psnr"], flush=True)

    # Standard ternary search on LINEAR rho. Stop on the final rho bracket width.
    left, right = LO, HI
    iteration = 0
    while right - left > TOL_RHO:
        iteration += 1
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        s1 = evaluate(m1)
        s2 = evaluate(m2)
        if s1["psnr"] < s2["psnr"]:
            left = m1
            decision = "discard_left"
        elif s1["psnr"] > s2["psnr"]:
            right = m2
            decision = "discard_right"
        else:
            left, right = m1, m2
            decision = "exact_tie_discard_ends"
        search_steps.append(
            {
                "iteration": iteration,
                "m1": m1,
                "m2": m2,
                "psnr1": s1["psnr"],
                "psnr2": s2["psnr"],
                "remaining_left": left,
                "remaining_right": right,
                "remaining_width": right - left,
                "decision": decision,
            }
        )
        print(
            "STEP",
            iteration,
            f"[{left:.10f},{right:.10f}]",
            f"width={right - left:.10f}",
            flush=True,
        )
        assert iteration <= 200

    probe_rhos = [
        left,
        left + (right - left) / 4,
        (left + right) / 2,
        left + 3 * (right - left) / 4,
        right,
    ]
    probes = [evaluate(rho, "final_bracket_probe") for rho in probe_rhos]
    winner = max(evaluations.values(), key=lambda x: x["psnr"])
    winner_in_final_bracket = left - 1e-12 <= winner["rho"] <= right + 1e-12
    chosen = max(probes, key=lambda x: x["psnr"])
    print("WINNER", winner, flush=True)
    print("FINAL_BRACKET", left, right, "width", right - left, flush=True)
    print("BRACKET_WINNER", chosen, flush=True)

    # If the best observed value is outside the final bracket, retain it but
    # explicitly report the inconsistency instead of claiming global optimality.
    image_best, _ = base.reconstruct(model, data, lam, winner["rho"])
    score_best = base.quality(image_best, gt)
    np.testing.assert_allclose(score_best["psnr"], winner["psnr"], rtol=0, atol=1e-5)
    psnr_probe_range = max(p["psnr"] for p in probes) - min(p["psnr"] for p in probes)

    with (OUT / "evaluations.csv").open("w", newline="") as f:
        rows = list(evaluations.values())
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "ternary_iterations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(search_steps[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(search_steps)

    np.savez_compressed(
        OUT / "optimal_reconstruction.npz",
        reconstruction=image_best,
        ground_truth=gt,
        rho=np.float64(winner["rho"]),
        baseline_C60=baseline,
        lambda_map=lam.squeeze().cpu().numpy(),
        measurement=y,
        indices=indices,
    )
    result = {
        "case": "Perovskite_0031",
        "scope": "single-image ground-truth oracle hyperparameter tuning",
        "search_algorithm": "standard linear-domain ternary search",
        "rho_search_range": [LO, HI],
        "tolerance_definition": "final interval width in rho <= 1e-4; not a global-optimum or PSNR error guarantee",
        "tolerance_rho": TOL_RHO,
        "final_bracket": [left, right],
        "final_bracket_width": right - left,
        "final_bracket_psnr_range_dB": psnr_probe_range,
        "iterations": iteration,
        "unique_evaluations": len(evaluations),
        "best_observed": winner,
        "best_observed_in_final_bracket": winner_in_final_bracket,
        "best_in_bracket": chosen,
        "best_improvement_vs_baseline_dB": winner["psnr"] - ref["psnr"],
        "diagnostic_candidates": diagnostics,
        "measurement_sha256": base.digest(y),
        "pattern_sha256": base.digest(indices),
        "checkpoint_sha256": base.digest_file(base.CHECKPOINT),
        "config_sha256": base.digest_file(base.CONFIG),
        "lambda_map_sha256": base.digest(lam.squeeze().cpu().numpy()),
        "pdhg_iterations": base.MODEL_ITERATIONS,
        "fixed": [
            "measurement",
            "sampled pattern indices",
            "C60 checkpoint",
            "CG input",
            "positive lambda map prior to scaling",
            "PDHG step sizes",
            "initial image",
            "512 iterations",
        ],
        "assumption_warning": "Ternary search is globally valid only for a unimodal PSNR-vs-rho objective; diagnostic evaluations do not prove unimodality.",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "TERNARY_COMPLETE",
        json.dumps(
            {
                "rho_best": winner["rho"],
                "psnr_best": winner["psnr"],
                "baseline_psnr": ref["psnr"],
                "final_bracket": [left, right],
                "width": right - left,
                "probe_psnr_span": psnr_probe_range,
                "evaluations": len(evaluations),
                "winner_in_bracket": winner_in_final_bracket,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
