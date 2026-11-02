"""Read-only Step 6 pilot replay for both historical solver-tolerance archives.

Re-solves the ten originally preselected pilot validation pairs with the
current checked Step 5 solver and original A,y, and quantifies differences
from archived images. It does NOT claim byte-exact replay if the Step 5
solver code differs from the archived historical commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    write_json_new,
)
from scripts.experiments.pcm.oracle_wls.weighted_bpdn import (
    load_calibration_record,
    load_step5_case,
    solve_step5_case,
    validate_frozen_archives,
)
from scripts.experiments.pcm.oracle_wls.review_step6_results import (
    ORIGINAL_ANCHOR,
    TIGHT_ANCHOR,
    EXPECTED_CALIBRATION,
)


def replay(acq, weights, calibration, original, tight, output):
    acq, weights, calibration, original, tight, out = map(
        Path, (acq, weights, calibration, original, tight, output)
    )
    if out.exists():
        raise FileExistsError("Cannot overwrite Step 6 reviewer pilot replay")
    validate_frozen_archives(acq, weights)
    if _sha256(calibration / "manifest.json") != EXPECTED_CALIBRATION:
        raise ValueError("Calibration archive identity changed")
    results = {
        "schema": "pcm-step6-independent-pilot-replay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_run_1e_minus3": ORIGINAL_ANCHOR,
        "tight_run_1e_minus6": TIGHT_ANCHOR,
        "code_sha256": _sha256(Path(__file__)),
        "warning": (
            "The Step 5 solver was changed after historical reconstruction: "
            "weighted RHS now uses float64 multiplication before float32 "
            "conversion. Byte equality is therefore not guaranteed. "
            "Numerical image differences, statuses and metrics are recorded."
        ),
        "archives": {},
    }
    torch.set_num_threads(2)
    for name, path, sha, iters, opt_tol in (
        ("historical_1e_minus3", original, ORIGINAL_ANCHOR, 120, 1e-3),
        ("tighter_1e_minus6", tight, TIGHT_ANCHOR, 1500, 1e-6),
    ):
        if _sha256(path / "manifest.json") != sha:
            raise ValueError("Historical or tighter manifest SHA changed")
        manifest = json.loads((path / "manifest.json").read_text())
        pilot = [r for r in manifest["cases"] if r["stage"] == "pilot"]
        if len(pilot) != 10:
            raise ValueError("Exactly 10 predeclared pilot cases required")
        values = []
        for record in pilot:
            cat, sid = record["category"], int(record["sample_id"])
            case = load_step5_case(acq, weights, cat, sid, verify_all=False)
            calib = load_calibration_record(
                calibration, cat, sid, expected_manifest_sha256=EXPECTED_CALIBRATION
            )
            row = {"category": cat, "sample_id": sid, "methods": {}}
            for method in ("ols", "wls"):
                epsilon = calib[f"epsilon_{method}_95"]
                image, info = solve_step5_case(
                    case,
                    method=method,
                    epsilon=epsilon,
                    iters=iters,
                    opt_tol=opt_tol,
                )
                stored = path / record["methods"][method]["reconstruction_file"]
                with np.load(stored, allow_pickle=False) as data:
                    old = data["image"].copy()
                current = image.cpu().detach().numpy()
                if old.shape != current.shape:
                    raise ValueError("Pilot replay changed reconstruction shape")
                difference = current.astype(np.float64) - old.astype(np.float64)
                denom = max(float(np.linalg.norm(old)), 1e-12)
                row["methods"][method] = {
                    "historical_solver_status": record["methods"][method]["solver"][
                        "solver_status_name"
                    ],
                    "current_solver_status": info["solver_status_name"],
                    "current_relative_duality_gap": info.get("relative_duality_gap"),
                    "historical_iterations": int(
                        record["methods"][method]["iterations"]
                    ),
                    "current_iterations": int(info["iterations"]),
                    "relative_image_l2_difference": float(
                        np.linalg.norm(difference) / denom
                    ),
                    "image_max_absolute_difference": float(np.abs(difference).max()),
                    "image_array_byte_identical": bool(np.array_equal(old, current)),
                    "current_reconstruction_image_sha256": hashlib.sha256(
                        current.tobytes()
                    ).hexdigest(),
                    "archived_reconstruction_file_sha256": record["methods"][method][
                        "reconstruction_sha256"
                    ],
                }
            values.append(row)
            print(
                name,
                cat,
                sid,
                "rel_ols",
                f"{row['methods']['ols']['relative_image_l2_difference']:.3e}",
                "rel_wls",
                f"{row['methods']['wls']['relative_image_l2_difference']:.3e}",
                flush=True,
            )
        differences = [
            r["methods"][method]["relative_image_l2_difference"]
            for r in values
            for method in ("ols", "wls")
        ]
        results["archives"][name] = {
            "pilot_pairs": len(values),
            "replayed_images": 2 * len(values),
            "byte_identical_images": sum(
                r["methods"][method]["image_array_byte_identical"]
                for r in values
                for method in ("ols", "wls")
            ),
            "mean_relative_image_l2_difference": float(np.mean(differences)),
            "max_relative_image_l2_difference": float(np.max(differences)),
            "cases": values,
        }
    out.mkdir(parents=True, exist_ok=False)
    write_json_new(out / "pilot_replay.json", results)
    print("PILOT_REPLAY_SHA256=" + _sha256(out / "pilot_replay.json"))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquisitions", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--tight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    replay(
        a.acquisitions,
        a.weights,
        a.calibration,
        a.original,
        a.tight,
        a.output,
    )


if __name__ == "__main__":
    main()
