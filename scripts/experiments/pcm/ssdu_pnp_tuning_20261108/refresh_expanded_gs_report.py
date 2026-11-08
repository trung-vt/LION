"""Refresh the expanded GS-PnP report from validation-only records.

Updates just report input fragments. The editable main LaTeX source is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import (
    ROOT,
    PREPARED,
    RESULTS,
    REPORT,
    get_grid,
    prepared_record,
    status,
)


def make():
    manifest = prepared_record()
    grid, digest = get_grid()
    states = status()
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in grid:
        name = candidate["candidate_id"]
        root = RESULTS / "candidates" / name
        summary = root / "summary.json"
        if summary.exists():
            obj = json.loads(summary.read_text())
            assert obj["candidate"] == candidate
            assert obj["grid_sha256"] == digest
            err = f'{obj["loss"]:.8f}'
            state = "reused control" if name == "gs_exp_001" else "complete"
        elif (root / "invalid_candidate.json").exists():
            err = "--"
            state = "invalid"
        else:
            err = "--"
            state = "not run"
        rows.append(
            r"\texttt{"
            + name.replace("_", r"\_")
            + "} & "
            + f'{candidate["lambda"]:.2f} & {candidate["sigma"]:.3f} & '
            + f'{candidate["tau0"]:.4f} & {err} & {state} '
            + r"\\"
        )
    (REPORT / "candidate_rows.tex").write_text(
        "\n".join(rows) + "\n" + r"\bottomrule" + "\n"
    )

    frozen = ROOT / "frozen_selection.json"
    if frozen.exists():
        x = json.loads(frozen.read_text())
        assert x["no_test_metrics_used"] and x["no_ground_truth_used"]
        win = x["winner"]
        selection = (
            r"The candidate search is complete and the selection is frozen. "
            + "The validation-selected winner is "
            + r"\texttt{"
            + win["candidate_id"].replace("_", r"\_")
            + "}, with "
            + r"\(\lambda="
            + f'{win["parameters"]["lambda"]:.3f}'
            + r"\), "
            + r"\(\sigma="
            + f'{win["parameters"]["sigma"]:.3f}'
            + r"\), "
            + r"\(\tau_0="
            + f'{win["parameters"]["tau0"]:.4f}'
            + r"\), and "
            + "held-out MSE "
            + f'{win["SSDU_MSE"]:.8f}'
            + "."
        )
    else:
        selection = (
            "The expanded parameter sweep has not been completed or frozen. "
            + "The old control's held-out MSE was 0.0043654339. "
            + "No higher-value configuration has yet been selected. "
            + "This is a preparation report, not a result claiming improved GS-PnP."
        )
    overview = (
        r"\textbf{Run status:} "
        + f'{len(states["valid"])} valid, {len(states["invalid"])} invalid, '
        + f'{len(states["incomplete"])} untested/incomplete candidates. '
        + selection
        + "\n"
    )
    (REPORT / "current_results.tex").write_text(overview)
    (REPORT / "status.json").write_text(
        json.dumps(
            {
                "status": "frozen" if frozen.exists() else "prepared_or_in_progress",
                "valid_candidates": len(states["valid"]),
                "invalid_candidates": len(states["invalid"]),
                "incomplete_candidates": len(states["incomplete"]),
                "candidate_grid_sha256": digest,
                "no_ground_truth_or_test_selection": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        "REFRESHED_REPORT",
        REPORT,
        "valid",
        len(states["valid"]),
        "pending",
        len(states["incomplete"]),
        flush=True,
    )


if __name__ == "__main__":
    make()
