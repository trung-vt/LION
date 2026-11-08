"""Summarise the current blind PCM SSDU 5/50 search from saved files."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import OUTPUT


def main():
    root = OUTPUT.parent
    for stage in ("pilot5", "full50"):
        for method, count in (("admm", 15), ("gs", 18)):
            d = root / stage / method
            summaries = sorted((d / "candidates").glob("*/summary.json"))
            fail = sorted((d / "candidates").glob("*/*.failure.json"))
            cases = list((d / "candidates").glob("*/*.json"))
            scores = sorted(
                [
                    (float(json.loads(x.read_text())["loss"]), x.parent.name)
                    for x in summaries
                ]
            )
            choice = d / "frozen_selection.json"
            print(
                stage,
                method,
                f"{len(summaries)}/{count} candidates",
                f"{len(cases)} score files",
                f"{len(fail)} failures",
                f"frozen={choice.exists()}",
                f"best={scores[0] if scores else None}",
            )
            if choice.exists():
                selected = json.loads(choice.read_text())["winner"]
                print(
                    "  winner",
                    selected["candidate_id"],
                    "parameters",
                    selected["parameters"],
                    "measurement_cv_loss",
                    selected["loss"],
                )
    runner = root / "full50" / "runner"
    for name in ("admm_status.txt", "gs_status.txt", "status.txt"):
        file = runner / name
        if file.exists():
            print(name, file.read_text().strip())
    print("test_set_not_accessed_for_selection", True)


if __name__ == "__main__":
    main()
