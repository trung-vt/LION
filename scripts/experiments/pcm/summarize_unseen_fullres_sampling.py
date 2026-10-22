#!/usr/bin/env python3
"""Summarize completed full-resolution PCM sweep cases without selecting winners.

This can safely run while the two tmux jobs are still writing other cases.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def gather(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for marker in sorted(root.glob("*/**/COMPLETED")):
        case_dir = marker.parent
        if not case_dir.name.startswith("s_"):
            continue
        # root/specimen/profile/condition/rate_case/COMPLETED
        specimen = case_dir.parents[2].name
        profile = case_dir.parents[1].name
        condition = case_dir.parent.name
        candidates = [
            p
            for p in case_dir.iterdir()
            if p.is_dir()
            and p.name.endswith("_matched")
            and (p / "config.yaml").is_file()
        ]
        if len(candidates) != 1:
            print(f"WARNING: ambiguous completed run at {case_dir}")
            continue
        run = candidates[0]
        expected = int(marker.read_text().strip())
        results = sorted(run.glob("*/results.csv"))
        if len(results) != expected:
            print(f"WARNING: result count changed at {run}")
            continue
        for source in results:
            df = pd.read_csv(source)
            if len(df) != 1:
                print(f"WARNING: expected one row in {source}")
                continue
            entry = df.iloc[0].to_dict()
            records.append(
                {
                    "specimen": specimen,
                    "profile": profile,
                    "condition": condition,
                    "run_dir": str(run),
                    "method": source.parent.name,
                    **entry,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("LION_TTV22_PATH", "/mnt/store_ext4/store/LION/ttv22")
        )
        / "pcm/results/20261021_unseen_fullres_sampling_sweep_v1",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically save the completed-only CSV as summary_completed.csv.",
    )
    args = parser.parse_args()
    root = args.root
    if not root.is_dir():
        raise FileNotFoundError(root)
    for specimen in ("cigs", "perovskite"):
        folder = root / specimen
        marks = list(folder.glob("**/COMPLETED"))
        failed = list(folder.glob("**/FAILED"))
        started = folder / "current.txt"
        print(
            f"{specimen}: {len(marks)}/192 completed cases, "
            f"{len(failed)} currently marked failed; "
            f"current: {started.read_text().strip() if started.exists() else 'none'}"
        )
        if failed:
            for path in failed[:5]:
                print("  FAILED", path.parent)
    result = gather(root)
    if not result.empty:
        print(f"Verified {len(result)} per-method rows from completed cases.")
        print(result.groupby(["specimen", "profile", "condition"]).size().to_string())
        if args.write:
            filename = root / "summary_completed.csv"
            tmp = root / "summary_completed.csv.tmp"
            result.to_csv(tmp, index=False)
            tmp.replace(filename)
            print("Saved:", filename)
    else:
        print("No completed rows available yet.")


if __name__ == "__main__":
    main()
