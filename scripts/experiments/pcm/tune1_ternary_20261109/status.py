"""Read-only progress report for the PCM tune-1, test45 experiment."""
import json
from pathlib import Path
from scripts.experiments.pcm.tune1_ternary_20261109.prepare import OUTPUT, CATEGORIES


def main():
    cal = json.loads((OUTPUT / "prepared/validation_calibrations.json").read_text())[
        "cases"
    ]
    test = json.loads((OUTPUT / "prepared/test45.json").read_text())["cases"]
    print("PREPARED", len(cal), "validation calibrations", len(test), "test45 images")
    total = 0
    for method in ("admm", "gs"):
        frozen = []
        cases = 0
        for record in cal:
            cat = record["category"]
            sid = int(record["sample_id"])
            root = OUTPUT / "tuning" / method / cat / f"{sid:04d}"
            if (root / "frozen_selection.json").exists():
                frozen.append((cat, sid))
            cases += len(
                list(
                    (OUTPUT / "test45" / method / cat / "records" / f"{sid:04d}").glob(
                        "*.json"
                    )
                )
            )
        print(
            "METHOD",
            method,
            "FROZEN",
            len(frozen),
            "/25",
            "TEST_RECONSTRUCTIONS",
            cases,
            "/225",
        )
        total += len(frozen)
    nbaselines = sum(
        len(list((OUTPUT / "test45/baselines/raw" / m).glob("*.npy")))
        for m in (
            "zero_filled",
            "spgl1_bp",
            "spgl1_bpdn",
            "pnp_admm_oracle",
            "tv",
            "utv_c60",
        )
    )
    print("BASELINE_RECONSTRUCTION_ARRAYS", nbaselines, "/270")
    print("METRIC_REPORT_READY", (OUTPUT / "analysis/test45_summary.json").exists())
    print(
        "TEST_45_QUALITATIVE_GALLERIES",
        len(list((OUTPUT / "analysis/galleries").glob("*_9_pages.pdf"))),
        "/5",
    )
    print("TOTAL_FROZEN_PARAMETER_SELECTIONS", total, "/50")


if __name__ == "__main__":
    main()
