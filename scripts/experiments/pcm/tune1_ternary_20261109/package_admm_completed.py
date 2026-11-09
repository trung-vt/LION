"""Package completed ADMM tune-1 calibration report, all scores and blind data."""
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import OUTPUT, digest

HERE = Path(__file__).resolve().parent
DEST = OUTPUT / "report/admm_tune1_completed"
ZIP = DEST / "ADMM_Tune1_Completed_Report_and_Deliverables.zip"


def make():
    selections = json.loads(
        (OUTPUT / "prepared/validation_calibrations.json").read_text()
    )["cases"]
    assert len(selections) == 25
    assert len(list((OUTPUT / "tuning/admm").glob("*/*/frozen_selection.json"))) == 25
    scorefiles = list((OUTPUT / "tuning/admm").glob("*/*/evaluations/t1_admm_*.json"))
    assert len(scorefiles) == 420
    assert not list((OUTPUT / "tuning/admm").glob("*/*/evaluations/*.failure.json"))
    assert not list((OUTPUT / "tuning/gs").glob("*/*/frozen_selection.json"))
    assert not list((OUTPUT / "test45/admm").glob("*/records/*/*.json"))
    bundle = {}
    for p in sorted((DEST / "figures").glob("*.pdf")):
        bundle["report/figures/" + p.name] = p
    for name in (
        "ADMM_Tune1_Validation_Report.pdf",
        "ADMM_Tune1_Validation_Report.tex",
        "ADMM_25_Frozen_Selections.csv",
        "ADMM_25_Category_Summary.json",
    ):
        bundle["report/" + name] = DEST / name
    for p in sorted((OUTPUT / "prepared").iterdir()):
        if p.is_file() and p.suffix in (".json", ".csv"):
            bundle["prepared/" + p.name] = p
    for p in sorted((OUTPUT / "tuning/admm").rglob("*.json")):
        bundle[str(p.relative_to(OUTPUT))] = p
    for p in sorted((OUTPUT / "tuning/admm").rglob("*.csv")):
        bundle[str(p.relative_to(OUTPUT))] = p
    logfile = OUTPUT / "logs/admm_tune1.log"
    assert logfile.is_file() and logfile.stat().st_size
    bundle["logs/admm_tune1.log"] = logfile
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".py", ".sh", ".md"):
            bundle["source/" + p.name] = p
    # Include exact blind validation input acquisitions and SSDU holdout indices.
    for case in selections:
        cat = case["category"]
        sid = int(case["sample_id"])
        acquisition = Path(case["source_file"])
        fold = Path(case["fold_file"])
        assert digest(acquisition) == case["source_file_sha256"]
        assert digest(fold) == case["folds_sha256"]
        bundle[f"blind_validation_acquisitions/{cat}_{sid:04d}.npz"] = acquisition
        bundle[f"ssdu_heldout_folds/{cat}_{sid:04d}.npz"] = fold
    assert (
        len([x for x in bundle if x.startswith("blind_validation_acquisitions/")]) == 25
    )
    assert len([x for x in bundle if x.startswith("ssdu_heldout_folds/")]) == 25
    readme = """# Completed PnP-ADMM Tune-1 validation tuning

Status: 25/25 independently frozen ADMM parameter selections.
SSDU candidate configurations: 420. Four folds each = 1,680
complete ADMM solver reconstructions. No invalid candidates.
Mean selected validation SSDU loss: 0.0043147667823385215.

This is the ADMM CALIBRATION phase, not the full five-category test.
GS-PnP tuning and test set 45 image reconstruction have not started.

Report:
  report/ADMM_Tune1_Validation_Report.pdf
  report/ADMM_Tune1_Validation_Report.tex
  report/ADMM_25_Frozen_Selections.csv
  report/ADMM_25_Category_Summary.json

Source:
  source/report_admm_completed.py
  source/run_tune1.py
  source/ternary.py
  source/score_grid.py (not copied; use parent LION branch)
  source/README.md and PLAN.md

Each frozen validation selection is in:
  tuning/admm/<category>/<calibration_id>/frozen_selection.json
  tuning/admm/<category>/<calibration_id>/search_trace.json
Candidate records:
  tuning/admm/<category>/<calibration_id>/evaluations/*.json

Original blind measurement acquisitions and fixed fold arrays are
bundled separately. No validation ground truth or test images are
included. The 45 test image IDs are in prepared/test45.json.

This source report reproduces the completed ADMM tuning only;
no claims about test PSNR should be made until subsequent evaluation.
"""
    DEST.mkdir(parents=True, exist_ok=True)
    readmepath = DEST / "README.md"
    readmepath.write_text(readme)
    bundle["README.md"] = readmepath
    assert all(p.is_file() and p.stat().st_size > 0 for p in bundle.values())
    manifest = {
        "stage": "admm_tune1_calibration_complete_only",
        "date": "2026-11-09",
        "validation_calibrations": 25,
        "unique_candidate_evaluations": 420,
        "SSDU_fold_reconstructions": 1680,
        "invalid_candidates": 0,
        "mean_selected_ssdu_mse": 0.0043147667823385215,
        "GS_search_started": False,
        "test45_run_started": False,
        "files": {
            name: {"sha256": digest(path), "size_bytes": path.stat().st_size}
            for name, path in bundle.items()
        },
    }
    mp = DEST / "archive_manifest.json"
    mp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    bundle["archive_manifest.json"] = mp
    assert not ZIP.exists()
    temp = ZIP.with_suffix(".zip.partial")
    with zipfile.ZipFile(
        temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7
    ) as archive:
        for name, p in sorted(bundle.items()):
            archive.write(p, name)
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(bundle)
        assert (
            archive.read("report/ADMM_Tune1_Validation_Report.pdf")
            == (DEST / "ADMM_Tune1_Validation_Report.pdf").read_bytes()
        )
    temp.replace(ZIP)
    print("ARCHIVE_READY", ZIP, "BYTES", ZIP.stat().st_size, "FILES", len(bundle))
    print("PDF_SHA256", digest(DEST / "ADMM_Tune1_Validation_Report.pdf"))
    print("ZIP_SHA256", digest(ZIP))


if __name__ == "__main__":
    make()
