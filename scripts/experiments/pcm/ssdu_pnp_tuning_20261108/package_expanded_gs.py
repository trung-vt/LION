"""Bundle the finished expanded GS SSDU sweep, test metrics, and report.

Copies every candidate's validation-only per-fold record and all 450 TEST
diagnostic/metric JSON files, but leaves full 450 raw npy reconstructions on
tuni to avoid oversize GitHub artifacts. Five displayed raw examples included.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs import (
    ROOT,
    REPORT,
    RESULTS,
    PREPARED,
    GRID_FILE,
    sha_file,
)

SOURCE = Path(
    "/home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108/scripts/experiments/pcm/ssdu_pnp_tuning_20261108"
)
OLD = ROOT.parent / "test450/v1/visual5"
TEST = ROOT / "test450"
OUTPUT = ROOT / "GS_PnP_Expanded_SSDU_Full_Report_and_Results.zip"


def package():
    selected = ROOT / "frozen_selection.json"
    complete = TEST / "analysis/analysis_complete.json"
    fullpdf = REPORT / "GS_PnP_Expanded_SSDU_Tuning_Full_Report.pdf"
    fulltex = REPORT / "GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex"
    assert selected.is_file() and complete.is_file()
    assert fullpdf.exists() and fulltex.exists()
    assert len(list((TEST / "records").glob("*/*.json"))) == 450
    metadata = json.loads(complete.read_text())
    assert metadata["test_cases"] == 450 and metadata["no_test_tuning"]
    bundle = {}
    for p in (
        ROOT / "README.md",
        ROOT / "PLAN.md",
        GRID_FILE,
        selected,
        ROOT / "preparation_manifest.json",
        PREPARED / "manifest.json",
        PREPARED / "candidate_grid_normalized.json",
    ):
        bundle[str(p.relative_to(ROOT))] = p
    for p in (
        sorted(REPORT.glob("*.tex"))
        + sorted(REPORT.glob("*.pdf"))
        + sorted(REPORT.glob("*.json"))
    ):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((REPORT / "figures").glob("*.pdf")):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((REPORT / "pretest_prepared_v1").glob("*")):
        if p.is_file() and p.stat().st_size:
            bundle[str(p.relative_to(ROOT))] = p
    for p in sorted(TEST.rglob("*.json")):
        # All 450 test metrics and diagnostics, plus summaries and hash manifests.
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((TEST / "analysis").glob("*.csv")):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((TEST / "analysis/figures").glob("*.pdf")):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((RESULTS / "candidates").glob("*/*.json")):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((ROOT / "logs").glob("*.log")):
        bundle[str(p.relative_to(ROOT))] = p
    for p in sorted((TEST / "logs").glob("*.log")):
        if p.stat().st_size:
            bundle[str(p.relative_to(ROOT))] = p
    for p in sorted(SOURCE.glob("*.py")):
        # Project-wide scripts also bundled to capture source provenance.
        bundle["source/" + p.name] = p
    for p in (
        SOURCE / "run_expanded_gs.sh",
        ROOT / "resume_test450.sh",
        ROOT / "run_test450_after_freeze.sh",
    ):
        if p.exists():
            bundle["source/" + p.name] = p
    for p in sorted((OLD / "arrays").glob("*.npz")):
        bundle["old_visual_raw/" + p.name] = p
    for p in sorted(OLD.glob("*.json")):
        bundle["old_visual_raw/" + p.name] = p
    for p in sorted((TEST / "raw").glob("*.npy")):
        name = p.stem
        if name in (
            "CIGS_0279",
            "Organic_0331",
            "Perovskite_0786",
            "Reference_0279",
            "Silicone_0370",
        ):
            bundle["new_expanded_visual_raw/" + p.name] = p
    assert (
        len([p for k, p in bundle.items() if k.startswith("new_expanded_visual_raw/")])
        == 5
    )
    assert all(p.is_file() and p.stat().st_size > 0 for p in bundle.values())
    manifest = {
        "status": "completed_and_frozen_before_test",
        "date": "2026-11-09",
        "selection_sha256": sha_file(selected),
        "selection": json.loads(selected.read_text())["winner"],
        "full_report_pdf_sha256": sha_file(fullpdf),
        "retrospective_450_test": True,
        "test_metrics_usable_for_tuning": False,
        "raw_full_450_numpy_arrays_kept_on_tuni": str(TEST / "raw"),
        "excluded_450_full_raw_numpy_arrays_from_zip": True,
        "files_sha256": {name: sha_file(p) for name, p in bundle.items()},
    }
    package_manifest = ROOT / "full_deliverables_manifest.json"
    package_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    bundle["full_deliverables_manifest.json"] = package_manifest
    assert not OUTPUT.exists(), OUTPUT
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
    ) as z:
        for name, p in sorted(bundle.items()):
            z.write(p, arcname=name)
    with zipfile.ZipFile(OUTPUT) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(bundle)
        assert z.read(str(fullpdf.relative_to(ROOT))) == fullpdf.read_bytes()
    print(
        "FULL_DELIVERABLES_READY",
        OUTPUT,
        OUTPUT.stat().st_size,
        "files",
        len(bundle),
        "raw_examples",
        5,
        flush=True,
    )


if __name__ == "__main__":
    package()
