"""Package the completed fixed-random Tune-1 test45 results for download.

All 45 raw reconstructions per new method and 270 baseline replay arrays
remain on tuni. The portable ZIP includes all metadata and scores, all
figures and five illustrative sets of raw arrays, plus runnable source.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    digest,
)
from scripts.experiments.pcm.tune1_ternary_20261109.random_one_analysis import (
    ROOT,
    METHODS,
)
from scripts.experiments.pcm.tune1_ternary_20261109.random_one_report import (
    REPORT,
    NAME,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"
ZIP = OUT / "PCM_Test45_Random_Tune1_All_Results_and_Source.zip"


def make():
    choices = json.loads((ROOT / "selection.json").read_text())
    assert choices["methods_for_comparison"] == [k for k, n in METHODS]
    assert choices["excluded_methods"] == ["pnp_admm_oracle", "gs_pnp_expanded5"]
    assert (
        "GALLERY_ALL_45_PAGES_COMPLETE"
        in (ROOT / "logs/gallery_status.txt").read_text()
    )
    assert "BOTH_45_COMPLETE" in (ROOT / "logs/status.txt").read_text()
    assert (
        "BASELINES_COMPLETE" in (ROOT / "logs/baselines_resume_status.txt").read_text()
    )
    pdf = REPORT / (NAME + ".pdf")
    tex = REPORT / (NAME + ".tex")
    gallery = ROOT / "analysis/galleries/Test45_All_10_Methods_45_Pages.pdf"
    assert pdf.is_file() and tex.is_file() and gallery.is_file()
    import subprocess

    assert "Pages:           45" in subprocess.check_output(
        ["pdfinfo", str(gallery)], text=True
    )
    files = {
        "report/PCM_Test45_Random_Tune1_Comparison.pdf": pdf,
        "report/PCM_Test45_Random_Tune1_Comparison.tex": tex,
        "report/report_data.json": REPORT / "report_data.json",
        "gallery/Test45_All_10_Methods_45_Pages.pdf": gallery,
        "results/test45_10_methods_per_image.csv": ROOT
        / "analysis/test45_10_methods_per_image.csv",
        "results/test45_10_methods_summary.json": ROOT
        / "analysis/test45_10_methods_summary.json",
        "results/paired_method_differences.json": ROOT
        / "analysis/paired_method_differences.json",
        "selection/random_one_per_category.json": ROOT / "selection.json",
        "selection/test45.json": OUTPUT / "prepared/test45.json",
        "selection/validation_calibrations.json": OUTPUT
        / "prepared/validation_calibrations.json",
        "selection/ternary_ranges.json": OUTPUT / "prepared/ranges.json",
    }
    for x in (ROOT / "analysis/figures").glob("*.pdf"):
        files["figures/" + x.name] = x
    for x in (ROOT / "analysis/galleries").glob("*_9_Pages.pdf"):
        files["gallery/categories/" + x.name] = x
    for src in HERE.glob("*.py"):
        if src.name in (
            "random_one_test45.py",
            "random_one_analysis.py",
            "random_one_gallery.py",
            "random_one_report.py",
            "random_one_package.py",
            "test45_run.py",
            "replay_baselines_random_one.py",
            "replay_c60.py",
            "prepare.py",
            "ternary.py",
            "run_tune1.py",
        ):
            files["source/" + src.name] = src
    for name in (
        "run_random_one_45.sh",
        "replay_references.sh",
        "resume_gs_only.sh",
        "resume_baselines.sh",
        "generate_gallery_45.sh",
    ):
        q = ROOT / name
        if q.exists():
            files["source/" + name] = q
    for m in ("admm", "gs"):
        for c in CATEGORIES:
            v = choices["selected_calibrations"][m][c]
            cid = int(v["calibration_sample_id"])
            p = OUTPUT / "tuning" / m / c / f"{cid:04d}" / "frozen_selection.json"
            assert digest(p) == v["selection_file_sha256"]
            files[f"selected_calibrations/{m}/{c}_{cid:04d}.json"] = p
            for f in (ROOT / "recon" / m / c / "records" / f"{cid:04d}").glob("*.json"):
                files[f"new_test_records/{m}/{c}_{cid:04d}/{f.name}"] = f
    assert sum(k.startswith("new_test_records/") for k in files) == 90
    # Per-method historical provenance for all 45 cases, including replay drift audits.
    historical = OUTPUT / "test45/baselines/provenance"
    for name in (
        "zero_filled",
        "spgl1_bp",
        "spgl1_bpdn",
        "pnp_admm_oracle",
        "tv",
        "utv_c60",
    ):
        ps = list((historical / name).glob("*.json"))
        assert len(ps) == 45, (name, len(ps))
        for f in ps:
            files["baseline_replay_provenance/" + name + "/" + f.name] = f
    # Store actual raw arrays for five representative test images, not all 45.
    test = json.loads((OUTPUT / "prepared/test45.json").read_text())["cases"]
    chosen = [next(x for x in test if x["category"] == c) for c in CATEGORIES]
    original = OUTPUT.parent / "ssdu_pnp_tuning_20261108_v1"
    baseline_root = OUTPUT / "test45/baselines/raw"
    for x in chosen:
        cat = x["category"]
        sid = int(x["sample_id"])
        for method, display in METHODS:
            if method in ("pnp_admm_tune1", "gs_pnp_tune1"):
                family = "admm" if method == "pnp_admm_tune1" else "gs"
                cid = choices["selected_calibrations"][family][cat][
                    "calibration_sample_id"
                ]
                f = (
                    ROOT
                    / "recon"
                    / family
                    / cat
                    / "raw"
                    / f"{cid:04d}"
                    / f"{sid:04d}.npy"
                )
            elif method == "pnp_admm_ssdu5":
                f = baseline_root / "pnp_admm_oracle" / f"{cat}_{sid:04d}.npy"
            elif method == "gs_pnp_oracle":
                f = original / "test450/v1/raw/gs_oracle_step6" / f"{cat}_{sid:04d}.npy"
            elif method == "gs_pnp_ssdu5":
                f = original / "test450/v1/raw/gs_ssdu" / f"{cat}_{sid:04d}.npy"
            else:
                f = baseline_root / method / f"{cat}_{sid:04d}.npy"
            assert f.is_file(), f
            files[f"raw_examples/{cat}_{sid:04d}/{method}.npy"] = f
    assert sum(k.startswith("raw_examples/") for k in files) == 50
    for f in (ROOT / "logs").glob("*.txt"):
        if f.stat().st_size > 0:
            files["logs/" + f.name] = f
    OUT.mkdir(parents=True, exist_ok=True)
    readme = r"""# Random Tune-1 on PCM test set 45, 9 November 2026

Status: COMPLETE. tuni confirmed. Exactly ten requested method labels.
One random validation-tuned parameter set per category and PnP algorithm,
frozen before examining test metrics. Five ADMM and five GS-PnP selected
configs. 45 matched 20% NPL-noisy acquisitions evaluated per method.

Requested references: zero-fill, SPGL1-BP, SPGL1-BPDN, PnP-ADMM-tuned-5,
GS-PnP validation-selected oracle, scalar TV, original UTV-C60,
GS-PnP-tuned-5, PnP-ADMM Tune-1, GS-PnP Tune-1. The old ADMM oracle and
expanded GS SSDU-5 were excluded, as requested.

- report/: editable TeX and completed PDF.
- gallery/: 45-page visual PDF plus 5 nine-page category PDFs.
- results/: 45-image ten-method metrics CSV, per-category/overall JSON,
  paired differences and comparisons.
- selected_calibrations/: ten frozen source JSON selections.
- new_test_records/: 90 raw reconstructed image metrics and solver records.
- raw_examples/: five category examples with ten raw arrays each.
- baseline_replay_provenance/: 45 records per replayed old method,
  including SPGL1-BP reconstruction differences.
- source/: scripts and frozen execution commands.

Full arrays for all 90 new test reconstructions and 270 historical
replays remain on tuni, without loss:
  /store/LION/ttv22/pcm/results/pcm/tune1_ternary_20261109_v1/
  test45/random_one_per_category_v1/recon/
  /store/LION/ttv22/pcm/results/pcm/tune1_ternary_20261109_v1/
  test45/baselines/raw/

The portable ZIP does not include all these large raw arrays,
but includes 50 representative raw arrays and complete scores/hashes.

Technical caveat: SPGL1-BP numerical replay differs slightly from the
original archived 450-case metrics for several cases. Comparison
tables retain the original archived baseline metrics. Visual
captions for BP are scored from the displayed replayed arrays.
No method parameters were changed to match a test score.

One random calibration choice per category is insufficient to
establish calibration selection variability. The report presents
this as one fixed retrospective comparison, not general optimality.
"""
    readme_path = OUT / "README.txt"
    readme_path.write_text(readme)
    files["README.txt"] = readme_path
    for k, p in files.items():
        assert p.exists() and p.stat().st_size > 0, (k, p)
    metadata = {
        "study": "PCM test45 random one calibration per category",
        "status": "complete",
        "date": "2026-11-09",
        "n_test_unique": 45,
        "n_method_labels": 10,
        "new_reconstructions": 90,
        "random_choices": 10,
        "image_rate": 0.2,
        "noise": "NPL-default same stored acquisition",
        "test_selected_without_metrics": True,
        "full_raw_tuni": str(OUTPUT / "test45"),
        "summary": str(ROOT / "analysis/test45_10_methods_summary.json"),
        "source_hashes": {
            k: {"sha256": digest(p), "bytes": p.stat().st_size}
            for k, p in files.items()
        },
    }
    manifest = OUT / "manifest.json"
    manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    files["manifest.json"] = manifest
    if ZIP.exists():
        raise FileExistsError(ZIP)
    tmp = ZIP.with_suffix(".zip.partial")
    with zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7
    ) as z:
        for name, p in sorted(files.items()):
            z.write(p, name)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
        assert (
            z.read("report/PCM_Test45_Random_Tune1_Comparison.pdf") == pdf.read_bytes()
        )
    tmp.replace(ZIP)
    print(
        "DELIVERABLE_PACKAGE_COMPLETE",
        ZIP,
        ZIP.stat().st_size,
        "FILES",
        len(files),
        flush=True,
    )
    print("REPORT_SHA256", digest(pdf), flush=True)
    print("GALLERY_SHA256", digest(gallery), flush=True)


if __name__ == "__main__":
    make()
