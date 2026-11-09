"""Update PCM tune-1 report input fragments from frozen preparation/results."""
import argparse
import json
from pathlib import Path

from scripts.experiments.pcm.tune1_ternary_20261109.prepare import (
    OUTPUT,
    CATEGORIES,
    SCHEMA,
    digest,
)

REPORT = OUTPUT / "report"
TEX = REPORT / "PCM_Tune1_Ternary_Search_Report.tex"


def table():
    c = json.loads((OUTPUT / "prepared/validation_calibrations.json").read_text())
    t = json.loads((OUTPUT / "prepared/test45.json").read_text())
    assert len(c["cases"]) == 25 and len(t["cases"]) == 45
    rows = []
    for cat in CATEGORIES:
        a = sorted(int(v["sample_id"]) for v in c["cases"] if v["category"] == cat)
        b = sorted(int(v["sample_id"]) for v in t["cases"] if v["category"] == cat)
        assert len(a) == 5 and len(b) == 9
        rows.append(
            cat
            + " & "
            + ", ".join(f"{n:04d}" for n in a)
            + " & "
            + ", ".join(f"{n:04d}" for n in b)
            + r" \\"
        )
    return (
        r"""\begin{center}\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lp{39mm}p{100mm}@{}}
\toprule
Category & Five validation calibrations & Nine test45 images\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{center}\normalsize
"""
    )


def completed_result_content():
    from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import (
        assert_all_frozen,
    )

    c, t, selected = assert_all_frozen()
    base = OUTPUT / "analysis"
    data = json.loads((base / "test45_summary.json").read_text())
    detailed = json.loads((base / "per_calibration_9_images.json").read_text())
    assert len(detailed) == 50
    labels = (
        ("zero_filled", "Zero-filled"),
        ("spgl1_bp", "SPGL1-BP"),
        ("spgl1_bpdn", "SPGL1-BPDN"),
        ("pnp_admm_oracle", "PnP-ADMM oracle"),
        ("gs_pnp_oracle", "GS-PnP oracle"),
        ("tv", "TV"),
        ("utv_c60", "UTV-C60"),
        ("pnp_admm_ssdu5", "ADMM SSDU-5"),
        ("gs_pnp_ssdu5", "GS SSDU-5"),
        ("gs_pnp_expanded5", "GS expanded-5"),
        ("pnp_admm_tune1", "PnP-ADMM tune-1"),
        ("gs_pnp_tune1", "GS-PnP tune-1"),
    )
    rows = []
    for method, label in labels:
        v = data["ALL"][method]

        def fmt(k, n):
            return f'{v[k]["mean"]:.{n}f} $\\pm$ {v[k]["sd_across_unique_test_images"]:.{n}f}'

        rows.append(
            label
            + " & "
            + fmt("psnr", 2)
            + " & "
            + fmt("ssim", 4)
            + " & "
            + fmt("haarpsi", 4)
            + " & "
            + f'{v["mse"]["mean"]:.3e}'
            + r" \\"
        )
    calibration_rows = []
    for cat in CATEGORIES:
        for method in ("admm", "gs"):
            group = sorted(
                (
                    v
                    for v in detailed.values()
                    if v["category"] == cat and v["method"] == method
                ),
                key=lambda v: v["calibration_sample_id"],
            )
            assert len(group) == 5
            for v in group:
                calibration_rows.append(
                    cat
                    + " & "
                    + method.upper()
                    + " & "
                    + f'{v["calibration_sample_id"]:04d}'
                    + " & "
                    + f'{v["calibration_ssdu_mse"]:.6f}'
                    + " & "
                    + f'{v["test_metrics"]["psnr"]["mean"]:.2f}'
                    + r" \\"
                )
    return (
        r"""All 50 tune-1 selections were frozen using the validation acquisitions,
before test45 reconstruction metrics were calculated.
Each selection was transferred to the same nine test images of its category.
\begin{table}[H]
\centering\tiny
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
Method & PSNR (dB) & SSIM & HaarPSI & MSE\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\caption{Test45 mean and SD across 45 unique test images. For tune-1,
we first average the five calibration choices per image,
without choosing a winner using test metrics.}
\end{table}
\begingroup\scriptsize
\begin{longtable}{@{}llrrr@{}}
\caption{All 50 independently selected tune-1 calibrations, each
evaluated on nine unique test images.}\\
\toprule
Category & Method & Calibration ID & SSDU loss & Mean test9 PSNR\\
\midrule
\endfirsthead
\toprule
Category & Method & Calibration ID & SSDU loss & Mean test9 PSNR\\
\midrule
\endhead
"""
        + "\n".join(calibration_rows)
        + r"""
\bottomrule
\end{longtable}\endgroup
\IfFileExists{../analysis/figures/test45_12_methods_mean_sd.pdf}{
\begin{figure}[H]\centering
\includegraphics[width=.92\linewidth]{../analysis/figures/test45_12_methods_mean_sd.pdf}
\caption{Twelve method labels compared on the same 45 images.}
\end{figure}}{}
The separate 45-page qualitative PDF contains one target and 20 reconstruction
panels per test image, including five independent tune-1 models for each
PnP method, with PSNR, SSIM, HaarPSI and MSE on each panel.
"""
    )


def update(stage="prepared"):
    assert stage in ("prepared", "completed")
    assert TEX.exists(), TEX
    REPORT.mkdir(exist_ok=True, parents=True)
    (REPORT / "selection_table.tex").write_text(table())
    if stage == "prepared":
        (REPORT / "status.tex").write_text(
            r"\textbf{Status: preparation complete. No tune-1 GPU search or test45 evaluation has run.}"
            + "\n"
        )
        (REPORT / "results_section.tex").write_text(
            "No new tune-1 parameter results have been generated. "
            "The 25 calibration acquisitions and 45 test cases are frozen. "
            "The report will be updated after all 50 validation-only "
            "parameter selections and the test45 reconstructions complete.\n"
        )
        (REPORT / "conclusion_dynamic.tex").write_text(
            "The experiment is prepared, not yet executed. "
            "Ternary search assumes one-dimensional unimodality along each "
            "coordinate; the joint PnP loss need not have one optimum. "
            "Test metrics will not be used to choose parameters.\n"
        )
    else:
        result = completed_result_content()
        (REPORT / "status.tex").write_text(
            r"\textbf{Status: tune-1 searches frozen and full test45 evaluation complete.}"
            + "\n"
        )
        (REPORT / "results_section.tex").write_text(result)
        (REPORT / "conclusion_dynamic.tex").write_text(
            "The tune-1 study is complete with 50 independent calibration "
            "selections and 45 matched test cases. "
            "The report separates calibration-to-calibration variability "
            "from variability between independent test images. "
            "A lower SSDU measurement loss is not automatically a better "
            "PSNR reconstruction, especially with correlated PCM noise. "
            "The results must not be used for test-driven retuning.\n"
        )
    (REPORT / "report_manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "stage": stage,
                "source_tex": str(TEX),
                "validation_selection_sha256": digest(
                    OUTPUT / "prepared/validation_calibrations.json"
                ),
                "test45_selection_sha256": digest(OUTPUT / "prepared/test45.json"),
                "ranges_sha256": digest(OUTPUT / "prepared/ranges.json"),
                "used_ground_truth_to_select_parameters": False,
                "used_test_metrics_for_parameter_selection": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("REPORT_INPUTS_READY", REPORT, "stage", stage)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prepared", "completed"), default="prepared")
    args = ap.parse_args()
    update(args.stage)
