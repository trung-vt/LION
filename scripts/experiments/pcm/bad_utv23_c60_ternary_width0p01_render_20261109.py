"""Render the 23-case C60 ternary oracle search comparisons.

Each PDF page has GT, PnP-ADMM-tuned-5, stored original C60, and the
per-image best observed PSNR C60 from ternary search to rho bracket <=1e-2.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

from LION.utils.metrics import compute_metrics
from LION.visualize.plotting_functions import make_colormap

from scripts.experiments.pcm.bad_utv23_c60_ternary_width0p01_20261109 import (
    ROOT,
    OUT,
    MANIFEST,
    METRICS,
    SCORE450,
    RHO_WIDTH,
    get_cases_and_table,
)


def metrics_against_gt(x, gt):
    p = torch.from_numpy(x)[None, None].float()
    target = torch.from_numpy(gt)[None, None].float()
    vals = compute_metrics(recon=p, target=target, data_range=1.0)
    return {k: float(vals[k]) for k in METRICS}


def verify_score(actual, recorded, name):
    for k in METRICS:
        np.testing.assert_allclose(
            actual[k],
            recorded[k],
            rtol=2e-5,
            atol=2e-4 if k == "haarpsi" else 2e-5,
            err_msg=f"reconstruction metric mismatch: {name} {k}",
        )


def main():
    cases, benchmark = get_cases_and_table()
    assert len(cases) == 23
    all_rows = []
    paired = []
    diagnostics = []
    for c in cases:
        sample = c["sample_name"]
        cat, sid = c["category"], int(c["sample_id"])
        path = OUT / "cases" / sample
        assert (path / "summary.json").is_file(), sample
        assert (path / "reconstructions.npz").is_file(), sample
        info = json.loads((path / "summary.json").read_text())
        with np.load(path / "reconstructions.npz", allow_pickle=False) as z:
            images = {
                k: z[k].copy()
                for k in ("gt", "admm", "c60_rho1", "optimal", "lambda_c60")
            }
            rho = float(z["rho_opt"])
        assert rho == float(info["rho_opt"])
        assert 1e-4 <= rho <= 10
        assert 0 < info["search_rho_width"] <= RHO_WIDTH + 1e-12
        assert images["gt"].shape == (256, 256)
        assert all(
            a.shape == (256, 256) and np.isfinite(a).all() for a in images.values()
        )
        for name, key in (
            ("admm", "admm_metrics"),
            ("c60_rho1", "rho1_metrics"),
            ("optimal", "optimal_metrics"),
        ):
            verify_score(
                metrics_against_gt(images[name], images["gt"]), info[key], sample
            )
        raw = benchmark[(cat, sid)]
        assert info["measurement_sha256"] == raw["measurement_sha256"]
        assert info["indices_sha256"] == raw["pattern_sha256"]
        assert (
            not info["selected_outside_bracket"] or info["best_probe_outside_bracket"]
        )
        row = {
            "category": cat,
            "sample_id": f"{sid:04d}",
            "sample": sample,
            "rho_opt": rho,
            "bracket_lo": info["rho_bracket"][0],
            "bracket_hi": info["rho_bracket"][1],
            "bracket_width": info["search_rho_width"],
            "ternary_iterations": info["iterations"],
            "evaluations": info["evaluations"],
            "selected_outside_bracket": info["selected_outside_bracket"],
            "measurement_sha256": info["measurement_sha256"],
            "indices_sha256": info["indices_sha256"],
            "admm_psnr": info["admm_metrics"]["psnr"],
            "c60_rho1_psnr": info["rho1_metrics"]["psnr"],
            "c60_opt_psnr": info["optimal_metrics"]["psnr"],
            "delta_opt_vs_c60_psnr": info["psnr_gain_vs_rho1"],
            "delta_opt_vs_admm_psnr": info["psnr_gap_to_admm"],
            "admm_ssim": info["admm_metrics"]["ssim"],
            "c60_rho1_ssim": info["rho1_metrics"]["ssim"],
            "c60_opt_ssim": info["optimal_metrics"]["ssim"],
            "admm_haarpsi": info["admm_metrics"]["haarpsi"],
            "c60_rho1_haarpsi": info["rho1_metrics"]["haarpsi"],
            "c60_opt_haarpsi": info["optimal_metrics"]["haarpsi"],
            "observation": info.get("case_observation", ""),
        }
        all_rows.append(row)
        paired.append((c, info, images))

    assert len(all_rows) == 23
    OUT.mkdir(parents=True, exist_ok=True)
    path_csv = OUT / "all_23_summary.csv"
    with path_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)

    # One compact archive for offline examination of the selected solutions.
    image_archive = {f"{c['sample_name']}_optimal": z["optimal"] for c, _, z in paired}
    image_archive.update(
        {
            f"{c['sample_name']}_rho": np.float64(info["rho_opt"])
            for c, info, _ in paired
        }
    )
    np.savez_compressed(OUT / "optimal_reconstructions_23.npz", **image_archive)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    path_pdf = OUT / "PCM_Bad_UTV_23_Ternary_1e-2_4_Panel_Comparisons.pdf"
    with PdfPages(
        path_pdf,
        metadata={
            "Title": "23 bad-UTV cases: frozen PCM measurement, C60 ternary, ADMM tuned-5",
            "Author": "PCM evaluation",
            "Subject": "One page per case: GT, ADMM tuned-5, original C60, ternary best C60",
        },
    ) as pdf:
        for ix, (c, info, images) in enumerate(paired, 1):
            fig = plt.figure(figsize=(17.2, 7.2))
            gs = fig.add_gridspec(
                1, 4, left=0.035, right=0.965, top=0.74, bottom=0.19, wspace=0.13
            )
            fig.text(
                0.5,
                0.973,
                f"Bad-UTV evaluation | {c['sample_name']}",
                ha="center",
                va="top",
                fontsize=21,
                weight="bold",
            )
            fig.text(
                0.5,
                0.921,
                "20% multilevel Hadamard | NPL-default noise | same frozen acquisition for all reconstructions",
                ha="center",
                va="top",
                fontsize=11.4,
            )
            labels = [
                ("GT", images["gt"], None),
                ("PnP-ADMM-tuned-5", images["admm"], info["admm_metrics"]),
                ("C60  (rho=1)", images["c60_rho1"], info["rho1_metrics"]),
                (
                    f"C60 optimum  (rho={info['rho_opt']:.5f})",
                    images["optimal"],
                    info["optimal_metrics"],
                ),
            ]
            for j, (label, image, scores) in enumerate(labels):
                ax = fig.add_subplot(gs[0, j])
                ax.imshow(
                    image,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    origin="upper",
                    rasterized=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#666666" if j != 3 else "#222222")
                    sp.set_linewidth(0.55 if j != 3 else 1.6)
                caption = (
                    "Ground truth"
                    if scores is None
                    else f"PSNR {scores['psnr']:.2f} dB | SSIM {scores['ssim']:.3f}"
                    f"\nHaarPSI {scores['haarpsi']:.3f}"
                )
                ax.set_title(
                    f"{label}\n{caption}",
                    fontsize=10.8,
                    pad=8,
                    weight="bold" if j == 3 else "normal",
                )
            fig.text(
                0.035,
                0.147,
                f"Per-image PSNR gain from scaling: {info['psnr_gain_vs_rho1']:+.3f} dB"
                f"     |     Compared with PnP-ADMM-tuned-5: {info['psnr_gap_to_admm']:+.3f} dB",
                ha="left",
                va="center",
                fontsize=11.2,
            )
            fig.text(
                0.035,
                0.106,
                f"Ternary rho range [0.0001, 10]; final bracket "
                f"[{info['rho_bracket'][0]:.5f}, {info['rho_bracket'][1]:.5f}]"
                f", width {info['search_rho_width']:.5f}; "
                f"{info['iterations']} iterations, {info['evaluations']} evaluations",
                ha="left",
                va="center",
                fontsize=9.8,
            )
            if info["selected_outside_bracket"]:
                fig.text(
                    0.965,
                    0.108,
                    "Caution: best observed rho lies outside ternary bracket",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#a52a20",
                )
            note = info.get("case_observation", "")
            if note:
                fig.text(
                    0.035,
                    0.063,
                    note,
                    ha="left",
                    va="center",
                    fontsize=9.3,
                    color="#555555",
                )
            cbax = fig.add_axes((0.35, 0.032, 0.30, 0.012))
            bar = fig.colorbar(
                ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                cax=cbax,
                orientation="horizontal",
            )
            bar.ax.tick_params(labelsize=8)
            fig.text(
                0.70,
                0.027,
                "Display [0,1]; metrics on raw images",
                ha="left",
                va="center",
                fontsize=8.5,
            )
            fig.text(0.965, 0.027, f"{ix}/23", ha="right", va="center", fontsize=9.5)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
            print(
                "PDF_PAGE",
                ix,
                23,
                c["sample_name"],
                "rho",
                f"{info['rho_opt']:.5f}",
                flush=True,
            )
    assert path_pdf.stat().st_size > 100000

    grouped = {}
    for r in all_rows:
        grouped.setdefault(r["category"], []).append(r)
    stats = {}
    for category, items in [("ALL", all_rows), *list(grouped.items())]:
        stats[category] = {
            "count": len(items),
            "mean_rho": float(np.mean([x["rho_opt"] for x in items])),
            "mean_psnr_admm": float(np.mean([x["admm_psnr"] for x in items])),
            "mean_psnr_c60_rho1": float(np.mean([x["c60_rho1_psnr"] for x in items])),
            "mean_psnr_c60_opt": float(np.mean([x["c60_opt_psnr"] for x in items])),
            "mean_psnr_gain_vs_c60_rho1": float(
                np.mean([x["delta_opt_vs_c60_psnr"] for x in items])
            ),
            "mean_psnr_gap_vs_admm": float(
                np.mean([x["delta_opt_vs_admm_psnr"] for x in items])
            ),
        }
    summary = {
        "name": "bad images for UTV, 23-case C60 ternary multiplier PSNR search",
        "status": "complete",
        "case_count": 23,
        "rho_bounds": [1e-4, 10],
        "rho_width_tolerance": 1e-2,
        "criterion": "ground truth PSNR",
        "method": "linear-rho ternary per case, extra diagnostic and bracket points",
        "oracle_warning": "GT-selected multipliers; not an unbiased generalisation estimate",
        "unimodality_warning": "No proof of PSNR unimodality, width limit not global optimum certificate",
        "all_cases_have_frozen_measurement_hashes": True,
        "all_metrics_validated_from_raw": True,
        "all_cases_in_final_bracket": all(
            not x["selected_outside_bracket"] for x in all_rows
        ),
        "cases_opt_better_than_rho1": sum(
            x["delta_opt_vs_c60_psnr"] > 1e-5 for x in all_rows
        ),
        "cases_opt_better_than_admm": sum(
            x["delta_opt_vs_admm_psnr"] > 1e-5 for x in all_rows
        ),
        "by_category": stats,
        "pdf": str(path_pdf),
        "csv": str(path_csv),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "RENDER_COMPLETE",
        str(path_pdf),
        "pdf_bytes",
        path_pdf.stat().st_size,
        "overall_stats",
        stats["ALL"],
        flush=True,
    )


if __name__ == "__main__":
    main()
