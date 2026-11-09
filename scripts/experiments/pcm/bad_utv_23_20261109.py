"""Rebuild 23 selected bad-UTV visual comparisons from the frozen PCM benchmark.

All 23 images have matched stored 20% sampling and NPL-default noise.
The original 14 and nine additional cases retain their existing verified
reconstructions; cached ADMM tuned-5 arrays avoid new solver runs.
This is a visual comparison, not a UTV parameter-multiplier sweep.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.data_loaders.pcm.data_primitives import SamplesParameter
from LION.experiments.pcm_presets.presets import QUICK_METHODS
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.visualize.plotting_functions import make_colormap
from scripts.experiments.pcm.pnp_admm_tuning.step3_final_heldout import (
    oldbench,
    live_admm,
    RHO,
    SIGMA,
)
from scripts.experiments.pcm.bad_utv_14_20261109 import (
    ARCHIVE,
    GS_RAW,
    METHODS,
    load_metrics,
    read_case,
)

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "bad_images_for_utv_20261109_v3_23"
BASELINE_45 = ROOT / "tune1_ternary_20261109_v1/test45/baselines/raw"
RAW_CACHE = OUT / "raw_new9"
CASES = OrderedDict(
    [
        ("Reference", (21, 66, 188, 199, 260, 357, 727, 793)),
        ("CIGS", (254, 710, 727, 805, 821, 873)),
        ("Perovskite", (21, 31, 79, 98, 187, 260, 288, 292)),
        ("Silicone", (767,)),
    ]
)
ORIGINAL_14 = (
    {("Reference", i) for i in (21, 66, 188, 727)}
    | {("CIGS", i) for i in (710, 727, 805, 821, 873)}
    | {("Perovskite", i) for i in (31, 79, 98, 260, 288)}
)
NOTES = {
    (
        "Reference",
        260,
    ): "Visual observation: all compared methods failed to recover the lines.",
    ("Reference", 793): "Visual observation: all compared methods failed.",
    (
        "Silicone",
        767,
    ): "Visual observation: vertical lines remain in zero-fill but are lost by the other methods.",
}
KEYS = [method[0] for method in METHODS]
METRICS = ("psnr", "ssim", "haarpsi", "mse")


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def get_gt(cfg, cat, sid, dev):
    ds = PCMSyntheticPNGTorchDataset(
        replace(cfg.data, categories=[cat], samples=SamplesParameter(ids=[sid]))
    )
    target = ds[0].contiguous()
    assert target.shape == (256, 256) and torch.isfinite(target).all()
    return target[None, None].to(dev)


def check_mse(images, metrics, category, sid):
    gt = images["gt"].astype(np.float64)
    for key, stats in metrics.items():
        im = images[key].astype(np.float64)
        estimated = float(np.mean((im - gt) ** 2))
        expected = stats["mse"]
        np.testing.assert_allclose(
            estimated,
            expected,
            rtol=5e-4,
            atol=2e-8,
            err_msg=f"mismatched frozen output: {category}_{sid:04d} {key}",
        )


def get_new_case(cat, sid, benchmark, cfg, dev, den):
    row = benchmark[(cat, sid)]
    gt = get_gt(cfg, cat, sid, dev)
    images = {"gt": gt.cpu().numpy().squeeze().copy()}
    for key, source in (
        ("zero_filled", "zero_filled"),
        ("spgl1_bpdn", "spgl1_bpdn"),
        ("tv", "tv"),
        ("utv_c60", "utv_c60"),
    ):
        src = BASELINE_45 / source / f"{cat}_{sid:04d}.npy"
        assert src.is_file(), src
        images[key] = np.load(src, allow_pickle=False)

    gs_path = GS_RAW / f"{cat}_{sid:04d}.npy"
    assert gs_path.is_file()
    images["gs_pnp_ssdu5"] = np.load(gs_path, allow_pickle=False)

    # The stored noise draws are replayed using the exact immutable benchmark
    # seed and image. Both measurement and sampled-pattern hashes must match.
    acq, indices = oldbench.acquisition(gt, cfg, cat, sid, dev)
    y = acq.measurement
    yhash = sha(y.detach().cpu().contiguous().numpy())
    phash = sha(np.asarray(indices, dtype=np.int64))
    assert yhash == row["measurement_sha256"], (cat, sid, "acquisition SHA mismatch")
    assert phash == row["pattern_sha256"], (cat, sid, "sampling SHA mismatch")

    cache = RAW_CACHE / f"{cat}_{sid:04d}.npy"
    if cache.is_file():
        images["pnp_admm_ssdu5"] = np.load(cache, allow_pickle=False)
    else:
        with torch.no_grad():
            rec = live_admm(acq.op, y, den)["x"]
        assert rec.shape == (1, 1, 256, 256)
        image = rec.detach().cpu().numpy().squeeze().copy()
        assert np.isfinite(image).all()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, image)
        images["pnp_admm_ssdu5"] = image

    metrics = {
        key: {m: float(row[f"{metric_key}_{m}"]) for m in METRICS}
        for key, _, _, metric_key in METHODS
        if metric_key is not None
    }
    check_mse(images, metrics, cat, sid)
    for k, a in images.items():
        assert a.shape == (256, 256) and np.isfinite(a).all(), (cat, sid, k)
    return (
        images,
        metrics,
        {
            "category": cat,
            "sample_id": sid,
            "sample_name": f"{cat}_{sid:04d}",
            "measurement_sha256": yhash,
            "pattern_sha256": phash,
            "acquisition_status": "exact deterministic replay matches frozen SHA256",
            "source": "test45 raw baseline plus 450-case GS SSDU-5; replay ADMM tuned-5",
            "gs_tuned5_path": str(gs_path),
            "admm_tuned5_cache": str(cache),
        },
    )


def render(pages):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    cmap = make_colormap("afmhot", 0.8)
    cmap.set_bad("white")
    path = OUT / "PCM_Bad_UTV_23_20pct_NPL_7_Methods.pdf"
    with PdfPages(
        path,
        metadata={
            "Title": "23 selected PCM images: matched 20 percent NPL-default, 7 methods",
            "Author": "PCM reconstruction study",
            "Subject": "23 verified paired cases, all with matched stored acquisitions",
        },
    ) as pdf:
        for page_num, (images, scores, meta) in enumerate(pages, 1):
            cat, sid = meta["category"], meta["sample_id"]
            fig = plt.figure(figsize=(16.55, 10.2))
            g = fig.add_gridspec(
                2,
                4,
                left=0.045,
                right=0.955,
                top=0.845,
                bottom=0.125,
                wspace=0.20,
                hspace=0.29,
            )
            fig.text(
                0.5,
                0.976,
                f"Bad images for UTV | {cat}_{sid:04d}",
                ha="center",
                va="top",
                fontsize=20,
                weight="bold",
            )
            caption = "Stored 20% multilevel Hadamard sampling | NPL-default noise | identical acquisition"
            fig.text(
                0.5,
                0.932,
                caption,
                ha="center",
                va="top",
                fontsize=11.7,
                color="#333333",
            )
            for j, (key, label, _, metric_key) in enumerate(METHODS):
                ax = fig.add_subplot(g[j // 4, j % 4])
                if key in images:
                    ax.imshow(
                        images[key],
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                        origin="upper",
                        rasterized=True,
                    )
                    if metric_key is None:
                        subtitle = "Ground truth"
                    else:
                        s = scores[key]
                        subtitle = (
                            f"PSNR {s['psnr']:.2f} dB  |  SSIM {s['ssim']:.3f}"
                            f"\nHaarPSI {s['haarpsi']:.3f}"
                        )
                else:
                    ax.set_facecolor("#f4f4f4")
                    ax.text(
                        0.5,
                        0.5,
                        "Not available\n(no stored acquisition)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=13,
                        color="#a12825",
                    )
                    subtitle = "No reconstruction"
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#555555")
                    spine.set_linewidth(1.5 if key == "utv_c60" else 0.6)
                ax.set_title(label + "\n" + subtitle, fontsize=9.8, pad=8)

            annotation = NOTES.get(
                (cat, sid),
                "Review loss of thin structures, contrast and local details.",
            )
            noteax = fig.add_subplot(g[1, 3])
            noteax.set_axis_off()
            noteax.text(
                0.05,
                0.82,
                "Comparison notes",
                transform=noteax.transAxes,
                fontsize=13,
                weight="bold",
                va="top",
            )
            desc = (
                "All methods share the exact frozen\n"
                "measurement and sampling indices.\n\n"
                "Scalar TV: lambda=0.011, 1024 iterations.\n"
                "UTV: original C60 checkpoint.\n"
                "PnP variants: frozen SSDU-5 choices.\n\n" + annotation
            )
            noteax.text(
                0.05,
                0.69,
                desc,
                transform=noteax.transAxes,
                fontsize=10.2,
                va="top",
                linespacing=1.36,
                wrap=True,
                color="#333333",
            )
            cax = fig.add_axes((0.27, 0.064, 0.46, 0.014))
            cbar = fig.colorbar(
                ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label(
                "Common display scale [0, 1]; metrics use unmodified images",
                fontsize=10,
            )
            cbar.ax.tick_params(labelsize=9)
            sha_note = (
                f"Measurement SHA256: {meta['measurement_sha256'][:20]}..."
                f"    Pattern SHA256: {meta['pattern_sha256'][:20]}..."
            )
            fig.text(0.028, 0.012, sha_note, fontsize=8, color="#555555", va="bottom")
            fig.text(
                0.970,
                0.012,
                f"{page_num} / {len(pages)}",
                ha="right",
                fontsize=9.5,
                va="bottom",
            )
            pdf.savefig(fig, dpi=170)
            plt.close(fig)
            print(
                "PDF_PAGE",
                page_num,
                len(pages),
                meta["sample_name"],
                "VERIFIED",
                flush=True,
            )
    return path


def main():
    assert os.getenv("LION_DATA_PATH"), "LION_DATA_PATH required"
    assert RHO == 0.5 and SIGMA == 0.075, "Unexpected ADMM tuned-5 parameters"
    assert sum(map(len, CASES.values())) == 23
    assert len(ORIGINAL_14) == 14
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_metrics()
    cfg, ids = oldbench.config_and_ids()
    assert len(ids) == 90
    assert all(sid in ids for cat, vals in CASES.items() for sid in vals)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(3)
    den = None
    pages = []
    metric_rows = []
    for cat, ids_for_category in CASES.items():
        for sid in ids_for_category:
            if (cat, sid) in ORIGINAL_14:
                images, scores, meta = read_case(cat, sid, table)
                meta[
                    "acquisition_status"
                ] = "stored acquisition archive, SHA256 verified"
                meta["source"] = "previous 14-case raw archive"
            else:
                if den is None and not (RAW_CACHE / f"{cat}_{sid:04d}.npy").is_file():
                    assert (
                        dev.type == "cuda"
                    ), "GPU required only if ADMM cache is missing"
                    den = build_grayscale_denoiser(
                        deepcopy(QUICK_METHODS["pnp_admm_drunet"].denoiser), dev
                    )
                    den.eval().requires_grad_(False)
                images, scores, meta = get_new_case(cat, sid, table, cfg, dev, den)
            meta["user_observation"] = NOTES.get(
                (cat, sid), "UTV C60 loses fine details or contrast"
            )
            pages.append((images, scores, meta))
            for key, label, _, metric_key in METHODS:
                if metric_key is not None and key in images:
                    metric_rows.append(
                        {
                            "category": cat,
                            "sample_id": f"{sid:04d}",
                            "method": label,
                            **scores[key],
                            "measurement_sha256": meta["measurement_sha256"],
                            "pattern_sha256": meta["pattern_sha256"],
                        }
                    )
            print("CASE_OK", f"{cat}_{sid:04d}", meta["acquisition_status"], flush=True)

    assert len(pages) == 23 and len(metric_rows) == 23 * 6
    pdf_path = render(pages)
    csv_path = OUT / "PCM_Bad_UTV_23_20pct_NPL_7_Methods_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "category",
                "sample_id",
                "method",
                *METRICS,
                "measurement_sha256",
                "pattern_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    manifest = {
        "name": "bad images for UTV",
        "version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_case_count": 23,
        "complete_matched_case_count": 23,
        "ground_truth_only_case_count": 0,
        "sampling_ratio": 0.20,
        "noise": "NPL default fixed PCM",
        "methods": [item[1] for item in METHODS],
        "acquisitions": (
            "14 original stored acquisition files plus 9 exact SHA256-"
            "matched deterministic replays; all 23 cases have frozen acquisitions"
        ),
        "display": "AFM hot colormap, shared vmin=0 vmax=1, metrics from raw arrays",
        "method_notes": {
            "spgl1_bpdn": "fixed pilot noise_rms=0.06, no debias",
            "pnp_admm_tuned5": "frozen SSDU-5: rho=0.5, sigma=0.075",
            "gs_pnp_tuned5": "frozen SSDU-5 arrays, not oracle/expanded",
            "tv": "lambda=0.011, 1024 iterations",
            "utv": "original UTV C60, no multiplier sweep",
        },
        "cases": [meta for _, _, meta in pages],
    }
    manifest_path = OUT / "bad_images_for_utv_dataset_manifest_23.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print("PDF", pdf_path, pdf_path.stat().st_size, flush=True)
    print("CSV", csv_path, csv_path.stat().st_size, flush=True)
    print("MANIFEST", manifest_path, flush=True)


if __name__ == "__main__":
    main()
