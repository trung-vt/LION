"""Render side-by-side reconstruction evidence with explicit provenance caveats."""
from pathlib import Path
import csv, json, math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("method_audit_evidence")
OLD = Path(
    "/store/LION/ttv22/pcm/results/20260920_002412_synthetic_png_dataset_box_plots/artifacts/CIGS"
)
CASES = (611, 68)
LIST = [
    ("Ground truth", "ground_truth"),
    ("Zero-filled quick inverse", "zero_filled"),
    ("SPGL1", "spgl1"),
    ("PnP-ADMM + DRUNet", "pnp_admm_drunet"),
    ("PnP-ADMM + DWTDNet", "pnp_admm_dwtdnet_tiny"),
    ("GS-PnP + GS-DRUNet", "pnp_gs_deepinv_drunet"),
    ("GS-PnP + GS-DWTDNet", "pnp_gs_deepinv_dwtdnet_tiny"),
    ("Supervised U-TV (epoch 38)", "supervised_utv_ep38"),
    ("Zero-shot U-TV (archived)", "zero_shot_utv"),
]


def score(t, r):
    dr = float(t.max() - t.min())
    return 10 * math.log10(dr * dr / float(((r - t) ** 2).mean()))


entries = []
for sid in CASES:
    name = f"CIGS_{sid:04d}"
    p = OUT / name
    prov = json.loads((p / "provenance.json").read_text())
    gt = np.load(p / "ground_truth.npy")
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 12), layout="constrained")
    fig.suptitle(
        f"CIGS test image {sid:04d}: reconstruction code audit",
        fontsize=17,
        weight="bold",
    )
    for ax, (title, method) in zip(axes.ravel(), LIST):
        ax.axis("off")
        if method == "zero_shot_utv":
            archive = (
                OLD
                / f"sample_{sid:04d}/trial_001/s_0.20_cJ_6/zero_shot_utv/recons/recon.npy"
            )
            if archive.is_file():
                arr = np.load(archive)
                shown = f"{title}\nDifferent noise/pattern (September archive)"
                status = "archived different measurement"
                psnr = score(gt, arr)
            else:
                arr = None
                shown = title + "\nArchive not found"
                status = "missing archive"
                psnr = float("nan")
        elif method in ("pnp_admm_dwtdnet_tiny", "pnp_gs_deepinv_dwtdnet_tiny"):
            arr = None
            shown = title + "\nNo trained weights on tuni"
            status = "weights missing"
            psnr = float("nan")
        else:
            file = p / (method + ".npy")
            arr = np.load(file) if file.exists() else None
            details = (
                json.loads((p / (method + ".json")).read_text())
                if (p / (method + ".json")).is_file()
                else {}
            )
            psnr = (
                details.get("metrics", {}).get("psnr", float("nan"))
                if method != "ground_truth"
                else float("nan")
            )
            status = (
                "matched experiment_10 CPU"
                if arr is not None and method != "ground_truth"
                else "ground truth"
            )
            shown = title + (f"\nPSNR {psnr:.2f} dB" if np.isfinite(psnr) else "")
        if arr is None:
            ax.set_facecolor("#eeeeee")
            ax.add_patch(
                plt.Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes, color="#eeeeee", zorder=1
                )
            )
            ax.text(
                0.5,
                0.5,
                "UNAVAILABLE",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=17,
                color="#777777",
                weight="bold",
                zorder=2,
            )
        else:
            ax.imshow(
                np.squeeze(arr), cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
        ax.set_title(shown, fontsize=10.8, pad=7)
        entries.append(
            {
                "category": "CIGS",
                "sample_id": sid,
                "method": method,
                "status": status,
                "psnr": psnr,
                "measurement_sha256": prov["measurement_sha256"]
                if status.startswith("matched")
                else "",
            }
        )
    fig.text(
        0.5,
        0.004,
        "Common acquisition: 20% multilevel Hadamard, coarse J=6, PCM NPL default; all methods except archived zero-shot.\n"
        "Blank panels indicate missing DWTDNet weights. Archived zero-shot used a DIFFERENT sampling/noise realisation.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    filename = OUT / f"{name}_method_evidence.jpg"
    fig.savefig(filename, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("CREATED", filename, filename.stat().st_size)
with (OUT / "method_evidence_status.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=entries[0].keys())
    w.writeheader()
    w.writerows(entries)
print("EVIDENCE_CREATED", len(entries))
