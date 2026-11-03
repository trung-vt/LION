"""Create fixed-20%-sampling, NPL-default PCM boxplots and 50-image PDF."""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("/store/LION/ttv22/pcm/results/pcm/pcm_utv_fixed20_npl_default_20261103")
CATS = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
METHODS = ("zero_filled", "spgl1_bp", "gs_pnp", "A", "B", "C", "old_utv")
METRICS = ("mse", "psnr", "ssim", "haarpsi")
VISUAL_SEED = 2026110316

NAMES = {
    "zero_filled": "Zero-filled",
    "spgl1_bp": "SPGL1 BP",
    "gs_pnp": "GS-PnP",
    "A": "U-TV A38",
    "B": "U-TV B44",
    "C": "U-TV C60",
    "old_utv": "Old U-TV",
}
VISUAL_ORDER = ("ground_truth", "zero_filled", "spgl1_bp", "gs_pnp", "C", "old_utv")
COLOURS = ("0.6", "0.75", "0.8", "0.55", "0.5", "0.3", "0.9")


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def checked_data():
    meta = json.loads((OUT / "protocol.json").read_text())
    assert meta["sampling_ratio"] == 0.20 and meta["noise_parameter_sampling"] == False
    allrows = []
    for cat in CATS:
        fragments = sorted((OUT / "results").glob(f"{cat}_*.csv"))
        assert fragments, (cat, "no completed test fragments")
        rows = [r for path in fragments for r in read_rows(path)]
        ids = {int(v) for v in meta["test_ids"]}
        observed = [int(x["sample_id"]) for x in rows]
        assert len(rows) == 90 and set(observed) == ids and len(set(observed)) == 90, (
            cat,
            len(rows),
        )
        for r in rows:
            assert r["category"] == cat and float(r["sampling_ratio"]) == 0.2
            assert int(r["num_patterns"]) == 13107
            for m in METHODS:
                for key in METRICS:
                    assert math.isfinite(float(r[f"{m}_{key}"]))
        allrows.extend(rows)
    assert len(allrows) == 450
    for cat, selected in meta["selected_50_ids"].items():
        for sid in selected:
            assert (OUT / "cases" / f"{cat}_{sid:04d}.npz").is_file()
    return meta, allrows


def data_cube(rows):
    # categories x 90 test images x 7 methods x 4 metrics
    a = np.full((5, 90, len(METHODS), len(METRICS)), np.nan, dtype=np.float64)
    for ci, cat in enumerate(CATS):
        subset = sorted(
            (x for x in rows if x["category"] == cat), key=lambda x: int(x["sample_id"])
        )
        assert len(subset) == 90
        for ii, r in enumerate(subset):
            for mi, m in enumerate(METHODS):
                for ki, k in enumerate(METRICS):
                    a[ci, ii, mi, ki] = float(r[f"{m}_{k}"])
    assert np.isfinite(a).all()
    return a


def statistics(meta, rows, cube):
    summary = {
        "protocol": meta["protocol"],
        "sampling_ratio": 0.20,
        "noise": meta["noise_parameters_fixed"],
        "images": 450,
        "noise_acquisitions_per_image": 1,
        "methods": METHODS,
        "per_method": {},
        "per_category": {},
        "selected_50_ids": meta["selected_50_ids"],
    }
    for mi, name in enumerate(METHODS):
        overall = cube[:, :, mi, :]
        summary["per_method"][name] = {
            metric: {
                "mean": float(overall[:, :, ki].mean()),
                "median": float(np.median(overall[:, :, ki])),
                "q25": float(np.quantile(overall[:, :, ki], 0.25)),
                "q75": float(np.quantile(overall[:, :, ki], 0.75)),
            }
            for ki, metric in enumerate(METRICS)
        }
    for ci, cat in enumerate(CATS):
        summary["per_category"][cat] = {}
        for mi, m in enumerate(METHODS):
            summary["per_category"][cat][m] = {
                metric: {
                    "mean": float(cube[ci, :, mi, ki].mean()),
                    "median": float(np.median(cube[ci, :, mi, ki])),
                }
                for ki, metric in enumerate(METRICS)
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "all_450_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summary


def plot_four(cube, title, output, counts):
    fig, axes = plt.subplots(2, 2, figsize=(16.54, 11.69))
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.14, top=0.87, hspace=0.34, wspace=0.23
    )
    for ki, (ax, key) in enumerate(zip(axes.flat, METRICS)):
        data = [cube[:, mi, ki] for mi in range(len(METHODS))]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 4,
            },
        )
        for patch, colour in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(colour)
        ax.set_xticks(
            np.arange(1, len(METHODS) + 1),
            [NAMES[x] for x in METHODS],
            rotation=28,
            ha="right",
            fontsize=9,
        )
        ax.set_title(
            {
                "mse": "MSE (lower is better)",
                "psnr": "PSNR, dB (higher is better)",
                "ssim": "SSIM (higher is better)",
                "haarpsi": "HaarPSI (higher is better)",
            }[key]
        )
        ax.grid(axis="y", alpha=0.23)
        if key == "mse":
            ax.set_yscale("log")
    fig.suptitle(title, fontsize=20)
    fig.text(
        0.08,
        0.045,
        f"Each box has {counts} paired test images; median, IQR, 1.5 x IQR whiskers and outliers. Black diamond: mean. MSE uses log scale.",
        fontsize=10,
    )
    fig.savefig(output, bbox_inches="tight")
    return fig


def export_category_boxplots(meta, cube):
    path = OUT / "Fixed20_NPL_Boxplots_By_Category.pdf"
    with PdfPages(
        path,
        metadata={
            "Title": "PCM fixed 20% default-noise boxplots: 450 images by category"
        },
    ) as pdf:
        fig = plot_four(
            cube.reshape(450, len(METHODS), len(METRICS)),
            "All 450 test images | fixed 20% | NPL default noise",
            OUT / "boxplots_all_450.png",
            450,
        )
        pdf.savefig(fig)
        plt.close(fig)
        for ci, cat in enumerate(CATS):
            fig = plot_four(
                cube[ci],
                f"{cat}: 90 held-out test images | fixed 20% | NPL default noise",
                OUT / f"boxplots_{cat}.png",
                90,
            )
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(16.54, 11.69))
        fig.text(
            0.08, 0.91, "Boxplot protocol and limitations", fontsize=23, weight="bold"
        )
        notes = [
            "Five categories, each with 90 held-out synthetic PCM images.",
            "Sampling ratio = 0.20 for every image, multilevel Hadamard, J=8, coarse J=6.",
            "Every acquisition uses the same NPL-default noise parameters; no sigma sampling.",
            "Independent noise traces and masks are deterministically seeded per image.",
            "All seven methods see exactly the same sampled operator and noisy measurement.",
            "Original SPGL1 is basis pursuit (db4, max 100 iterations), then 10 debias steps.",
            "GS-PnP is the pretrained GS-DRUNet preset with lambda=0.1, sigma=0.05, max 50 iterations.",
            "U-TV A epoch 38, B epoch 44, C epoch 60, old clean U-TV epoch 99.",
            "The old and new U-TV models also differ in architecture and unrolling depth.",
            "The full test contains one noise realisation per image; this cannot measure repeated-noise variability.",
            "MSE uses a log axis so severe outliers remain visible; the box diamonds show means.",
            "Synthetic images with simulated measurements are not physical measured PCM acquisitions.",
        ]
        for i, n in enumerate(notes):
            fig.text(0.09, 0.83 - i * 0.058, f"{i+1:02d}. {n}", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
    print("CATEGORY_BOXPLOTS_PDF", path, path.stat().st_size)


def export_visual_pdf(meta, rows):
    lookup = {(r["category"], int(r["sample_id"])): r for r in rows}
    selections = meta["selected_50_ids"]
    assert all(len(selections[c]) == 10 for c in CATS)
    path = OUT / "Fixed20_NPL_50_Visual_Comparisons.pdf"
    titles = [
        "Ground truth",
        "Zero-filled",
        "Original SPGL1",
        "GS-PnP",
        "New U-TV C60",
        "Old clean U-TV",
    ]
    with PdfPages(
        path,
        metadata={
            "Title": "PCM fixed 20% NPL default 50 image reconstruction comparisons"
        },
    ) as pdf:
        fig = plt.figure(figsize=(16.54, 11.69))
        fig.text(
            0.07,
            0.91,
            "Fixed 20% PCM test | 50 visual examples",
            fontsize=25,
            weight="bold",
        )
        fig.text(
            0.07,
            0.84,
            "10 randomly selected held-out images from each of 5 categories",
            fontsize=15,
        )
        fig.text(
            0.07,
            0.76,
            "Ground truth | Zero-filled | SPGL1 BP | GS-PnP | U-TV C60 | Old clean U-TV",
            fontsize=15,
        )
        notes = [
            "Fixed 20% multilevel Hadamard subsampling, coarse J=6.",
            "Fixed NPL-default sigma parameters for every image.",
            "Random mask and noise trace per image; same measurement used by all solvers.",
            "SPGL1 basis pursuit: db4, 100 iterations and 10 debias iterations.",
            "GS-PnP: GS-DRUNet pretrained preset, up to 50 iterations.",
            "U-TV C: noise-trained 3 x 56 U-Net, epoch 60, 512 unrolled iterations.",
            "Old U-TV: clean-trained 2 x 8 U-Net, epoch 99, 256 unrolled iterations.",
            "Full test: 450 images and one acquisition per image, seven reconstruction methods.",
            "PSNR / SSIM / HaarPSI computed on raw reconstruction values.",
            "All image displays use fixed grayscale intensity [0,1].",
            "The comparison tests full models, not noise training in isolation.",
        ]
        for i, n in enumerate(notes):
            fig.text(0.09, 0.67 - i * 0.047, f"{i+1:02d}. {n}", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        for cat in CATS:
            choices = selections[cat]
            for start in (0, 4, 8):
                part = choices[start : start + 4]
                fig, axs = plt.subplots(
                    len(part), 6, figsize=(16.54, 11.69), squeeze=False
                )
                fig.subplots_adjust(
                    left=0.012,
                    right=0.986,
                    top=0.905,
                    bottom=0.047,
                    wspace=0.05,
                    hspace=0.37,
                )
                fig.suptitle(
                    f"{cat}: selected test images {start+1}-{start+len(part)} of 10 | fixed 20%, NPL default",
                    fontsize=17,
                    y=0.977,
                )
                for ri, sid in enumerate(part):
                    entry = lookup[(cat, sid)]
                    file = OUT / "cases" / f"{cat}_{sid:04d}.npz"
                    with np.load(file) as a:
                        assert float(a["sampling_ratio"]) == 0.20
                        names = (
                            "ground_truth",
                            "zero_filled",
                            "spgl1_bp",
                            "gs_pnp",
                            "C",
                            "old_utv",
                        )
                        images = [np.squeeze(a[k].copy()) for k in names]
                    for ci, (name, im) in enumerate(zip(names, images)):
                        assert im.shape == (256, 256), (cat, sid, name, im.shape)
                        ax = axs[ri, ci]
                        ax.imshow(
                            im, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if ri == 0:
                            ax.set_title(titles[ci], fontsize=10, weight="bold")
                        if ci == 0:
                            ax.text(
                                0.02,
                                0.96,
                                f"#{sid:04d}",
                                transform=ax.transAxes,
                                va="top",
                                fontsize=9,
                                backgroundcolor="white",
                            )
                            scores = "PSNR inf | SSIM 1.000\nHaarPSI 1.000"
                        else:
                            psnr = float(entry[name + "_psnr"])
                            ssim = float(entry[name + "_ssim"])
                            hp = float(entry[name + "_haarpsi"])
                            scores = (
                                f"PSNR {psnr:.2f} | SSIM {ssim:.3f}\nHaarPSI {hp:.3f}"
                            )
                        ax.set_xlabel(scores, fontsize=7.4, labelpad=4)
                fig.text(
                    0.02,
                    0.012,
                    f"Category: {cat} | fixed sampling and fixed NPL default noise | all panels share display [0,1]",
                    fontsize=9,
                )
                pdf.savefig(fig)
                plt.close(fig)
    print("VISUAL_50_PDF", path, path.stat().st_size)


def main():
    meta, rows = checked_data()
    cube = data_cube(rows)
    summary = statistics(meta, rows, cube)
    export_category_boxplots(meta, cube)
    export_visual_pdf(meta, rows)
    print(
        "FINISHED",
        len(rows),
        "matched acquisitions",
        len(METHODS),
        "reconstruction methods",
    )
    for name, s in summary["per_method"].items():
        print("OVERALL", name, *[f"{key}={s[key]['mean']:.6g}" for key in METRICS])


if __name__ == "__main__":
    main()
