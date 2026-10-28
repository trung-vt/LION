"""Paired, category-stratified analysis of preselected test acquisitions."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv, json
import numpy as np

OUT = Path("/store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028")
manifest = json.loads((OUT / "protocol_manifest_50.json").read_text())
C = manifest["categories"]
R = manifest["balanced_noise_conditions"]
M = ("4070", "4090")
V = ("old", "new")
K = ("mse", "psnr", "ssim", "haarpsi")
assert len(C) == 5 and len(R) == 2
old = {
    (r["category"], int(r["sample_id"]), r["condition"]): r
    for r in csv.DictReader((OUT / "old_reference_metrics_subset.csv").open())
}
assert len(old) == 100
x = np.empty((5, 10, 2, 2, 2, 4), dtype=np.float64)
all_rows = []
for ci, cat in enumerate(C):
    ids = manifest["subset_per_category"][cat]
    assert len(ids) == 10 and len(set(ids)) == 10
    file = OUT / f"current_{cat}.csv"
    curr = {
        (int(r["sample_id"]), r["condition"]): r for r in csv.DictReader(file.open())
    }
    assert len(curr) == 20
    for ji, sid in enumerate(ids):
        for ri, condition in enumerate(R):
            a = old[(cat, sid, condition)]
            b = curr[(sid, condition)]
            for key in (
                "category",
                "sample_id",
                "condition",
                "sampling_ratio",
                "num_patterns",
                "pattern_sha256",
                "measurement_sha256",
            ):
                assert str(a[key]) == str(b[key]), (
                    cat,
                    sid,
                    condition,
                    key,
                    a[key],
                    b[key],
                )
            combined = {
                k: a[k]
                for k in (
                    "category",
                    "sample_id",
                    "condition",
                    "sampling_ratio",
                    "num_patterns",
                    "pattern_sha256",
                    "measurement_sha256",
                    "dark_sigma_raw",
                    "bias_sigma",
                    "bias_gradient_sigma",
                    "beam_sigma",
                )
            }
            for mi, model in enumerate(M):
                for ki, metric in enumerate(K):
                    values = [
                        float(a[f"{model}_{metric}"]),
                        float(b[f"{model}_{metric}"]),
                    ]
                    assert np.isfinite(values).all()
                    x[ci, ji, ri, mi, :, ki] = values
                    for vi, version in enumerate(V):
                        combined[f"{version}_{model}_{metric}"] = values[vi]
            all_rows.append(combined)
assert len(all_rows) == 100
with (OUT / "all_paired_results_old_new.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(all_rows)

# Sample 10 complete image-level pairs with replacement within each category.
# Both noise regimes and all 4 checkpoints stay paired for each selected image.
N = 20000
rng = np.random.default_rng(20261028)
bsidx = rng.integers(0, 10, size=(N, 5, 10))


def effects(d):
    # d has shape (category=5, images=10, regimes=2)
    boot = d[np.arange(5)[None, :, None], bsidx].mean(axis=(1, 2))
    ans = {}
    for cond, idx in [(R[0], 0), (R[1], 1), ("balanced_50_50", None)]:
        vals = d[:, :, idx] if idx is not None else d.mean(axis=2)
        draws = boot[:, idx] if idx is not None else boot.mean(axis=1)
        ans[cond] = {
            "mean": float(vals.mean()),
            "ci95": [float(z) for z in np.percentile(draws, [2.5, 97.5])],
            "sd_across_images": float(vals.std(ddof=1)),
        }
    return ans


summary = {
    "experiment": manifest["name"],
    "n_images": 50,
    "n_acquisitions": 100,
    "n_new_reconstructions": 200,
    "comparisons": {},
    "means": {},
    "by_category": {},
    "wins": {},
    "sampling": {},
    "interval_method": "20000 category-stratified bootstrap resamples: 10 test images with replacement independently in each of the 5 categories; preserve both noise acquisitions and all checkpoints per image.",
    "primary_metric": "mse",
    "higher_is_better": {"mse": False, "psnr": True, "ssim": True, "haarpsi": True},
    "test_reuse_warning": manifest["test_reuse_caveat"],
}
contrasts = {
    "old_4090_minus_old_4070": x[:, :, :, 1, 0, :] - x[:, :, :, 0, 0, :],
    "new_4090_minus_new_4070": x[:, :, :, 1, 1, :] - x[:, :, :, 0, 1, :],
    "4070_new_minus_old": x[:, :, :, 0, 1, :] - x[:, :, :, 0, 0, :],
    "4090_new_minus_old": x[:, :, :, 1, 1, :] - x[:, :, :, 1, 0, :],
    "head_to_head_delta_change": (x[:, :, :, 1, 1, :] - x[:, :, :, 0, 1, :])
    - (x[:, :, :, 1, 0, :] - x[:, :, :, 0, 0, :]),
}
for mi, model in enumerate(M):
    for vi, version in enumerate(V):
        label = f"{model}_{version}"
        summary["means"][label] = {}
        for ki, k in enumerate(K):
            summary["means"][label][k] = effects(x[:, :, :, mi, vi, ki])
for label, d in contrasts.items():
    summary["comparisons"][label] = {
        k: effects(d[:, :, :, ki]) for ki, k in enumerate(K)
    }

for ci, cat in enumerate(C):
    summary["by_category"][cat] = {}
    for ri, reg in enumerate(R):
        each = {}
        for mi, model in enumerate(M):
            for vi, ver in enumerate(V):
                each[f"{model}_{ver}"] = {
                    k: float(x[ci, :, ri, mi, vi, ki].mean()) for ki, k in enumerate(K)
                }
        each["delta_4090_minus_4070_new"] = {
            k: float(contrasts["new_4090_minus_new_4070"][ci, :, ri, ki].mean())
            for ki, k in enumerate(K)
        }
        each["gain_4070_new_minus_old"] = {
            k: float(contrasts["4070_new_minus_old"][ci, :, ri, ki].mean())
            for ki, k in enumerate(K)
        }
        each["gain_4090_new_minus_old"] = {
            k: float(contrasts["4090_new_minus_old"][ci, :, ri, ki].mean())
            for ki, k in enumerate(K)
        }
        summary["by_category"][cat][reg] = each
for ri, reg in enumerate(R):
    summary["wins"][reg] = {}
    for ki, k in enumerate(K):
        sign = -1 if k == "mse" else 1
        curr = sign * contrasts["new_4090_minus_new_4070"][:, :, ri, ki].flatten()
        summary["wins"][reg][k] = {
            "4090": int((curr > 0).sum()),
            "4070": int((curr < 0).sum()),
            "tie": int((curr == 0).sum()),
        }
        for model, label in [
            ("4070", "4070_new_minus_old"),
            ("4090", "4090_new_minus_old"),
        ]:
            diff = sign * contrasts[label][:, :, ri, ki].flatten()
            summary["wins"][reg][k][f"{model}_new_better"] = int((diff > 0).sum())
# Ratio-bin stratification (all measurements share each ground truth ratio).
ratio = np.array(
    [
        [
            float(old[(cat, sid, R[0])]["sampling_ratio"])
            for sid in manifest["subset_per_category"][cat]
        ]
        for cat in C
    ]
)
summary["sampling"] = {
    "min_ratio": float(ratio.min()),
    "max_ratio": float(ratio.max()),
    "mean_ratio": float(ratio.mean()),
    "bins": {},
}
for low, high in ((0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50)):
    sel = (ratio >= low) & (ratio < high if high < 0.5 else ratio <= high)
    bname = f"{low:.2f}-{high:.2f}"
    summary["sampling"]["bins"][bname] = {
        "images": int(sel.sum()),
        "new_4090_minus_4070_mse": float(
            contrasts["new_4090_minus_new_4070"][:, :, :, 0][sel].mean()
        )
        if sel.any()
        else None,
        "4070_gain_mse": float(contrasts["4070_new_minus_old"][:, :, :, 0][sel].mean())
        if sel.any()
        else None,
        "4090_gain_mse": float(contrasts["4090_new_minus_old"][:, :, :, 0][sel].mean())
        if sel.any()
        else None,
    }
(OUT / "summary_50.json").write_text(json.dumps(summary, indent=2) + "\n")
print("FULL_SUMMARY", OUT / "summary_50.json")
for m in M:
    for v in V:
        print(
            "MODEL",
            m,
            v,
            "BALANCED",
            {
                k: round(summary["means"][f"{m}_{v}"][k]["balanced_50_50"]["mean"], 8)
                for k in K
            },
        )
for con in contrasts:
    print(
        "EFFECT",
        con,
        "BALANCED",
        {k: summary["comparisons"][con][k]["balanced_50_50"] for k in K},
    )
print("RECORDS", len(all_rows), "SAMPLE_IDS", manifest["subset_per_category"])
