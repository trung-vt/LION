from pathlib import Path
import numpy as np, csv, hashlib, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from LION.visualize.plotting_functions import make_colormap

R = Path("/store/LION/ttv22/pcm/results/pcm")
S = R / "perovskite0031_utv_c60_rho_sweep_20261109_v2"
O = R / "perovskite0031_c60_sweep_with_4_baselines_20261109_v1"
A = (
    R
    / "ten_methods_matched_450_20261105_v1/visual_14_paired_exact_v1/raw/Perovskite_0031.npz"
)
B = R / "pcm_utv_fixed20_npl_default_20261103/cases/Perovskite_0031.npz"
G = R / "ssdu_pnp_tuning_20261108_v1/test450/v1/raw/gs_ssdu/Perovskite_0031.npy"
M = R / "ssdu_pnp_tuning_20261108_v1/test450/v1/merged/all_450_11_method_metrics.csv"
rho = (0.01, 0.05, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 0.9, 0.95, 1.0)
with np.load(S / "reconstructions.npz") as z:
    s = {k: z[k] for k in z.files}
with np.load(A) as z:
    a = {k: z[k] for k in z.files}
with np.load(B) as z:
    b = {k: z[k] for k in ("ground_truth", "measurement", "sampled_indices", "C")}
with M.open() as f:
    row = next(
        r
        for r in csv.DictReader(f)
        if r["category"] == "Perovskite" and int(r["sample_id"]) == 31
    )
digest = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
assert digest(b["measurement"]) == row["measurement_sha256"]
assert digest(b["sampled_indices"]) == row["pattern_sha256"]
for x, y in (
    ("ground_truth", "ground_truth"),
    ("measurement", "measurement"),
    ("indices", "sampled_indices"),
):
    np.testing.assert_array_equal(s[x], b[y])
np.testing.assert_array_equal(s["rho_1"], b["C"])
np.testing.assert_array_equal(a["ground_truth"], b["ground_truth"])
with (S / "metrics.csv").open() as f:
    sr = {float(r["rho"]): r for r in csv.DictReader(f)}
assert set(sr) == set(rho)
entries = [("Ground truth", s["ground_truth"], None)]
rows = []
for name, im, key in (
    ("Zero-fill", a["zero_filled"], "zero_filled"),
    ("SPGL1-BPDN", a["bpdn"], "spgl1_bpdn"),
    ("PnP-ADMM-tuned-5", a["pnp_tuned"], "pnp_admm_ssdu5"),
    ("GS-PnP-tuned-5", np.load(G), "gs_pnp_ssdu5"),
):
    stats = {m: float(row[key + "_" + m]) for m in ("psnr", "ssim", "haarpsi", "mse")}
    mse = float(
        np.mean((im.astype("float64") - s["ground_truth"].astype("float64")) ** 2)
    )
    np.testing.assert_allclose(mse, stats["mse"], rtol=3e-5, atol=1e-8)
    entries.append((name, im, stats))
    rows.append({"method": name, "rho": "", **stats})
for v in rho:
    stats = {m: float(sr[v][m]) for m in ("psnr", "ssim", "haarpsi", "mse")}
    im = s[f"rho_{v:g}"]
    np.testing.assert_allclose(
        np.mean((im.astype("float64") - s["ground_truth"].astype("float64")) ** 2),
        stats["mse"],
        rtol=1e-5,
        atol=1e-8,
    )
    entries.append((f"UTV C60 rho={v:g}", im, stats))
    rows.append({"method": "UTV C60", "rho": v, **stats})
assert len(entries) == 17
fig, ax = plt.subplots(3, 6, figsize=(22, 12))
fig.subplots_adjust(
    left=0.03, right=0.97, top=0.82, bottom=0.08, wspace=0.12, hspace=0.38
)
cm = make_colormap("afmhot", 0.8)
for x in ax.flat:
    x.axis("off")
for i, (name, im, z) in enumerate(entries):
    x = ax.flat[i]
    x.imshow(im, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    title = (
        name
        if z is None
        else f"{name}\nPSNR {z['psnr']:.2f} dB | SSIM {z['ssim']:.3f}\nHaarPSI {z['haarpsi']:.3f}"
    )
    x.set_title(
        title,
        fontsize=10.5,
        pad=6,
        weight="bold" if i == 0 or name.endswith("=0.45") else "normal",
    )
ax.flat[-1].text(
    0.03,
    0.8,
    "Best UTV PSNR: rho=0.45\nBest overall PSNR:\nPnP-ADMM-tuned-5",
    transform=ax.flat[-1].transAxes,
    fontsize=13,
    va="top",
)
fig.suptitle(
    "Perovskite 0031 | UTV C60 scaling and 4 matched baselines", fontsize=21, y=0.98
)
fig.text(
    0.5,
    0.916,
    "Same stored 20% multilevel Hadamard acquisition and NPL-default noise | UTV C60, 512 PDHG iterations",
    ha="center",
    fontsize=12,
)
fig.text(
    0.5,
    0.045,
    "All panels display [0,1]. PSNR, SSIM and HaarPSI use raw unclipped reconstructions.",
    ha="center",
    fontsize=10,
)
fig.text(
    0.5,
    0.02,
    "PnP parameters: fixed tuned-5 selections; UTV scaling is a single-image post-hoc experiment.",
    ha="center",
    fontsize=10,
)
O.mkdir(exist_ok=True)
p = O / "Perovskite_0031_C60_Rho_Sweep_With_4_Baselines.pdf"
fig.savefig(p, dpi=170)
plt.close(fig)
with (O / "metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["method", "rho", "psnr", "ssim", "haarpsi", "mse"],
        lineterminator="\n",
    )
    w.writeheader()
    w.writerows(rows)
(O / "provenance.json").write_text(
    json.dumps(
        {
            "measurement_sha256": digest(b["measurement"]),
            "pattern_sha256": digest(b["sampled_indices"]),
            "reconstructions": str(S),
            "baselines": str(A),
            "gs_tuned5": str(G),
        },
        indent=2,
    )
)
print("PDF", p, p.stat().st_size, "PANELS", len(entries), "METRICS", len(rows))
