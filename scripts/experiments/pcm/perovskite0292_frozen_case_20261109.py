"""Materialize an exact SHA256-verified replay of Perovskite 0292 for C60 rho search."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch
from PIL import Image

from scripts.experiments.pcm.fixed_utv_test_20261103 import benchmark as oldbench

R = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = R / "perovskite0292_c60_ternary_20261109_v1"
BASELINE = R / "tune1_ternary_20261109_v1/test45/baselines/raw"
CSV = R / "pcm_utv_fixed20_npl_default_20261103/all_450_metrics.csv"


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    cfg, ids = oldbench.config_and_ids()
    assert 292 in ids
    with CSV.open() as f:
        row = next(
            x
            for x in csv.DictReader(f)
            if x["category"] == "Perovskite" and int(x["sample_id"]) == 292
        )
    path = oldbench.DATA / "Perovskite/image0292.png"
    with Image.open(path) as image:
        gt = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    assert gt.shape == (256, 256) and np.min(gt) == 0 and np.max(gt) == 1
    device = torch.device("cuda:0")
    x = torch.from_numpy(gt)[None, None].to(device)
    acq, idx = oldbench.acquisition(x, cfg, "Perovskite", 292, device)
    y = acq.measurement.detach().cpu().contiguous().numpy().copy()
    idx = np.asarray(idx, dtype=np.int64)
    assert y.shape == (1, 1, 13107) and idx.shape == (13107,)
    assert sha(y) == row["measurement_sha256"], (sha(y), row["measurement_sha256"])
    assert sha(idx) == row["pattern_sha256"], (sha(idx), row["pattern_sha256"])
    c = np.load(BASELINE / "utv_c60/Perovskite_0292.npy", allow_pickle=False)
    zero = np.load(BASELINE / "zero_filled/Perovskite_0292.npy", allow_pickle=False)
    assert c.shape == zero.shape == gt.shape
    mse_c = float(np.mean((c.astype(np.float64) - gt.astype(np.float64)) ** 2))
    mse_zero = float(np.mean((zero.astype(np.float64) - gt.astype(np.float64)) ** 2))
    np.testing.assert_allclose(mse_c, float(row["C_mse"]), rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(
        mse_zero, float(row["zero_filled_mse"]), rtol=3e-5, atol=1e-8
    )
    OUT.mkdir(exist_ok=True, parents=True)
    dest = OUT / "frozen_case.npz"
    if dest.exists():
        with np.load(dest) as old:
            for k, v in dict(
                ground_truth=gt,
                measurement=y,
                sampled_indices=idx,
                C=c,
                zero_filled=zero,
            ).items():
                np.testing.assert_array_equal(old[k], v)
    else:
        np.savez_compressed(
            dest,
            ground_truth=gt,
            measurement=y,
            sampled_indices=idx,
            C=c,
            zero_filled=zero,
            sampling_ratio=np.float64(0.2),
        )
    audit = {
        "case": "Perovskite_0292",
        "replay": "fixed default NPL 20% deterministic seed, SHA256 exact match to frozen test450",
        "measurement_sha256": sha(y),
        "pattern_sha256": sha(idx),
        "C_mse": mse_c,
        "zero_mse": mse_zero,
        "baseline_C60_path": str(BASELINE / "utv_c60/Perovskite_0292.npy"),
        "raw_gt_path": str(path),
    }
    (OUT / "frozen_case_provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("CASE_VERIFIED", audit, flush=True)


if __name__ == "__main__":
    main()
