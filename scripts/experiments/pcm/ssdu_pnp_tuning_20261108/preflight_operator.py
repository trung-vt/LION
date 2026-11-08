"""One-image operator-only SSDU smoke test; no PnP tuning and no GT."""
from __future__ import annotations
import os

os.environ.setdefault("LION_DATA_PATH", "/store/LION/ttv22/pcm/data")
os.environ.setdefault("LION_TTV22_PATH", "/store/LION/ttv22")
import json
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    OUTPUT,
    immutable_json,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
    read_blind_acquisition,
    make_fold_inputs,
    heldout_mse,
)


def main():
    p = json.loads((OUTPUT / "manifest.json").read_text())
    pilot = p["pilot5"][0]
    case = next(
        q
        for q in p["validation_cases"]
        if q["category"] == pilot["category"] and q["sample_id"] == pilot["sample_id"]
    )
    y, idx = read_blind_acquisition(Path(case["source_file"]))
    with np.load(case["fold_file"], allow_pickle=False) as f:
        h = f["heldout_0"].copy()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this operator smoke test")
    device = torch.device("cuda:0")
    op_fit, fit_y, op_val, hidden_y = make_fold_inputs(y, idx, h, device)
    assert fit_y.shape[-1] + hidden_y.shape[-1] == 13107
    assert hidden_y.shape[-1] in range(1600, 3000)
    with torch.no_grad():
        zf = op_fit.pseudo_inv(fit_y)
        assert tuple(zf.shape) == (1, 1, 256, 256)
        score = heldout_mse(zf, op_val, hidden_y)
    if not np.isfinite(score) or score < 0:
        raise FloatingPointError("Invalid held-out MSE")
    print(
        "BLIND_OPERATOR_SMOKE_PASS",
        pilot,
        "fit_count",
        fit_y.shape[-1],
        "heldout_count",
        hidden_y.shape[-1],
        "heldout_zero_fill_mse",
        score,
        "device",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    immutable_json(
        OUTPUT / "operator_smoke.json",
        {
            "status": "passed",
            "case": pilot,
            "fit_count": int(fit_y.shape[-1]),
            "heldout_count": int(hidden_y.shape[-1]),
            "zero_filled_cv_mse": score,
            "uses_ground_truth": False,
            "device": torch.cuda.get_device_name(0),
            "is_hyperparameter_tuning": False,
        },
    )


if __name__ == "__main__":
    main()
