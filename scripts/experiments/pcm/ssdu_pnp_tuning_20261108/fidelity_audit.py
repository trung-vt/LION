"""Audit normalized PCM operator and fit/holdout algebra without an image target."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import (
    OUTPUT,
    immutable_json,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
    make_fold_inputs,
    read_blind_acquisition,
)


@torch.no_grad()
def audit():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    pilot = manifest["pilot5"][0]
    entry = next(
        c
        for c in manifest["validation_cases"]
        if c["category"] == pilot["category"] and c["sample_id"] == pilot["sample_id"]
    )
    with np.load(entry["fold_file"], allow_pickle=False) as f:
        held = f["heldout_0"].copy()
    y, idx = read_blind_acquisition(Path(entry["source_file"]))
    device = torch.device("cuda:0")
    opfit, yfit, opval, yval = make_fold_inputs(y, idx, held, device)
    opfull = PhotocurrentMapOp(j_order=8, s_indices=idx, normalized=True, device=device)
    factors = [float(o.normalization_factor) for o in (opfit, opval, opfull)]
    assert np.allclose(factors, factors[0], atol=0, rtol=0), factors
    # A_f^* A_f and A_v^* A_v use identical normalized raw Hadamard row conventions.
    # The subset loss remains 0.5||A_fit x-y_fit||² with NO extra 1/m rescaling.
    n = len(idx)
    probe = torch.Generator(device=device).manual_seed(919)
    x = torch.randn((1, 1, 256, 256), device=device, generator=probe)
    fit_yy = torch.randn(yfit.shape, device=device, generator=probe)
    val_yy = torch.randn(yval.shape, device=device, generator=probe)

    def check_adj(op, x, z):
        left = float((op(x) * z).sum())
        right = float((x * op.adjoint(z)).sum())
        return abs(left - right) / max(1, abs(left), abs(right))

    a = check_adj(opfit, x, fit_yy)
    b = check_adj(opval, x, val_yy)
    # Matched combined forward projections must be a permutation of the full rows.
    yhfull = opfull(x).cpu().numpy().squeeze()
    yhfit = opfit(x).cpu().numpy().squeeze()
    yhval = opval(x).cpu().numpy().squeeze()
    training = np.setdiff1d(np.arange(n), held, assume_unique=True)
    assert len(training) + len(held) == n
    fit_delta = float(np.max(np.abs(yhfull[training] - yhfit)))
    val_delta = float(np.max(np.abs(yhfull[held] - yhval)))
    assert a < 2e-5 and b < 2e-5, (a, b)
    assert fit_delta < 1e-5 and val_delta < 1e-5, (fit_delta, val_delta)
    result = {
        "status": "passed",
        "fit_rows": len(training),
        "holdout_rows": len(held),
        "normalization_factors": factors,
        "adjoint_relative_error_fit": a,
        "adjoint_relative_error_holdout": b,
        "heldout_projection_delta_max": val_delta,
        "fit_projection_delta_max": fit_delta,
        "fidelity_scaling_policy": "raw normalized Hadamard A with summed 0.5||Ax-y||^2; no additional fold-size division",
        "fold_to_full_caveat": "fit uses fewer rows than final full reconstruction, so effective data information differs. Freeze direct candidate parameters without rescaling at test.",
        "noise_correlation_caveat": "Independence of NPL noise between fit and holdout measurements cannot be established from a single acquisition; block splitting mitigates nearby-row leakage, not global correlated noise.",
        "uses_ground_truth": False,
        "test_set_access": False,
    }
    immutable_json(OUTPUT / "fidelity_audit.json", result)
    print("AUDIT_PASSED", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    audit()
