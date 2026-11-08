"""Numerical equivalence smoke checks for blind PCM PnP adapters."""
from __future__ import annotations

import json
import numpy as np
import torch

from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.prepare import OUTPUT
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.ssdu_risk import (
    read_blind_acquisition,
    make_fold_inputs,
)
from scripts.experiments.pcm.ssdu_pnp_tuning_20261108.solver_adapter import (
    make_model,
    reconstruct_gs,
    reconstruct_admm,
)
from scripts.experiments.pcm.gs_pnp_gs_drunet_tuned_20261104.step6_frozen_reconstructor import (
    reconstruct_frozen,
    load_frozen_selection,
)
from pathlib import Path


@torch.no_grad()
def check():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = json.loads((OUTPUT / "manifest.json").read_text())
    pilot = data["pilot5"][0]
    case = next(
        r
        for r in data["validation_cases"]
        if (r["category"], r["sample_id"]) == (pilot["category"], pilot["sample_id"])
    )
    y, idx = read_blind_acquisition(Path(case["source_file"]))
    with np.load(case["fold_file"], allow_pickle=False) as f:
        held = f["heldout_0"].copy()
    device = torch.device("cuda:0")
    op, yy, _, _ = make_fold_inputs(y, idx, held, device)
    selection = load_frozen_selection()
    cfg = selection["config"]
    ad = {
        "candidate_id": "gs_equivalence",
        "lambda": cfg["lambda"],
        "sigma": cfg["sigma"],
        "tau0": cfg["tau0"],
        "max_accepted_iterations": cfg["max_accepted_iters"],
        "max_backtracking_attempts": cfg["max_backtracking_attempts"],
        "gamma_backtracking": cfg["gamma_backtracking"],
        "eta_backtracking": cfg["eta_backtracking"],
        "output": "paper_final_z",
        "stopping_policy": "deepinv_relative_objective",
        "stopping_eps": selection["stopping"]["eps"],
    }
    model = make_model("gs", device)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    xa, diaga = reconstruct_gs(op, yy, model, ad)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    xb, diagb = reconstruct_frozen(op, yy, model, selection)
    diff = float((xa - xb).abs().max())
    assert diff < 3e-5, (diff, diaga, diagb)
    assert diaga["accepted_iterations"] == diagb["accepted_iterations"], (diaga, diagb)
    assert diaga["output"] == diagb["output"] == "paper_final_z"
    print(
        "GS_ADAPTER_EQUIVALENT",
        diff,
        "accepted",
        diaga["accepted_iterations"],
        flush=True,
    )
    del model, xa, xb
    torch.cuda.empty_cache()
    grid = json.loads((OUTPUT / "candidate_grid.json").read_text())
    model = make_model("admm", device)
    sample = grid["admm"][0]
    x, diag = reconstruct_admm(op, yy, model, sample)
    assert x.shape == (1, 1, 256, 256) and torch.isfinite(x).all()
    assert 20 <= diag["stop_iteration"] <= 100
    print(
        "ADMM_ADAPTER_VALID", sample["candidate_id"], diag["stop_iteration"], flush=True
    )
    result = {
        "status": "passed",
        "gs_vs_oracle_frozen_max_abs_diff": diff,
        "gs_accepted": diaga["accepted_iterations"],
        "admm_accepted": diag["stop_iteration"],
        "no_ground_truth": True,
    }
    (OUTPUT / "adapter_equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    check()
