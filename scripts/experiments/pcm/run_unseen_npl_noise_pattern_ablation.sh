#!/usr/bin/env bash
# HISTORICAL: depends on earlier presets with simulated noise enabled.
# Current raw-text-map presets have noise=None, so noisy-condition assertions
# intentionally fail. See README_NPL_RAWTXT_SCALING_CORRECTION.md.
# Paired 2x2 noise/pattern-order comparison on the two complete NPL active maps.
# Reuses the normal single-sample runner and the existing QUICK_METHODS verbatim.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
export LION_DATA_PATH="${LION_DATA_PATH:-$PWD/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PANEL_ROOT="${PANEL_ROOT:-$LION_TTV22_PATH/pcm/results/20261021_unseen_noise_pattern_quick_ablation}"

python - "$@" <<'PY'
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from LION.experiments.pcm_presets.presets import QUICK_METHODS
from LION.pcm.config_cli import parse_pcm_experiment_config_cli
from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment

# These names stay fixed between conditions, so the experiment's mask seed
# also stays fixed instead of depending on its timestamped output directory.
presets = (
    "unseen_npl_cigs_active_256",
    "unseen_npl_perovskite_active_512",
)
conditions = (
    ("noisy_randomized", (), "uniform", False),
    ("noiseless_randomized", ("noise:none",), "uniform", True),
    ("noisy_in_order", ("--sampling.randomising-scheme", "None"), None, False),
    (
        "noiseless_in_order",
        ("--sampling.randomising-scheme", "None", "noise:none"),
        None,
        True,
    ),
)

selected = tuple(sys.argv[1:]) or presets
if any(p not in presets for p in selected):
    raise SystemExit(f"Unknown preset in {selected}; valid presets: {presets}")

root = Path(os.environ["PANEL_ROOT"])
noise_seed = 20261021
for preset in selected:
    sample_name = f"{preset}_fixed_comparison"
    for label, options, scheme, no_noise in conditions:
        cfg = deepcopy(
            parse_pcm_experiment_config_cli(
                ["--preset", preset, "--log.save-plots", *options]
            )
        )
        assert (cfg.noise is None) == no_noise
        assert cfg.sampling.randomising_scheme == scheme
        assert cfg.methods == QUICK_METHODS
        assert cfg.sampling.sampling_ratios == [0.2]
        assert cfg.runtime.num_trials == 1

        cfg.runtime.device = "cuda"
        cfg.runtime.root_output_dir = root / preset / label
        output = cfg.runtime.root_output_dir / sample_name
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing results: {output}"
            )
        print(f"=== {preset} | {label} ===", flush=True)
        print(f"Output: {output}", flush=True)
        finished = run_single_sample_experiment(
            cfg,
            sample_name=sample_name,
            noise_seed=noise_seed,
            show_progress_bars=False,
        )
        assert finished == output
        (output / "ablation_settings.json").write_text(
            json.dumps(
                {
                    "preset": preset,
                    "condition": label,
                    "noise": "none" if no_noise else "npl_default",
                    "sampling_scheme": scheme,
                    "sampling_ratios": cfg.sampling.sampling_ratios,
                    "coarse_j_values": cfg.sampling.coarse_j_values,
                    "base_seed": cfg.base_seed,
                    "noise_seed": noise_seed,
                    "fixed_sample_name": sample_name,
                    "methods": list(cfg.methods),
                    "note": (
                        "The same sampled indices are used in noisy and "
                        "noiseless cases for each sampling scheme."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"COMPLETED {preset} | {label}: {finished}", flush=True)
PY
