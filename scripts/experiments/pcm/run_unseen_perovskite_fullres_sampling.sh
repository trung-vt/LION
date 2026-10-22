#!/usr/bin/env bash
# HISTORICAL: paired with run_unseen_fullres_sampling_common.sh;
# current raw-map presets do not reproduce the old noisy experiments.
# Full active perovskite sampling and inference-parameter sweep.
# GPU 1; use CUDA_VISIBLE_DEVICES in the shared loop.
# Runs all eight sampling levels and the original QUICK_METHODS baseline.
# Parameter variants each change one controlled method-parameter family.
set -euo pipefail

GPU_INDEX=1
SPECIMEN=perovskite
BASE_PRESET=unseen_npl_perovskite_active_512
J_ORDER=9

# Requested sampling fractions: 1%, 5%, 10%, 20%, 30%, 40%, 50%, 80%.
SAMPLING_RATIOS=(0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.80)

source "$(dirname "$0")/run_unseen_fullres_sampling_common.sh"
