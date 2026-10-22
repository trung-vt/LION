#!/usr/bin/env bash
# NOTE: presets now use raw .txt.map scaling and noise=None; running this
# helper does not reproduce the older normalized-image experiments.
# Run the existing QUICK_METHODS on both full active NPL cell maps.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
export LION_DATA_PATH="${LION_DATA_PATH:-$PWD/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"

# On thome, select the RTX 5090 by default. An externally set value wins.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

presets=(
    unseen_npl_cigs_active_256
    unseen_npl_perovskite_active_512
)
if (( $# > 0 )); then
    presets=("$@")
fi

for preset in "${presets[@]}"; do
    echo "=== PCM unseen single-sample: $preset ==="
    python scripts/experiments/pcm/pcm_test_single_sample.py \
        --log.save-plots \
        --preset "$preset"
done
