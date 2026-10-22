#!/usr/bin/env bash
# HISTORICAL: 2026-10-21 pre-correction normalized-image result path.
# Running today with the corrected raw .txt.map presets does not reproduce
# those old results; use run_unseen_rawtxt_scaled_noiseless_ordered.sh instead.
# Reproduce noiseless 100% ordered PCM experiments for the two unseen NPL maps.
# All compressed measurements are simulated from the saved full inversion maps.
# The source files and the existing QUICK_METHODS are unchanged.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
export LION_DATA_PATH="${LION_DATA_PATH:-$PWD/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PCM_PYTHON:-/mnt/others_ext4/installs/micromamba/envs/lion_2/bin/python}"
ROOT="${LION_TTV22_PATH}/pcm/results/20261021_npl_100pct_noiseless_ordered_quick"
mkdir -p "$ROOT"

for case in "cigs unseen_npl_cigs_active_256 6" "perovskite unseen_npl_perovskite_active_512 7"; do
    read -r specimen preset coarse <<< "$case"
    sample="${specimen}_100pct_noiseless_ordered_matched"
    folder="$ROOT/$specimen"
    output="$folder/$sample"
    mkdir -p "$folder"
    if [[ -f "$folder/COMPLETED" ]]; then
        echo "SKIP completed: $output"
        continue
    fi
    if [[ -e "$output" ]]; then
        echo "ERROR: refusing to overwrite an incomplete existing run at $output"
        exit 1
    fi
    echo "RUNNING $specimen, 100% ordered/noise:none on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    PCM_SINGLE_SAMPLE_NAME="$sample" PCM_SINGLE_SAMPLE_NOISE_SEED=20261021 \
      timeout -k 60s 3600s \
      "$PYTHON" scripts/experiments/pcm/pcm_test_single_sample.py \
          --preset "$preset" \
          --sampling.sampling-ratios 1.0 \
          --sampling.coarse-j-values "$coarse" \
          --sampling.randomising-scheme None \
          --runtime.root-output-dir "$folder" \
          --runtime.device cuda \
          --log.save-plots > "$folder/run.log" 2>&1
    count="$(find "$output" -mindepth 2 -maxdepth 2 -name results.csv -type f | wc -l)"
    if [[ "$count" != 9 ]]; then
        echo "ERROR: $specimen ended with $count/9 method results; inspect $folder/run.log"
        exit 1
    fi
    printf '%s\n' "$sample" > "$folder/COMPLETED"
    echo "COMPLETED $specimen, $count method results: $output"
done
echo ALL_COMPLETED
