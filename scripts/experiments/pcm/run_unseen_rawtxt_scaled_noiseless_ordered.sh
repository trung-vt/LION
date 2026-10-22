#!/usr/bin/env bash
# Re-run the original QUICK methods on NPL raw .txt.map active regions.
# These experiments use the specified physical current scale during reconstruction.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTHONPATH=.
export LION_DATA_PATH="${LION_DATA_PATH:-$PWD/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PCM_PYTHON:-/mnt/others_ext4/installs/micromamba/envs/lion_2/bin/python}"
ROOT="$LION_TTV22_PATH/pcm/results/20261021_npl_rawtxt_scaled_1p8e9_4p0e9_v1"
mkdir -p "$ROOT"
for case in "cigs unseen_npl_cigs_active_256 6" "perovskite unseen_npl_perovskite_active_512 7"; do
    read -r specimen preset coarse <<< "$case"
    # New low-budget comparisons are also ordered, with no added noise.
    # At 5% and 2%, use one fewer coarse level so the fully included
    # coarse block fits inside the sampling budget (1.5625% versus 6.25%).
    for rate in 1.00 0.80 0.50 0.20 0.10 0.05 0.02; do
        case_coarse="$coarse"
        if [[ "$rate" == 0.05 || "$rate" == 0.02 ]]; then
            case_coarse="$((coarse - 1))"
        fi
        rate_tag="${rate/./p}"
        name="${specimen}_s${rate_tag}_ordered_no_added_noise_rawtxt_scaled"
        folder="$ROOT/$specimen/s_${rate_tag}"
        output="$folder/$name"
        mkdir -p "$folder"
        if [[ -f "$folder/COMPLETED" ]]; then
            echo "SKIPPED completed $specimen $rate"
            continue
        fi
        if [[ -d "$output" ]]; then
            echo "ERROR: refusing to overwrite partial outputs at $output"
            exit 1
        fi
        printf 'RUNNING %s %s (coarseJ=%s), GPU %s\n' "$specimen" "$rate" "$case_coarse" "$CUDA_VISIBLE_DEVICES"
        PCM_SINGLE_SAMPLE_NAME="$name" PCM_SINGLE_SAMPLE_NOISE_SEED=20261021 \
          timeout -k 60s 1800s \
          "$PYTHON" scripts/experiments/pcm/pcm_test_single_sample.py \
            --preset "$preset" \
            --sampling.sampling-ratios "$rate" \
            --sampling.coarse-j-values "$case_coarse" \
            --sampling.randomising-scheme None \
            --runtime.root-output-dir "$folder" \
            --runtime.device cuda \
            --log.save-plots \
            noise:none > "$folder/run.log" 2>&1
        count="$(find "$output" -mindepth 2 -maxdepth 2 -name results.csv -type f | wc -l)"
        if [[ "$count" -ne 9 ]]; then
            echo "ERROR: $specimen $rate completed $count methods, expected 9"
            exit 1
        fi
        printf '9\n' > "$folder/COMPLETED"
        printf 'COMPLETED %s %.2f, %d metric rows\n' "$specimen" "$rate" "$count"
    done
done
echo ALL_RAW_TXT_MAP_SCALED_RUNS_COMPLETED
