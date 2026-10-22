#!/usr/bin/env bash
# HISTORICAL: this sweep originally ran with clipped normalized-image presets.
# The current raw-text-map presets default to noise=None; the labels
# 'noisy_randomized'/'noisy_in_order' no longer imply noise is enabled.
# Do not rerun to reproduce historical results without correcting the noise model.
# Shared long-running PCM sweep loop, invoked by the two per-specimen scripts.
# Preserve all partial outputs and avoid silent selection on unseen test data.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH=.
export LION_DATA_PATH="${LION_DATA_PATH:-$PWD/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# Default to the known-good thome micromamba environment even in detached tmux.
PCM_PYTHON="${PCM_PYTHON:-/mnt/others_ext4/installs/micromamba/envs/lion_2/bin/python}"
test -x "$PCM_PYTHON"

: "${SPECIMEN:?SPECIMEN is required}"
: "${BASE_PRESET:?BASE_PRESET is required}"
: "${J_ORDER:?J_ORDER is required}"
: "${GPU_INDEX:?GPU_INDEX is required}"
: "${SAMPLING_RATIOS[0]:?sampling rates are required}"

SWEEP_ROOT="${UNSEEN_LARGE_SWEEP_ROOT:-$LION_TTV22_PATH/pcm/results/20261021_unseen_fullres_sampling_sweep_v1}"
ROOT="$SWEEP_ROOT/$SPECIMEN"
mkdir -p "$ROOT"
PROGRESS="$ROOT/progress.tsv"
if [[ ! -f "$PROGRESS" ]]; then
    printf 'timestamp\tprofile\tcondition\tsampling_ratio\tcoarse_j\tstatus\tmethod_results\toutput\n' > "$PROGRESS"
fi
declare -A EXPECTED_ROWS=(
    [base]=9
    [admm_rho_003]=3
    [admm_rho_100]=3
    [pnp_sigma_020]=5
    [pnp_sigma_100]=5
    [gs_lambda_030]=3
    [gs_lambda_300]=3
    [tv_lambda_005]=2
    [tv_lambda_050]=2
    [spgl1_db2]=2
    [iteration_cap_x2]=7
)
PROFILES=(
    base admm_rho_003 admm_rho_100 pnp_sigma_020 pnp_sigma_100
    gs_lambda_030 gs_lambda_300 tv_lambda_005 tv_lambda_050
    spgl1_db2 iteration_cap_x2
)
BASE_CONDITIONS=(noisy_randomized noiseless_randomized noisy_in_order noiseless_in_order)
VARIANT_CONDITIONS=(noisy_randomized noiseless_randomized)

log_status() {
    local profile="$1" condition="$2" ratio="$3" coarse="$4" status="$5" rows="$6" path="$7"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$profile" "$condition" "$ratio" "$coarse" "$status" "$rows" "$path" \
        >> "$PROGRESS"
    printf '[%s] %-21s %-22s rate=%-5s cJ=%-2s %-9s rows=%s\n' \
        "$(date -Is)" "$profile" "$condition" "$ratio" "$coarse" "$status" "$rows"
}

for profile in "${PROFILES[@]}"; do
    preset="$BASE_PRESET"
    if [[ "$profile" != base ]]; then preset="${BASE_PRESET}_${profile}"; fi
    if [[ "$profile" == base ]]; then
        conditions=("${BASE_CONDITIONS[@]}")
    else
        conditions=("${VARIANT_CONDITIONS[@]}")
    fi
    for ratio in "${SAMPLING_RATIOS[@]}"; do
        # At 1%, 5%, >=10%, respectively: the coarse block occupies
        # 0.39%, 1.56%, and 6.25% and fits completely into the budget.
        if [[ "$ratio" == 0.01 ]]; then
            coarse=$((J_ORDER - 4))
        elif [[ "$ratio" == 0.05 ]]; then
            coarse=$((J_ORDER - 3))
        else
            coarse=$((J_ORDER - 2))
        fi
        rate_tag="${ratio/./p}"
        shared_name="${SPECIMEN}_s${rate_tag}_cJ${coarse}_matched"
        for condition in "${conditions[@]}"; do
            case_root="$ROOT/$profile/$condition/s_${rate_tag}_cJ_${coarse}"
            output="$case_root/$shared_name"
            complete="$case_root/COMPLETED"
            failed="$case_root/FAILED"
            if [[ -f "$complete" ]]; then
                log_status "$profile" "$condition" "$ratio" "$coarse" SKIPPED \
                    "$(cat "$complete")" "$output"
                continue
            fi
            if [[ -f "$failed" && "${RETRY_FAILED:-0}" != "1" ]]; then
                log_status "$profile" "$condition" "$ratio" "$coarse" FAIL_SKIP 0 "$output"
                continue
            fi
            mkdir -p "$case_root"
            if [[ -d "$output" ]]; then
                mv "$output" "${output}_interrupted_$(date +%Y%m%dT%H%M%S)"
            fi
            rm -f "$failed"
            args=(
                --preset "$preset"
                --log.save-plots
                --sampling.sampling-ratios "$ratio"
                --sampling.coarse-j-values "$coarse"
                --runtime.root-output-dir "$case_root"
                --runtime.device cuda
            )
            if [[ "$condition" == *in_order ]]; then
                args+=(--sampling.randomising-scheme None)
            fi
            if [[ "$condition" == noiseless_* ]]; then
                args+=(noise:none)
            fi
            log_status "$profile" "$condition" "$ratio" "$coarse" STARTED 0 "$output"
            printf '%s\n' "$profile | $condition | $ratio | coarseJ=$coarse" > "$ROOT/current.txt"

            if PCM_SINGLE_SAMPLE_NAME="$shared_name" \
               PCM_SINGLE_SAMPLE_NOISE_SEED=20261021 \
               timeout -k 60s 1800s \
               "$PCM_PYTHON" scripts/experiments/pcm/pcm_test_single_sample.py \
                   "${args[@]}" >"$case_root/run.log" 2>&1; then
                count=0
                if [[ -d "$output" ]]; then
                    count="$(find "$output" -mindepth 2 -maxdepth 2 \
                        -name results.csv -type f | wc -l)"
                fi
                if [[ "$count" == "${EXPECTED_ROWS[$profile]}" ]]; then
                    printf '%s\n' "$count" > "$complete"
                    log_status "$profile" "$condition" "$ratio" "$coarse" COMPLETED \
                        "$count" "$output"
                else
                    printf 'Expected %s result CSVs; found %s\n' \
                        "${EXPECTED_ROWS[$profile]}" "$count" > "$failed"
                    log_status "$profile" "$condition" "$ratio" "$coarse" INCOMPLETE \
                        "$count" "$output"
                fi
            else
                status="$?"
                printf 'Nonzero return code %s. Inspect run.log\n' "$status" > "$failed"
                log_status "$profile" "$condition" "$ratio" "$coarse" FAILED \
                    "$status" "$output"
            fi
        done
    done
done
printf 'Finished at %s\n' "$(date -Is)" > "$ROOT/sweep_finished.txt"
if awk -F '\t' '$6=="FAILED" || $6=="INCOMPLETE" {bad=1} END {exit !bad}' "$PROGRESS"; then
    printf 'Some runs failed: review %s\n' "$PROGRESS"
else
    printf 'All runs completed or were previously completed.\n'
fi
