#!/usr/bin/env bash
# The numerical sweep requires an explicit 'execute' argument.
set -Eeuo pipefail
ACTION=status
if [ "$#" -gt 0 ];then ACTION="$1";fi
WT=/home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108
DIR=/store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/next_gs_ssdu_sweep_v1
PY=/home/t/micromamba/envs/lion_3/bin/python
cd "$WT"
export PYTHONDONTWRITEBYTECODE=1
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export LION_TTV22_PATH=/store/LION/ttv22
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
MOD=scripts.experiments.pcm.ssdu_pnp_tuning_20261108.expanded_gs
REF=scripts.experiments.pcm.ssdu_pnp_tuning_20261108.refresh_expanded_gs_report
case "$ACTION" in
  status)
    "$PY" -B -m "$MOD" status
    ;;
  prepare)
    "$PY" -B -m "$MOD" prepare
    "$PY" -B -m "$REF"
    ;;
  execute)
    test -s "$DIR/prepared/manifest.json"
    test ! -e "$DIR/frozen_selection.json"
    mkdir -p "$DIR/logs"
    exec 9>"$DIR/logs/sweep.lock"
    flock -n 9 || { echo 'Another expanded GS sweep holds the lock';exit 66; }
    export CUDA_VISIBLE_DEVICES=GPU-fcdc8ea0-7187-86d7-15ba-615774f8801f
    date -Is
    "$PY" -B -u -m "$MOD" run --first 1 --last 27 2>&1 | tee -a "$DIR/logs/gs_expanded_sweep.log"
    "$PY" -B -u -m "$MOD" freeze 2>&1 | tee -a "$DIR/logs/freeze.log"
    "$PY" -B -m "$REF"
    cd "$DIR/report"
    latexmk -pdf -interaction=nonstopmode -halt-on-error GS_PnP_Expanded_SSDU_Tuning_Report.tex
    ;;
  *)
    echo 'Usage: run_expanded_gs.sh status|prepare|execute' >&2
    exit 2
    ;;
esac
