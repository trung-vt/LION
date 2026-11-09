#!/usr/bin/env bash
# Only 'execute' starts the full experimental workflow.
set -Eeuo pipefail
ACTION=status
if [ "$#" -gt 0 ];then ACTION="$1";fi
WORKTREE=/home/t/Documents/GIT/LION_pcm_tune1_ternary_20261109
RESULTS=/store/LION/ttv22/pcm/results/pcm/tune1_ternary_20261109_v1
PY=/home/t/micromamba/envs/lion_3/bin/python
MOD=scripts.experiments.pcm.tune1_ternary_20261109
SRC="$WORKTREE/scripts/experiments/pcm/tune1_ternary_20261109"
GPU_ADMM=GPU-05267a12-f71f-101d-59e5-05e5c430d8ef
GPU_GS=GPU-fcdc8ea0-7187-86d7-15ba-615774f8801f
cd "$WORKTREE"
export PYTHONDONTWRITEBYTECODE=1
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export LION_TTV22_PATH=/store/LION/ttv22
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

case "$ACTION" in
  status)
    "$PY" -B -m "$MOD.status"
    ;;
  prepare)
    "$PY" -B -m "$MOD.prepare"
    ;;
  tune-admm|tune-gs)
    if [ "$ACTION" = tune-admm ];then method=admm;uuid="$GPU_ADMM"
    else method=gs;uuid="$GPU_GS";fi
    export CUDA_VISIBLE_DEVICES="$uuid"
    for cat in CIGS Organic Perovskite Reference Silicone;do
      "$PY" -B -u -m "$MOD.run_tune1" --method "$method" --category "$cat"
    done
    ;;
  test-admm|test-gs)
    if [ "$ACTION" = test-admm ];then method=admm;uuid="$GPU_ADMM"
    else method=gs;uuid="$GPU_GS";fi
    export CUDA_VISIBLE_DEVICES="$uuid"
    for cat in CIGS Organic Perovskite Reference Silicone;do
      "$PY" -B -u -m "$MOD.test45_run" --method "$method" --category "$cat"
    done
    ;;
  replay-baselines)
    export CUDA_VISIBLE_DEVICES="$GPU_ADMM"
    for cat in CIGS Organic Perovskite Reference Silicone;do
      "$PY" -B -u -m "$MOD.replay_baselines" --category "$cat"
    done
    ;;
  replay-c60)
    export CUDA_VISIBLE_DEVICES="$GPU_GS"
    for cat in CIGS Organic Perovskite Reference Silicone;do
      "$PY" -B -u -m "$MOD.replay_c60" --category "$cat"
    done
    ;;
  analyse)
    "$PY" -B -u -m "$MOD.analyse"
    ;;
  galleries)
    for cat in CIGS Organic Perovskite Reference Silicone;do
      "$PY" -B -u -m "$MOD.gallery" --category "$cat"
    done
    command -v pdfunite >/dev/null
    pdfunite "$RESULTS/analysis/galleries/CIGS_9_pages.pdf" \
             "$RESULTS/analysis/galleries/Organic_9_pages.pdf" \
             "$RESULTS/analysis/galleries/Perovskite_9_pages.pdf" \
             "$RESULTS/analysis/galleries/Reference_9_pages.pdf" \
             "$RESULTS/analysis/galleries/Silicone_9_pages.pdf" \
             "$RESULTS/analysis/galleries/Test45_All_Reconstructions_45_Pages.pdf"
    test "$(pdfinfo "$RESULTS/analysis/galleries/Test45_All_Reconstructions_45_Pages.pdf" |
            awk '/^Pages:/{print $2}')" = 45
    ;;
  report)
    "$PY" -B -m "$MOD.report" --stage completed
    cd "$RESULTS/report"
    latexmk -pdf -interaction=nonstopmode -halt-on-error PCM_Tune1_Ternary_Search_Report.tex
    ;;
  execute)
    mkdir -p "$RESULTS/logs"
    exec 9>"$RESULTS/logs/controller.lock"
    flock -n 9 || { echo 'Another tune1 controller has the lock';exit 66; }
    "$PY" -B -m "$MOD.prepare"
    echo "RUNNING_TUNING $(date -Is)" > "$RESULTS/logs/status.txt"
    bash "$SRC/run_all.sh" tune-admm > "$RESULTS/logs/tune_admm.log" 2>&1 & A=$!
    bash "$SRC/run_all.sh" tune-gs > "$RESULTS/logs/tune_gs.log" 2>&1 & B=$!
    a=0;b=0
    wait "$A" || a=$?
    wait "$B" || b=$?
    if ((a!=0 || b!=0));then
      echo "TUNING_FAILED_ADMM_$a_GS_$b" > "$RESULTS/logs/status.txt"
      exit 1
    fi
    "$PY" -B - <<'PYTHON'
from scripts.experiments.pcm.tune1_ternary_20261109.test45_run import assert_all_frozen
a,t,s=assert_all_frozen()
assert len(s)==50 and len(t)==45
print("BOTH_TUNERS_FROZEN_BEFORE_TEST")
PYTHON
    echo "RUNNING_TEST45 $(date -Is)" > "$RESULTS/logs/status.txt"
    bash "$SRC/run_all.sh" test-admm > "$RESULTS/logs/test_admm.log" 2>&1 & A=$!
    bash "$SRC/run_all.sh" test-gs > "$RESULTS/logs/test_gs.log" 2>&1 & B=$!
    a=0;b=0
    wait "$A" || a=$?
    wait "$B" || b=$?
    if ((a!=0 || b!=0));then
      echo "TEST_FAILED_ADMM_$a_GS_$b" > "$RESULTS/logs/status.txt"
      exit 1
    fi
    echo "REPLAYING_BASELINES $(date -Is)" > "$RESULTS/logs/status.txt"
    bash "$SRC/run_all.sh" replay-baselines > "$RESULTS/logs/baselines.log" 2>&1 & A=$!
    bash "$SRC/run_all.sh" replay-c60 > "$RESULTS/logs/c60.log" 2>&1 & B=$!
    a=0;b=0
    wait "$A" || a=$?
    wait "$B" || b=$?
    if ((a!=0 || b!=0));then
      echo "BASELINE_REPLAY_FAILED_$a_$b" > "$RESULTS/logs/status.txt"
      exit 1
    fi
    bash "$SRC/run_all.sh" analyse > "$RESULTS/logs/analysis.log" 2>&1
    bash "$SRC/run_all.sh" galleries > "$RESULTS/logs/galleries.log" 2>&1
    bash "$SRC/run_all.sh" report > "$RESULTS/logs/report.log" 2>&1
    echo "TUNE1_TEST45_COMPLETE $(date -Is)" > "$RESULTS/logs/status.txt"
    ;;
  *)
    echo "Usage: run_all.sh status|prepare|tune-admm|tune-gs|test-admm|test-gs|replay-baselines|replay-c60|analyse|galleries|report|execute" >&2
    exit 2
    ;;
esac
