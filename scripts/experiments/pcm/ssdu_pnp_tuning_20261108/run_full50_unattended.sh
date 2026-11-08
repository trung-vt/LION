#!/usr/bin/env bash
# Resume every remaining SSDU full50 candidate, then freeze both selections.
# Run detached using nohup. No test-set or ground-truth data accessed.
set -Eeuo pipefail

WT=/home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108
OUT=/store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1
PY=/home/t/micromamba/envs/lion_3/bin/python
LOGDIR="$OUT/full50/runner"
mkdir -p "$LOGDIR"
cd "$WT"
exec 9>"$LOGDIR/experiment.lock"
flock -n 9 || { echo "Another full50 supervisor already holds the lock"; exit 66; }
umask 002
export PYTHONDONTWRITEBYTECODE=1
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export LION_TTV22_PATH=/store/LION/ttv22
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

printf "RUNNER_STARTED=%s\n" "$(date -Is)"
printf "SOURCE_HEAD=%s\n" "$(git rev-parse HEAD)"
printf "PILOT_FROZEN=%s\n" "$(test -f "$OUT/pilot5/frozen_pair.json" && echo yes || echo no)"
test -f "$OUT/pilot5/frozen_pair.json"
test -f "$OUT/prepared/fidelity_audit.json"
test -f "$OUT/prepared/adapter_equivalence.json"

# Store code+measurement fingerprints, so a future rerun can identify the exact version.
"$PY" -B - <<'PYTHON'
from pathlib import Path
import hashlib,json,subprocess
out=Path("/store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1")
wt=Path("/home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108")
p=wt/"scripts/experiments/pcm/ssdu_pnp_tuning_20261108"
files=("score_grid.py","solver_adapter.py","ssdu_risk.py","freeze_selection.py",
       "prepared_snapshot/candidate_grid.json")
d={"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=wt,text=True).strip(),
   "files_sha256":{name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in files},
   "selection_has_no_ground_truth":True,
   "test_not_used":True}
f=out/"full50"/"runner"/"numerical_code_manifest.json"
payload=json.dumps(d,indent=2,sort_keys=True)+"\n"
if f.exists():assert f.read_text()==payload,"Source version changed during parameter tuning"
else:f.write_text(payload)
print("CODE_FROZEN_FOR_SEARCH",d["git_head"],flush=True)
PYTHON

(
  export CUDA_VISIBLE_DEVICES=GPU-05267a12-f71f-101d-59e5-05e5c430d8ef
  echo "ADMM_STARTED=$(date -Is)" > "$LOGDIR/admm_status.txt"
  if "$PY" -B -u -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.score_grid \
      --stage full50 --method admm --first 2 --last 15 \
      > "$LOGDIR/admm_run.log" 2>&1; then
    echo "ADMM_EXIT=0 $(date -Is)" > "$LOGDIR/admm_status.txt"
  else
    ec=$?
    echo "ADMM_EXIT=$ec $(date -Is)" > "$LOGDIR/admm_status.txt"
    exit "$ec"
  fi
) &
PIDA=$!

(
  export CUDA_VISIBLE_DEVICES=GPU-fcdc8ea0-7187-86d7-15ba-615774f8801f
  echo "GS_STARTED=$(date -Is)" > "$LOGDIR/gs_status.txt"
  if "$PY" -B -u -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.score_grid \
      --stage full50 --method gs --first 2 --last 18 \
      > "$LOGDIR/gs_run.log" 2>&1; then
    echo "GS_EXIT=0 $(date -Is)" > "$LOGDIR/gs_status.txt"
  else
    ec=$?
    echo "GS_EXIT=$ec $(date -Is)" > "$LOGDIR/gs_status.txt"
    exit "$ec"
  fi
) &
PIDG=$!

echo "ADMM_PID=$PIDA GS_PID=$PIDG" > "$LOGDIR/worker_pids.txt"
echo "BOTH_SEARCHES_LAUNCHED $(date -Is)"
EXIT_A=0
EXIT_G=0
wait "$PIDA" || EXIT_A=$?
wait "$PIDG" || EXIT_G=$?
echo "WORKERS_EXIT admm=$EXIT_A gs=$EXIT_G $(date -Is)"

if (( EXIT_A != 0 || EXIT_G != 0 )); then
  echo "SEARCH_INCOMPLETE: non-zero worker exit" > "$LOGDIR/status.txt"
  exit 1
fi
"$PY" -B - <<'PYTHON'
from pathlib import Path
root=Path("/store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1")
for method,required in (("admm",15),("gs",18)):
    d=root/"full50"/method/"candidates"
    summaries=list(d.glob("*/summary.json"))
    failures=list(d.glob("*/*.failure.json"))
    print("GRID_VERIFY",method,len(summaries),required,"failures",len(failures),flush=True)
    if len(summaries)!=required or failures:raise RuntimeError("Incomplete SSDU full50 grid: "+method)
PYTHON

"$PY" -B -u -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.freeze_selection \
 --stage full50 > "$LOGDIR/freeze.log" 2>&1
test -s "$OUT/full50/frozen_pair.json"
echo "FULL50_COMPLETE_AND_FROZEN $(date -Is)" > "$LOGDIR/status.txt"
echo "COMPLETE=$(date -Is) $OUT/full50/frozen_pair.json"
