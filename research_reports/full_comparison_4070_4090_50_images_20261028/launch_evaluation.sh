#!/usr/bin/env bash
set -euo pipefail
W=/tmp/lion_pcm_utv_zero_4070_train_20261026
OUT=/store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028
PY=/home/t/micromamba/envs/lion_3/bin/python
cd "$W"
export PYTHONPATH="$W" LION_DATA_PATH=/store/LION/ttv22/pcm/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
[ "$(git rev-parse HEAD)" = 9188b0735843c96526537b6b9bfd834fd089f7cd ]
[ -z "$(git status --porcelain)" ]
[ "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | head -1)" = 'NVIDIA GeForce RTX 4070' ]
test -f "$OUT/protocol_manifest_50.json"
for category in CIGS Organic Perovskite Reference Silicone;do
 echo "$(date -u -Is) BEGIN ${category}" | tee -a "$OUT/evaluation_progress.log"
 nice -n 10 "$PY" -u "$OUT/evaluate_current_50.py" --category "$category" > "$OUT/eval_${category}.log" 2>&1
 echo "$(date -u -Is) DONE ${category}" | tee -a "$OUT/evaluation_progress.log"
done
printf '%s ALL_EVALUATIONS_COMPLETED\n' "$(date -u -Is)" | tee -a "$OUT/evaluation_progress.log"
