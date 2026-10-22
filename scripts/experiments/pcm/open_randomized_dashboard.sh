#!/usr/bin/env bash
# Open the read-only PCM monitoring dashboard on this Linux Mint machine.
set -euo pipefail
ROOT="${1:-/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot}"
PORT="${PCM_DASHBOARD_PORT:-8799}"
export PYTHONPATH="${PYTHONPATH:-.}"
export LION_DATA_PATH="${LION_DATA_PATH:-/mnt/store_ext4/GIT/LION/data}"
export LION_TTV22_PATH="${LION_TTV22_PATH:-/mnt/store_ext4/store/LION/ttv22}"
if [[ ! -d "$ROOT" ]]; then
  printf 'Missing experiment results: %s\n' "$ROOT" >&2
  exit 1
fi
printf 'Open http://127.0.0.1:%s/ (Ctrl+C to stop the dashboard)\n' "$PORT"
python -m LION.pcm.randomized.dashboard --root "$ROOT" --port "$PORT"
