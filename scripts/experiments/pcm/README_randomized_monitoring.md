# Instructions for the local training/convergence dashboard - PCM randomized-noise monitoring

Existing 2026-10-14 pilot runs already contain per-update training loss,
gradient norm, elapsed time and occasional validation/SSDU selection scores.
They do **not** contain intermediate clean-image PnP convergence metrics.
This patch can replay frozen test cases and generate them without retraining.

## Open the dashboard

From the LION repository root on Linux Mint:

```bash
micromamba activate lion_2
export PYTHONPATH=.
export LION_DATA_PATH=/mnt/store_ext4/GIT/LION/data
export LION_TTV22_PATH=/mnt/store_ext4/store/LION/ttv22
ROOT=/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot
python -m LION.pcm.randomized.dashboard --root "$ROOT" --port 8799
```

Open `http://127.0.0.1:8799/` in a browser. This is local-only, read-only
and *manual-refresh by default*. Optional five-second refresh is available
on the page. Close the server with Ctrl+C. The dashboard displays every
saved training history, validation snapshot (future training), PnP figure
and saved result image. It does not start training itself or access the internet.

## Plot existing training histories

```bash
python - <<'PY'
from pathlib import Path
from LION.pcm.randomized.monitoring import training_figures
print(training_figures(Path("/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot")))
PY
xdg-open "$ROOT/monitoring/figures"
```

Plots show all stored per-update training losses/gradient norms/times and
validation/SSDU selection values at logged checkpoints. For the 2026-10-14
run, validation was sampled every 32 or 128 updates. Missing historical
checkpoint images or additional validation values cannot be reconstructed
without rerunning the saved weights/checkpoints.

## Replay the selected PnP tests to measure per-iteration quality

```bash
python -m LION.pcm.randomized.convergence --root "$ROOT" --device cuda:0
xdg-open "$ROOT/pnp_sweep/convergence/figures"
xdg-open "$ROOT/monitoring/figures"
```

The replay uses the saved selected parameters, checkpoint files and same
held-out test images, sampling masks and noise seeds. It checks the replayed
final MSE against the original result and logs any differences rather than
silently claiming perfect reproduction. In the 2026-10-14 replay, three
GS-DWTDNet cases differed by 0.00024--0.00061 MSE, while 93/96 matched
the selected tolerance of 0.0002. Those 3 traces are diagnostic, not exact
replays of the original output. Metrics include PSNR,
SSIM, image MSE, data-residual MSE, relative iterate changes, iteration
number, elapsed time, and GS objective/step size/backtracking where applicable.
Fixed-iteration snapshots 0, 1, 5, 10, 20 and final are saved in small
compressed NPZ files, with selected labelled PNG contact sheets. The reference
image is only used for reporting, not the reconstruction algorithm or model
selection. Metric calculation adds overhead, so replay timing is not a fair
performance benchmark. The command refuses to overwrite previous replays.

## Future training runs

`history.jsonl` is append-only and records each completed optimizer update,
allowing the dashboard to view progress before the next validation. Each
validation checkpoint also saves a `validation_steps/step_XXXXXX.csv`
with per-case metrics and a fixed natural/synthetic PNG preview.

This does **not** change optimization, checkpoint ranking, learning-rate
schedules, noise simulation or training splits. Existing runs are unchanged.
The dashboard does not refresh until the user clicks Refresh or enables polling.

For broader scientifically meaningful convergence claims, compare a larger,
pre-registered test set and multiple seeds. Flat PSNR alone is not a proof of
fixed-point or objective convergence.

## Monitoring smoke checks

Separate two-update DWTDNet and SSDU U-TV smoke runs verified append-only
update logs and per-validation CSV/PNG snapshots. Those files are stored
under the monitoring_smoke_runs directory and are not part of the original
12-run comparisons. A dashboard HTTP smoke test and 51 selected Python
tests passed.
