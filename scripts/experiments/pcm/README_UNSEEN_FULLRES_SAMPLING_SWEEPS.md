# Full-resolution NPL sampling and inference-parameter sweeps

**Historical record, not a current runbook.** These results were made with
the earlier clipped/normalized unseen-sample presets. The current
`unseen_npl_*` presets instead load signed raw `.txt.map` files and default
to `noise=None`. In particular, the legacy runners' `noisy_*` labels would
**not** reproduce the original noise conditions without explicit noise
configuration. Do not compare a fresh run with the saved October 2026
results as if the scaling and noise model were unchanged. For the current
validated protocol, see `README_NPL_RAWTXT_SCALING_CORRECTION.md` and
`run_unseen_rawtxt_scaled_noiseless_ordered.sh`.

Status: historical engineering sweep completed 21 October 2026 on thome. Both
source images were measured as full current maps, but **every 20% or other
undersampled acquisition is simulated**, not independently acquired at NPL.

## Commands and GPUs

From the LION repository on thome, in environment `lion_2`:

```bash
bash scripts/experiments/pcm/run_unseen_cigs_fullres_sampling.sh
bash scripts/experiments/pcm/run_unseen_perovskite_fullres_sampling.sh
```

The CIGS (full 256 x 256 active region) script pins to GPU 0, RTX 4090.
The perovskite (full 512 x 512 active region) script pins to GPU 1, RTX 5090.

Already running in detached tmux sessions:

```bash
tmux -L pcm20261021 ls
tmux -L pcm20261021 attach -t cigs_gpu0
tmux -L pcm20261021 attach -t perov_gpu1
```

Detach from a session with Ctrl+B then D. Stopping the attached terminal
without detaching can stop a GPU experiment; avoid Ctrl+C unless intending to
terminate the current run.

Output root on thome:

```text
/mnt/store_ext4/store/LION/ttv22/pcm/results/20261021_unseen_fullres_sampling_sweep_v1/
```

Monitor with:

```bash
ROOT=/mnt/store_ext4/store/LION/ttv22/pcm/results/20261021_unseen_fullres_sampling_sweep_v1
tail -f "$ROOT/cigs/master.log"
tail -f "$ROOT/perovskite/master.log"
tail -n 25 "$ROOT/cigs/progress.tsv"
tail -n 25 "$ROOT/perovskite/progress.tsv"
cat "$ROOT/cigs/current.txt" "$ROOT/perovskite/current.txt"
nvidia-smi
python scripts/experiments/pcm/summarize_unseen_fullres_sampling.py --write
```

The summary command reads only cases with a `COMPLETED` marker and atomically
updates `summary_completed.csv`. It does not pick the best test-set settings.

The scripts are restartable. Existing successful cases have a `COMPLETED`
marker and are skipped. Any partial output is renamed with `_interrupted_...`
rather than overwritten. Failed cases are marked `FAILED`, and the loop
continues; setting `RETRY_FAILED=1` permits retry on a later invocation.
Each case also saves the complete normal `pcm_test_single_sample.py` log in
`run.log`, the original `config.yaml`, per-method metrics, reconstruction
arrays, PNG figures, absolute errors and sampling mask.

## Experimental grid

Sampling percentages **1, 5, 10, 20, 30, 40, 50, 80**. The corresponding
Hadamard `coarseJ` values guarantee that the complete coarse block fits:

| Sampling | CIGS J=8 coarseJ | Perovskite J=9 coarseJ |
| --- | ---: | ---: |
| 1% | 4 | 5 |
| 5% | 5 | 6 |
| 10%, 20%, 30%, 40%, 50%, 80% | 6 | 7 |

The original base presets do **not** change. New derived presets are registered
in `LION/experiments/pcm_presets/presets.py`, with implementations in
`LION/experiments/pcm_presets/unseen_sweep_presets.py`.

The base profile uses all eight existing `QUICK_METHODS` at each sampling
level under four noise/pattern combinations:

- noisy randomized; noiseless randomized
- noisy first-in-order; noiseless first-in-order

Each parameter variant uses two cases: noisy randomized and noiseless randomized.
It only runs the relevant changed methods (zero-fill is included automatically),
reusing the same mask and noise seed across corresponding conditions/profiles.
This avoids rerunning unchanged neural networks hundreds of times.

| Variant suffix | Methods rerun | Changes relative to QUICK_METHODS |
| --- | --- | --- |
| `admm_rho_003` | ADMM DRUNet, ADMM DWTDNet | Data proximal `rho=0.003` (was 0.01) |
| `admm_rho_100` | Both ADMM | `rho=0.1` |
| `pnp_sigma_020` | Four PnP | Denoiser sigma 0.02 (was 0.05) |
| `pnp_sigma_100` | Four PnP | Denoiser sigma 0.1 |
| `gs_lambda_030` | Both GS-PnP | Explicit GS regularisation lambda 0.03 (was 0.1) |
| `gs_lambda_300` | Both GS-PnP | lambda 0.3 |
| `tv_lambda_005` | Scalar TV | TV lambda 0.005 (was 0.02) |
| `tv_lambda_050` | Scalar TV | TV lambda 0.05 |
| `spgl1_db2` | SPGL1 | db2 wavelet (was db4) |
| `iteration_cap_x2` | Four PnP, TV, SPGL1 | PnP max iterations 100 vs 50, TV 2048 vs 1024, SPGL1 200 vs 100 |

**Important implementation detail:** `DWTDenoiseNet.forward(x, sigma=...)` in
`LION/models/fully_learned/DWTDenoiseNet.py` currently accepts but **does not
use** the supplied `sigma`. Therefore the DWTDNet parts of the `pnp_sigma_020`
and `pnp_sigma_100` profiles are effectively negative controls, not meaningful
DWTDNet noise-level tuning. The DRUNet parts can respond to this parameter.
The ADMM `rho` and GS-PnP `lambda` sweeps still change the DWTDNet-based
reconstruction algorithms.

U-TV checkpoints are held fixed. Their training selected the 20%, coarseJ=6
case, and they are only evaluated at the other sampling levels. For GS-PnP,
the existing early-stopping rule remains enabled; a higher iteration cap may
not increase the number of accepted iterations.

There are 32 base acquisitions and 160 profile acquisitions **per specimen**,
192 experiment cases per GPU when every case is completed. Parameter variants
do not rerun unrelated methods, so method-result counts vary by profile.
The scripts can take substantial time, especially at 512 x 512 and high
sampling fractions.

The new optional environment variables `PCM_SINGLE_SAMPLE_NAME` and
`PCM_SINGLE_SAMPLE_NOISE_SEED` in `pcm_test_single_sample.py` ensure matched
sampling/noise across profiles. With these unset the original script keeps
its previous behavior.

## Scientific interpretation

These are **inference-parameter explorations on unseen test specimens**.
Do not use the two unseen test images to select hyperparameters for a final,
independent paper test. Use separate validation images for parameter selection;
these runs can reveal failure modes and guide future validation experiments.

References within this repository:
- `LION/pcm/noise.py`, `PCMNoiseSimulation.npl_default`
- `LION/pcm/run_pcm_single_sample_experiment.py`, sampling and trial logic
- `LION/pcm_tools/pcm_sampling.py`, Hadamard row selection
- `LION/experiments/pcm_presets/presets.py`, original `QUICK_METHODS`
