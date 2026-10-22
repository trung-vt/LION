# Randomized-noise PCM pilots

This suite separates image-domain denoising, physical PCM reconstruction, and
validation-only PnP tuning. The 2026-10-14 runs are bounded pilots, not converged
models or final paper evidence. No datasets were downloaded and no Git commits were made.

## Environment

Run from the LION repository root:

```bash
micromamba activate lion_2
export PYTHONPATH=.
export OMP_NUM_THREADS=2
export LION_DATA_PATH=/mnt/store_ext4/GIT/LION/data
export LION_TTV22_PATH=/mnt/store_ext4/store/LION/ttv22
export ROOT=/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot
```

## Data

`data_manifest.json` records absolute source paths, dimensions, SHA-256 hashes,
domain labels and split membership. Pools are:

| Domain | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Natural | 800 | 100 | 68 |
| Synthetic PCM | 4000 | 500 | 500 |

Eight hundred duplicate natural files were excluded. Synthetic image IDs are
kept in the same split across material categories. Underlying synthetic template
ancestry is unknown, so template-disjoint performance is not claimed.

Training samples natural and synthetic pools with equal probability, uses
64-pixel grayscale crops, and applies flips/rotations. Each denoiser pilot used
256 crop draws from 243 distinct sources. Each regularizer pilot used 1024 draws
from 861 sources. Indexing all 4800 images does not mean every image was trained.
The fixed evaluation panel has only four source crops per split: two per domain.

To create a new manifest (existing manifests are not overwritten):

```bash
python scripts/experiments/pcm/prepare_randomized_data.py \
  --output "$ROOT/data_manifest_new.json" \
  --natural-train \
    /store/LION/datasets/Download_df2k/datasets/DIV2K_train_HR \
    /store/LION/datasets/Download_df2k/datasets/DF2K/DF2K_train_HR \
  --natural-validation \
    /store/LION/datasets/Download_df2k/datasets/DIV2K/DIV2K_valid_HR \
  --natural-test /store/LION/datasets/deepinv/CBSD68/0 \
  --synthetic-root "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data" \
  --seed 20261014
```

## Noise interpretation

Independent parameters are drawn for each training crop:

| Parameter | Uniform range |
| --- | --- |
| Drift-slope SD `sigma_g` | `[0, 3e-7]` |
| White-bias SD `sigma_bias` | `[0, 4e-5]` |
| ARIMA innovation SD `sigma_beam` | `[0, 1e-3]` |
| Additive dark SD `sigma_dark` | `[0, 1]` |

AR intervals are `[-.93,-.92]`, `[-.18,-.17]`, `[-.08,-.07]`, `[-.04,-.03]`.
MA intervals are `[.03,.04]`, `[-.37,-.36]`. ARMA increments must be stationary
and invertible; integration order remains `d=1`. The integrated beam trace is
not stationary. Signal-to-bias ratio is 0.3.

The image surrogate uses rasterized multiplicative beam and additive bias,
drift and dark noise, removing the constant illumination offset. It contains no
Hadamard transform or subsampling and is not a model of backprojected PCM noise.
Mixture training draws this surrogate or Gaussian noise with equal probability.
Clean crops are in `[0,1]`; noisy inputs and outputs are not clipped for metrics
or training. DRUNet receives nominal dark/Gaussian SD, not oracle residual RMS.
Current DWTDNet ignores its sigma argument.

Physical reconstruction uses the existing differential-pattern simulator.
Sampling is uniform on `[.01,1]`, with row counts rounded upward, valid coarseJ
and DC retained. It uses the exact-count coarse-block plus uniform-tail sampler.
The older multilevel allocator failed high-rate row-count boundary checks and
was not used or modified by these new pilots.

## Training commands

Denoisers use Adam, learning rate 0.0002, gradient clipping at one and effective
batch size two. Gaussian-only controls share initialization and crop sequence
with mixture runs. All models start from random initialization.

```bash
python scripts/experiments/pcm/randomized_noise.py \
  --manifest "$ROOT/data_manifest.json" --output-root "$ROOT/reproduction_runs" \
  --models dwtdnet gs_dwtdnet drunet gs_drunet \
  --noise-modes gaussian mixture --steps 128 --side 64 --batch-size 2 \
  --validation-every 32 --validation-per-domain 2 --device cuda:0
```

The four regularizer pilots use 512 updates and 64 unrolled PDHG iterations:

```bash
python scripts/experiments/pcm/randomized_noise.py \
  --manifest "$ROOT/data_manifest.json" --output-root "$ROOT/reproduction_runs" \
  --models utv_supervised utgv_supervised utv_ssdu utgv_ssdu \
  --steps 512 --side 64 --batch-size 2 --unroll-iterations 64 \
  --validation-every 128 --validation-per-domain 2 --device cuda:0
```

These commands run sequentially on one GPU. The recorded campaign used GPU 0
for DWT/TV and GPU 1 for DRUNet/TGV. Check resource occupancy before another run.
The CLI refuses to overwrite run folders. Checkpoints include optimizer and RNG
states, but automatic resumption is not implemented.

Supervised reconstruction minimizes clean-image MSE. SSDU splits one generated
noisy acquisition into disjoint input and held-out rows, keeping DC on the
input side. Both SSDU training and checkpoint selection use held-out measurement
loss, not clean targets. Clean-image validation metrics are diagnostic only.
Noise can correlate across the split, so unbiasedness is not assumed. All
available rows are used at inference.

## PnP tuning

```bash
python -m LION.pcm.randomized.sweep \
  --manifest "$ROOT/data_manifest.json" --runs "$ROOT/runs" \
  --output "$ROOT/pnp_sweep_reproduction" --device cuda:0 \
  --iterations 20 --per-domain 2
```

Ordinary priors use existing LION PnP-ADMM, initialized at zero. GS priors use
proximal gradient on an explicit potential, initialized at the adjoint, with
LION's exact PCM proximal. Initialization policies are therefore different.
GS stops at relative change `1e-6` or the iteration budget and returns the last
accepted iterate. This numerical safeguard is not a convergence certificate.

Search axes are ADMM rho `{.01,.1,1,10}`, GS lambda `{.01,.1,1}`, GS initial step
`{.5,2}`, and DRUNet sigma `{.05,.2}`. DWTDNet sigma is not swept. Rates are
`{.01,.2,1}` and dark SDs `{.05,.2}`, at beam innovation SD `.0006`. Candidates
share acquisitions. Validation MSE selects each winner before separate test
evaluation. Test scores never select parameters or best iterates.

`pnp_sweep_initial_line_search_failure/` preserves the first failed attempt,
which hit float32 stagnation. `pnp_sweep/` is the completed corrected run:
720 validation reconstructions, 96 test reconstructions and 24 selected settings.

## Matched comparison and reports

Every run saves configurations, all noise/source draws, training curves,
validation/test CSVs, `best.pt`, `last.pt`, and fixed-case prediction arrays.
Reproduce reports in this order:

```bash
python -m LION.pcm.randomized.audit --root "$ROOT"
python -m LION.pcm.randomized.figures --root "$ROOT"
python -m LION.pcm.randomized.report --root "$ROOT"
python -m LION.pcm.randomized.matched --root "$ROOT" --device cuda:0
```

The final command evaluates the four fixed adaptive checkpoints on the exact
PnP test acquisitions (96 additional reconstructions) and inserts the matched
comparison into the generated report and slide source. Run it after `report`,
which regenerates the base documents. No checkpoint or parameter is retuned.
Source hashes and baseline measurement statistics are checked for equality.
This is matched-data testing, not a resource-matched benchmark: adaptive methods
use 64 PDHG steps while PnP uses at most 20 iterations.

`report_bundle/` holds LaTeX reports, include-ready manuscript text, Beamer
frames, figures, tables and per-case results. Representative images are fixed
in advance, use common display scales and do not determine quantitative scores.

The current pilots use one training seed and four distinct test source crops.
Longer training, larger test panels, resolution-dependent noise calibration and
an independent-noise SSDU ablation are required before final paper claims.

## Validation

The combined selected regression run passed 86 tests, with six GPU/CT-marked
cases deselected. Twenty-eight of these tests are new:

```bash
python -m pytest -q -o addopts='' \
  pytests/pcm/test_randomized_noise.py \
  pytests/pcm/test_randomized_sweep.py
```

Checks include sampling endpoints, parameter bounds, image corruption without
physics, float64 zero-noise identities, nonleaking SSDU partitions, original
noisy measurement reuse, measurement-only checkpoint selection, ADMM recovery
and GS objective steps. The eight model types also passed finite, nonzero
forward/backward gradient smoke checks. Existing simulator/reconstructor files
were not modified. Black and Ruff were unavailable in the selected environment;
syntax and tests were run, but no formatter check is claimed.

## References

- Hurault, Leclaire and Papadakis, *Gradient Step Denoiser for convergent Plug-and-Play*, ICLR 2022: https://arxiv.org/abs/2110.03220
- Yaman et al., *Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data*, MRM 2020: https://arxiv.org/abs/1912.07669
- Agustsson and Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, CVPR Workshops 2017: https://data.vision.ee.ethz.ch/cvl/DIV2K/
