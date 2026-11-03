# PCM U-TV comparison: fixed 20% sampling, NPL default noise

This is a **new experiment**, not a reuse of the variable-sampling and variable-noise test. It runs the full 450-image held-out synthetic test split at **exactly 20%** multilevel Hadamard sampling (J=8, coarse J=6) and the **fixed NPL default** noise parameters.

The relevant source is `benchmark.py` and `plots.py`. The actual PCM reconstruction library is pinned to Git revision `9188b0735843c96526537b6b9bfd834fd089f7cd`, because the currently checked-out LION code has subsequently changed its model-loading API. All original model checkpoints remain unmodified.

## Protocol and data

The test split uses permutation positions **911-1000**, giving 90 test images in each category (CIGS, Organic, Perovskite, Reference, Silicone). The source images are 256 x 256 grayscale PNGs, mapped to [0,1] by dividing by 255.

The sampling ratio is always 0.20 and the pattern family is always multilevel Hadamard, with coarse J=6. The chosen rows vary deterministically by image; the *percentage* does not. Each image has one fresh simulated noisy measurement. Every model receives the **exact same measurement tensor**, operator and ground truth for that image.

Noise parameters are fixed, in normalized model units:

| Parameter | Value |
|---|---:|
| beam_sigma | 0.00012 |
| bias_sigma | 0.000029 |
| bias_gradient_sigma | 0.00000016 |
| dark_sigma | 0.040290178571428574 |

Only the stochastic *noise realisation*, not its parameters, changes by image. The simulator retains its NPL default temporal correlations and plus/minus split measurement convention.

## Methods

All seven methods run for all 450 test images: zero-filled, original SPGL1 basis pursuit (not BPDN), DeepInv GS-PnP with GS-DRUNet, U-TV A epoch 38, U-TV B epoch 44, U-TV C epoch 60, and the old no-noise-trained U-TV epoch 99.

The original SPGL1 preset uses db4, data factor 1e5, up to 100 solver iterations and 10 debias iterations. The existing GS-PnP preset uses sigma 0.05, lambda 0.1 and at most 50 iterations, with its existing stopping and backtracking behaviour. No method is tuned on the test results.

Each chosen reconstruction is scored using LION's original MSE, PSNR, SSIM and HaarPSI. Display-only image clipping to [0,1] does not change the metrics.

## Reproduce on tuni

Use the pinned LION source tree, with the new benchmark script from the fork:

```bash
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export LION_TTV22_PATH=/store/LION/ttv22
export CUDA_VISIBLE_DEVICES=GPU-05267a12-f71f-101d-59e5-05e5c430d8ef
export PYTHONPATH=/store/LION/ttv22/pcm/results/pcm/pcm_utv_three_full_retest_20261103/source_9188b07
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
cd /store/LION/ttv22/pcm/results/pcm/pcm_utv_three_full_retest_20261103/source_9188b07

SCRIPT=/home/t/Documents/GIT/LION/scripts/experiments/pcm/fixed_utv_test_20261103/benchmark.py
python "$SCRIPT" --stage prepare
python "$SCRIPT" --stage run --category CIGS --start 0 --end 15
```

Repeat the disjoint 15-image chunks (0:15, 15:30, 30:45, 45:60, 60:75, 75:90) for each of the five categories. A one-image smoke test may instead use 0:1 and remaining chunk 1:15. The script refuses to overwrite completed results. Do not run two jobs concurrently on the RTX 4070.

After all 450 rows and 50 selected reconstruction NPZ files are present:

```bash
python /home/t/Documents/GIT/LION/scripts/experiments/pcm/fixed_utv_test_20261103/plots.py
```

The experiment output is kept **outside the Git repository**:

`/store/LION/ttv22/pcm/results/pcm/pcm_utv_fixed20_npl_default_20261103/`

Outputs include the protocol manifest with config/checkpoint hashes, 450 result rows, 50 selected image arrays, a category-by-category boxplot PDF and a 50-image comparison PDF.

## Interpretation and limitations

A fixed sampling ratio and fixed noise parameters do not mean an identical sampling mask or identical noise trace on every image. Different images receive independent deterministic draws. The experiment has one noise realisation per image, not repeated-noise error bars.

Comparisons of U-TV C and old U-TV do **not** isolate training with noise: the two models also have different U-Net sizes, PDHG unrolling lengths, training sampling schemes and epochs. These results are limited to held-out *synthetic* PCM images and simulated noisy acquisitions.

The earlier test was made under varying conditions and should not be cited as the fixed 20% result. Its results are preserved separately; this new experiment does not read or merge them.
