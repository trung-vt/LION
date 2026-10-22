# PCM randomized-noise pilots: completed configurations and possible scale-up

**Reviewed:** 20 October 2026. **Status:** completed engineering pilots, not converged models or paper-ready evidence.
**Workspace:** thome, /mnt/store_ext4/GIT/LION, branch `experiments_5` (reviewed commit `6305040`).
**Result root:** `/mnt/store_ext4/store/LION/ttv22/pcm/results/randomized_noise_20261014_pilot`.

This note was compiled from existing run metadata and result CSVs. It documents possible future runs only; it does **not** record new training or new reconstructions.

## 1. Data and exposure

| Item | Existing pilot |
| --- | --- |
| Image pools, train | 800 natural DIV2K and 4,000 synthetic PCM images |
| Image pools, validation | 100 natural and 500 synthetic |
| Image pools, test | 68 natural CBSD68 and 500 synthetic; **only two of each** used in the first fixed test panel |
| Input | Grayscale, 64 x 64 crops, flips/rotations; draw natural/synthetic with equal probability |
| Denoiser exposure per run | 128 updates x batch 2 = **256 crop draws from 243 distinct source files** |
| U-TV/U-TGV exposure per run | 512 updates x batch 2 = **1,024 crop draws from 861 distinct source files** |
| Training seed | 20261014 (one seed) |
| Optimizer, learning rate | Adam, 2e-4 |
| Denoiser validation | Every 32 updates, two validation images/domain (4 images) |
| Adaptive-model validation | Every 128 updates, two validation images/domain (4 images) |

The training *pool* was reasonably large for a pilot, but the models received few optimizer updates and saw a small fraction of the available files. Data augmentation/cropping does not change that limitation. Checkpoints were initialized randomly, not fine-tuned from pretrained denoisers.

## 2. All 12 main training runs

All trained at 64 x 64, batch size 2, seed 20261014 and Adam LR 2e-4. Each run has `best.pt`, `last.pt`, `run.json`, `validation.csv` and `test.csv` in `RESULT_ROOT/runs/<run>/`, replacing `RESULT_ROOT` with the path above.

| Run | Training corruption / objective | Updates | Best step | Test rows | Mean test PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| `dwtdnet_gaussian` | Gaussian denoising | 128 | 128 | 24 | 16.00 |
| `dwtdnet_mixture` | Gaussian/PCM-like surrogate denoising | 128 | 128 | 24 | 18.60 |
| `gs_dwtdnet_gaussian` | Gaussian denoising | 128 | 128 | 24 | 17.18 |
| `gs_dwtdnet_mixture` | Gaussian/PCM-like surrogate denoising | 128 | 128 | 24 | 15.92 |
| `drunet_gaussian` | Gaussian denoising | 128 | 96 | 24 | 15.83 |
| `drunet_mixture` | Gaussian/PCM-like surrogate denoising | 128 | 96 | 24 | 15.95 |
| `gs_drunet_gaussian` | Gaussian denoising | 128 | 96 | 24 | 19.65 |
| `gs_drunet_mixture` | Gaussian/PCM-like surrogate denoising | 128 | 128 | 24 | 19.58 |
| `utv_supervised_physical_pcm` | Supervised physical-PCM reconstruction | 512 | 512 | 36 | 22.12 |
| `utv_ssdu_physical_pcm` | SSDU physical-PCM reconstruction | 512 | 128 | 36 | 15.09 |
| `utgv_supervised_physical_pcm` | Supervised physical-PCM reconstruction | 512 | 512 | 36 | 21.07 |
| `utgv_ssdu_physical_pcm` | SSDU physical-PCM reconstruction | 512 | 128 | 36 | 15.10 |

**Important:** "Completed" means the requested update budget ended, not training convergence. The denoiser comparison is over image-space corruption, **not** true compressed PCM measurements. Ordinary denoisers use no Hadamard forward operator; mixture draws Gaussian or raster PCM-like surrogate noise with probability 0.5 each. The adaptive U-TV/U-TGV models instead train with the physical differential PCM simulator, a random measurement fraction in [0.01, 1], and 64 unrolled PDHG iterations. SSDU uses held-out measurement loss for both training and checkpoint selection, while supervised training uses clean targets.

Noise bounds for randomized training (`NoiseRange`): dark SD [0, 1], beam-innovation SD [0, 1e-3], bias SD [0, 4e-5], bias-gradient SD [0, 3e-7]; signal-to-bias ratio 0.3. These are simulated normalized model-unit parameters, not verified ampere-level noise calibration.

## 3. What produced the two quoted PSNR comparisons?

- **"DWTDNet 16.00 -> 18.60 dB":** arithmetic mean of *per-case PSNR* over **24** test entries = 4 distinct test crops (2 natural, 2 synthetic) x 3 fixed severities x 2 input-noise types (Gaussian and PCM-like raster surrogate). It compares the Gaussian-trained checkpoint with the mixture-trained checkpoint, not model output against noisy-input PSNR. It does not mean mixed training wins at every severity.
- **"U-TV supervised 22.12 vs SSDU 15.09 dB":** arithmetic mean over **36** test entries = 4 test crops x 3 fixed severities x 3 sampling fractions (1%, 20%, 100%), all with *simulated physical PCM* noise. The two models have different learning objectives and checkpoint-selection rules.
- PSNR is calculated separately for each prediction as `-10 log10(MSE)` on normalized image values (data range 1), then its values are averaged. These figures are **not** PSNR computed from the mean MSE.

Sources: `report_bundle/run_summary.csv`, `report_bundle/denoising_test.csv`, `report_bundle/regularizer_test.csv` and `LION/pcm/randomized/runner.py` (`evaluate`, `metrics`).

## 4. We already tested three noise severities separately

The fixed pilot evaluation uses:

| Label | Dark SD | Beam-innovation SD |
| --- | ---: | ---: |
| Low | 0.05 | 0.00012 |
| Medium | 0.20 | 0.00060 |
| Extreme | 0.80 | 0.00100 |

Other sampled noise parameters remain drawn from the defined randomized ranges. **The training did not produce a separate checkpoint for each severity.** These are separate test *conditions* for the existing checkpoints, with one noise draw per crop/condition.

### 4.1 Denoiser test PSNR by input noise and severity

Each cell averages four test crops. "G" means Gaussian-only training, "Mix" means Gaussian/PCM-like training.

| Model / training | Gaussian low | Gaussian medium | Gaussian extreme | PCM-like low | PCM-like medium | PCM-like extreme |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DWTDNet / G | 18.42 | 17.30 | 12.31 | 18.41 | 17.30 | 12.25 |
| DWTDNet / Mix | 22.92 | 20.38 | 12.59 | 22.87 | 20.27 | 12.58 |
| GS-DWTDNet / G | 23.01 | 17.98 | 10.51 | 23.07 | 18.08 | 10.43 |
| GS-DWTDNet / Mix | 18.87 | 17.63 | 11.15 | 18.90 | 17.73 | 11.25 |
| DRUNet / G | 16.49 | 16.41 | 14.54 | 16.48 | 16.39 | 14.66 |
| DRUNet / Mix | 16.80 | 16.64 | 14.34 | 16.79 | 16.61 | 14.50 |
| GS-DRUNet / G | 22.68 | 21.32 | 14.70 | 22.71 | 21.46 | 15.05 |
| GS-DRUNet / Mix | 21.35 | 20.80 | 16.40 | 21.36 | 20.86 | 16.68 |

For context, the **noisy input itself** scores 25.98 dB, 13.94 dB, 1.90 dB for PCM-like low, medium and extreme noise, respectively. Thus even the mixture-trained DWTDNet is *worse than doing nothing* at low noise in this small test panel. Do not interpret its 18.60 dB pooled score as proof of useful denoising across the range.

### 4.2 U-TV/U-TGV test PSNR at 20% sampling, physical PCM

Each cell averages four test crops; there are also stored results at 1% and 100% sampling.

| Reconstruction model | Low | Medium | Extreme |
| --- | ---: | ---: | ---: |
| Zero-filled adjoint | 22.29 | 15.72 | 5.65 |
| U-TV, supervised | 24.38 | 23.24 | 18.80 |
| U-TV, SSDU | 22.42 | 15.77 | 5.66 |
| U-TGV, supervised | 25.27 | 21.68 | 14.46 |
| U-TGV, SSDU | 22.45 | 15.77 | 5.66 |

The original all-rate average (22.12 vs 15.09 dB) hides this severe noise dependence. SSDU is approximately equal to zero-filled here at medium/extreme noise. That is not evidence that SSDU fundamentally fails; only that this pilot training/configuration did poorly.

## 5. More saved evaluations, no GPU rerun needed

### 5.1 Matched test acquisitions at 20% sampling

Four ordinary held-out test crops, with the same PCM acquisitions shared between methods. Dark SD fixed at 0.05 or 0.20, beam-innovation SD fixed at 0.00060. 8 methods x 4 crops x 3 sampling rates (1%, 20%, 100%) x 2 noise levels = **192 matched results**. PnP methods had validation-selected parameters with a 20-iteration budget; adaptive U-TV/TGV used 64 PDHG steps, so runtimes/iterations are not resource matched.

| Method | Dark SD 0.05 | Dark SD 0.20 |
| --- | ---: | ---: |
| Zero-filled adjoint | 24.44 | 16.80 |
| U-TV supervised | 25.34 | 24.02 |
| U-TV SSDU | 24.79 | 16.85 |
| U-TGV supervised | 27.22 | 23.33 |
| U-TGV SSDU | 24.82 | 16.85 |
| PnP-ADMM + DWTDNet | 20.38 | 18.88 |
| PnP-ADMM + DRUNet | 18.04 | 16.16 |
| GS-PnP + GS-DWTDNet | 24.47 | 16.80 |
| GS-PnP + GS-DRUNet | 24.50 | 19.38 |

Source: `report_bundle/matched_test.csv`. The underlying `pnp_test.csv` contains 96 PnP reconstructions; matching the four U-TV/TGV models adds 96 more.

### 5.2 Separate 256 x 256 unseen-NPL-cell benchmark at 20% sampling

Two previously unseen **fully measured** maps (CIGS and a perovskite-centre crop). Undersampled/noisy acquisitions were **simulated** from those maps, NOT measured as independently compressed data. One simulated noise realization per case. Other sampling fractions tested: 10% and 50%. This benchmark stores 216 reconstructions = 2 maps x 3 rates x 4 scenarios x 9 methods.

| Method | Noiseless | Gaussian dark | NPL-like default 10 ms | Short 100 us stress |
| --- | ---: | ---: | ---: | ---: |
| Zero-filled | 27.73 | 26.20 | 26.10 | 16.23 |
| Scalar TV (fixed lambda 0.02) | 32.13 | 30.87 | 30.72 | 16.02 |
| SPGL1 | 30.20 | 25.45 | 25.31 | 12.68 |
| U-TV supervised | 22.84 | 22.88 | 22.89 | 22.65 |
| U-TV SSDU | 30.27 | 26.98 | 26.83 | 16.00 |
| U-TGV supervised | 27.21 | 27.11 | 27.10 | 24.28 |
| U-TGV SSDU | 30.38 | 27.03 | 26.87 | 16.01 |
| PnP-ADMM + DWTDNet | 23.38 | 23.29 | 23.29 | 21.29 |
| GS-PnP + DRUNet | 25.85 | 25.46 | 25.44 | 19.89 |

Source: `unseen_npl_fixed_20261014_v1/reconstruction_metrics.csv`; protocol `scripts/experiments/pcm/README_fixed_unseen_npl.md`. Noise is NPL-inspired and normalized, **not calibrated** to the measured currents of these two new cells. Avoid generalization claims based on two specimens.

### 5.3 Additional SSDU diagnostic training

Six further completed **192-update** runs exist under `ssdu_diagnostics_fixed_20261014/`: U-TV SSDU and U-TGV SSDU, each under the three regimes `noiseless`, `gaussian_dark`, and `pcm_default_10ms`. These use 64 x 64, batch 2, 64 unrolled iterations, Adam LR 2e-4, seed 20261014, validation every 64 updates and four validation images per domain. They are short diagnostics, not fully trained final checkpoints.

## 6. Proposed next training configuration (NOT RUN)

Change **training duration first** while holding data protocol, learning rate, noise distributions and architecture fixed, so we can attribute any improvement to longer training. Budgets are starting points for controlled comparisons, not guarantees of convergence.

| Axis | Current denoisers | Proposed next denoisers | Current U-TV/U-TGV | Proposed next adaptive models |
| --- | --- | --- | --- | --- |
| Optimizer updates | 128 | **10,000 first**, then 30,000-50,000 if validation keeps improving | 512 | **2,000 first**, then 5,000-10,000 if validation keeps improving |
| Batch | 2 | Keep 2 initially; test effective 4-8 later | 2 | Keep 2 initially; increase by accumulation only if useful |
| Training crop | 64 x 64 | Keep 64 initially, compare 128 later | 64 x 64 | Keep 64 first; compare 128 only after checking memory |
| Total crop draws at first longer budget | 256 | 20,000 at 10k updates x batch 2 | 1,024 | 4,000 at 2k updates x batch 2 |
| Validation images | 2 natural + 2 synthetic | At least 20 per domain, fixed separate panel | 2 natural + 2 synthetic | At least 20 per domain, fixed separate panel |
| Independent training seeds | 1 | 3 if feasible | 1 | 3 if feasible |
| Test noise repetitions | 1 per source/condition | At least 3 independent realizations per case | 1 per source/condition | At least 3 independent realizations per case |
| Test panel | 4 crops | At least 20 per domain, both Gaussian and PCM-like, each severity separately | 4 crops | At least 20 per domain, stratified by sampling and physical PCM severity |
| Training noise | Gaussian-only or 50/50 Gaussian + raster PCM-like | Keep both controls first; later compare better calibrated image-domain corruption | Full randomized physical PCM | Keep current first; then assess NPL-like default/short calibrated regimes separately |
| Unrolled PDHG | N/A | N/A | 64 iterations per optimizer update | Keep 64 first; vary only after longer training |
| Checkpoint selection | Clean validation MSE | Validation curve and held-out selection, no test-driven tuning | Supervised clean MSE / SSDU held-out measurement MSE | Preserve distinct valid selection criteria; record both diagnostic metrics |

**Cost interpretation:** More optimizer updates increases the number of passes/training draws; it does **not** necessarily make each optimizer update slower. Increasing crop side or PDHG unrolled iterations makes each update slower and increases memory demand. More noise realizations/test crops increases inference and evaluation cost. For a clean attribution, do not vary all these axes together.

Before treating denoisers as "realistic PCM noise trained", explicitly decide whether the denoising target is the raster image surrogate or the noise distribution induced by a PCM measurement/reconstruction operator; the current denoisers use only the former. The physical noise in U-TV/U-TGV is not an independent real-world calibration.

No further low/medium/extreme tests are necessary just to **obtain a severity breakdown**; those already exist. Future tests are mainly needed for a larger sample panel, repeated noise draws, fixed NPL-like regimes, and uncertainty estimates.

## 7. Traceable source locations

- Code: `LION/pcm/randomized/runner.py` (TrainingConfig, `evaluate`, `metrics`, `train`); `LION/pcm/randomized/noise.py` (NoiseRange, image_corruption, randomized_acquisition).
- Original protocol: `scripts/experiments/pcm/README_randomized_noise.md`. It states: "the 2026-10-14 runs are bounded pilots, not converged models or final paper evidence."
- Fixed NPL benchmark protocol: `scripts/experiments/pcm/README_fixed_unseen_npl.md`.
- Individual config and optimizer states: `RESULT_ROOT/runs/<run>/run.json`, `best.pt` and `last.pt`.
- Aggregates: `RESULT_ROOT/report_bundle/run_summary.csv`, `denoising_test.csv`, `regularizer_test.csv`, `pnp_test.csv` and `matched_test.csv`.
- Existing PDF (already generated): `RESULT_ROOT/report_bundle/report.pdf` and `slides.pdf`.
- Separate unseen-cell results: `RESULT_ROOT/unseen_npl_fixed_20261014_v1/reconstruction_metrics.csv`.

All tables display arithmetic means of the stored per-case PSNR entries, rounded to 2 decimals. Some reported table rows contain only four distinct source images (or two unseen cells); no confidence intervals are claimed.
