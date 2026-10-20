# PCM PnP noise robustness experiments (20 October 2026)

## Context and scope

Branch: `pcm-noise-admm-sweep-20261020`, based on integrated upstream LION commit `c67359b`.

This is a **pilot**, not the full PCM-paper benchmark. No packages were installed.
Both tuni GPUs were used (GPU 0 RTX 4070, GPU 1 RTX 4090).
Runs and checkpoints are stored outside Git at:

`/store/LION/ttv22/pcm/results/pcm_admm_noise_study_20261020/`

### Main questions

1. Is the catastrophic simulated-noise degradation of PnP-ADMM due largely to parameter tuning?
2. Does diagonal weighted least squares (WLS) improve over ordinary least squares (OLS)?
3. Does training denoisers on PCM rather than Gaussian corruption improve held-out denoising and ultimately PnP?

## Experimental design

- Normalized Hadamard PCM, 256 x 256 synthetic images, sampling rate 20%, multilevel coarse J=6.
- NPL-default PCM simulated noise; fixed sampling for both OLS and WLS and for different noise realizations.
- Validation: three images (CIGS, Organic, Silicone) from permutation positions 801 onward using seed 5678. Search over rho, sigma, and fixed stopping iteration.
- Held-out reconstruction: five image categories with one sample/category from permutation positions 991-995, two independent noise realizations (10 reconstructions per condition).
- No test-ground-truth-based tuning. Metrics in the sweep use ground truth to rank **validation** conditions only. Fixed 50 iterations in the held-out test.
- Training: distinct samples in positions 1-800, validation positions 801-900, held-out denoising evaluation positions 901-904; two small residual CNNs trained for three epochs.

IMPORTANT: The 3 validation images constitute a very small tuning set. This is a proof-of-concept, not sufficient for final paper claims.

## Parameter search

Original common setting: `rho=0.01, sigma=0.05`; 50 ADMM iterations.
The search covered:

- coarse: rho 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5, 1.0; sigma 0.01, 0.025, 0.05, 0.1; 30 iterations.
- refine: rho 0.3, 0.5, 0.8, 1.5, 3.0; sigma 0.07, 0.1, 0.15, 0.2; 70 iterations.

Best validation mean (OLS): `rho=0.5, sigma=0.1, iteration=65`, PSNR **38.0358 dB**. Performance was essentially stationary between 30 and 70 iterations.

### Held-out reconstruction results

| Method | rho | sigma | iterations | Mean PSNR over 10 runs | Minimum PSNR |
|---|---:|---:|---:|---:|---:|
| zero-filled | n/a | n/a | 0 | 24.831 dB | 21.293 dB |
| OLS PnP-ADMM | 0.01 | 0.05 | 50 | 12.314 dB | 8.359 dB |
| WLS PnP-ADMM | 0.01 | 0.05 | 50 | 12.395 dB | 8.420 dB |
| OLS PnP-ADMM | 0.01 | 0.10 | 50 | 17.884 dB | 14.325 dB |
| WLS PnP-ADMM | 0.01 | 0.10 | 50 | 17.972 dB | 14.396 dB |
| OLS PnP-ADMM | 0.5 | 0.05 | 50 | 33.217 dB | 25.434 dB |
| WLS PnP-ADMM | 0.5 | 0.05 | 50 | 33.222 dB | 25.427 dB |
| **OLS PnP-ADMM** | **0.5** | **0.10** | **50** | **36.302 dB** | **31.471 dB** |
| WLS PnP-ADMM | 0.5 | 0.10 | 50 | 36.301 dB | 31.469 dB |

- Original OLS setting beat zero-filled for **0/10** held-out measurements (mean -12.517 dB).
- Tuned OLS setting beat zero-filled for **10/10** held-out measurements (mean +11.471 dB, minimum improvement +9.492 dB).
- Tuned WLS was very close to tuned OLS (mean -0.001 dB). **No evidence in this pilot that diagonal WLS helps.**

Do not compare these absolute PSNR values directly against the earlier E27/CIGS runs with different sample/mask/noise seeds. The previous CIGS October 6 run had: zero-filled 22.349 dB, noisy original PnP-ADMM 6.405 dB, noisy GS-PnP 28.663 dB; these are contextual, not matched tests.

### WLS definition and limitations

Weighted fidelity is `0.5 * sum_i w_i (Ax-y)_i^2`, with a closed-form Hadamard-domain proximal update `alpha_i = (w_i y_i + rho q_i)/(w_i + rho)` on observed coefficients. Unobserved coefficients have weight zero. Scale-invariant normalized weights are estimated from the noise simulator conditional on an observed-data-derived estimate (zero-filled reconstruction), **not** from ground truth.

This is **diagonal WLS**, not full generalized least squares (GLS); it discards measurement-to-measurement correlations in beam drift. Estimated weights were clipped to [0.1,10]. A covariance-aware formulation and comparing alternative weight estimators remain future work. Check the conditioning and variance estimator against the exact noise assumptions before claiming physical optimality.

## New compact denoiser models (pilot)

Three epochs, 120 synthetic training images (24/category), 20 validation images (4/category), Adam, compact five-convolution residual CNN, `sigma=0.05`. One model trained with Gaussian pixel noise, the other with reconstructed full-pattern PCM simulated noise. This latter training corruption includes artifacts from full-sampling PCM; it is **not** the same distribution as a PnP iterate at 20% subsampling.

Held-out denoising (20 images, distinct from train and validation):

| Corruption | Input PSNR | Gaussian-trained CNN | PCM-trained CNN |
|---|---:|---:|---:|
| Gaussian | 26.0215 | 32.6159 | **33.1197** |
| PCM simulated + full Hadamard inverse | 24.1431 | 31.1166 | **31.9038** |

The PCM-trained pilot was also tried in PnP-ADMM on the 3 validation images: with `rho=1.0, sigma=0.05`, at iteration 50, it reached **30.759 dB** versus **30.402 dB** for the Gaussian-trained pilot. These lightweight models did **not** outperform the tuned off-the-shelf DRUNet validation score of 38.036 dB; they are not yet publication-grade replacements.

## Remaining checks for the PCM paper

1. Repeat validated settings on larger disjoint test sets (500 synthetic images) and multiple acquisition noise seeds.
2. Compare matched GS-PnP and SPGL1/TV/U-TV baselines using identical masks, noise, and data partitions.
3. Verify the simulator's differential-noise covariance and replace diagonal WLS with covariance-aware GLS or a validated marginal variance model.
4. Check more aperture settings (10 ms, 5 ms, 1 ms, 100 us) with trustworthy measured parameters, not guessed numbers.
5. Train full DRUNet/GS-DRUNet on matched PCM and Gaussian noise, validate denoiser and final PnP reconstructions.
6. Report HaarPSI, SSIM, PSNR, image quality, runtime per iteration and total convergence time. For the paper, reserve final test images until models and hyperparameters are frozen.

### Reproducing the current pilot

Run with `LION_DATA_PATH=/store/LION/ttv22/pcm/data`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pcm/pcm_admm_noise_study.py --phase coarse --fidelities ols wls --rhos 0.001 0.003 0.01 0.03 0.1 --sigmas 0.01 0.025 0.05 0.1 --iters 30 --eval-every 2 --case-limit 3
CUDA_VISIBLE_DEVICES=1 python scripts/pcm/pcm_train_noise_denoiser.py --mode pcm --epochs 3 --train-per-category 24 --val-per-category 4
CUDA_VISIBLE_DEVICES=0 python scripts/pcm/pcm_evaluate_noise_denoisers.py
```

Script references: `LION/pcm/simulation.py`, `LION/pcm/noise.py`, `LION/pcm_tools/pcm_prox.py`, `LION/reconstructors/PnPADMM.py`. Draft paper experiment scope: `main_ieee.tex` around lines 620-750 of the `paper` workspace. Differential-noise variance derivation: `fidelity.tex` in the paper workspace.
