# Step 6: OLS versus oracle WLS validation on the same PCM acquisitions

This step compares two existing SPGL1-BPDN methods on exactly the same
50 frozen noisy validation acquisitions. It does not create any new noisy
measurement, alter ground-truth images, tune parameters, or touch the
450-image held-out test split.

## Experiment rules

- Fixed 20% dyadically ordered Hadamard mask, M=13,107, per-image frozen A,y.
- WLS uses Step 3 oracle marginal inverse variances D=diag(1/sqrt(v)).
- OLS is unweighted. The only variations are D in forward and adjoint
  and each method's separately calibrated Step 5 95% noise residual bound.
- Both use the original Wavelet2D db4, periodization and SPGL1 solver.
- Identical max 120 iterations, opt_tol=1e-3, scaling factor 1.0.
- Debiasing disabled for both. Solver float32; master data/weights float64.
- First two validation images per category: ten-image pilot.
  Those ten are reused in the final full 50-pair comparison.
- Order of solving OLS/WLS alternates by permutation position for timing
  fairness. Record status, feasibility, time, iterations and seconds per
  iteration; solver failures must not be silently filtered.
- RTX 4070 was free, but SPGL1/PyWavelets run on CPU; this experiment used
  CPU and did not interrupt 4090 UTV training or occupy either GPU.

## Mathematical formulation

With B=A Psi^-1, original image-space observation y and diagonal
D_mm=1/sqrt(v_m), solve

OLS: min ||z||_1 subject to ||B z-y||_2 <= epsilon_OLS,
WLS: min ||z||_1 subject to ||D B z-D y||_2 <= epsilon_WLS.

Weighted adjoint: (D B)^* r = B^* D r.
The db4 wavelet, y and A must be identical across methods.

## Image quality

Use the original LION.utils.metrics.compute_metrics_2d function,
with data_range=1.0. MSE, PSNR and SSIM operate on raw float32
reconstructions without clipping. HaarPSI clips the reconstruction
to [0,1] internally. Ground truth comes from the original grayscale
PNG converted to float64 and divided by 255.

Qualitative fine-detail observations are exploratory; there is
no validated thin-line mask for quantitative F1. Do not describe
ordinary HaarPSI as thin-line recall.

## Speed and paired statistics

Each runtime includes the solver and setup in solve_step5_case.
Seconds per iteration = measured time / SPGL1 iterations, so
differences can arise from iteration counts and/or per-iteration costs.
Single short CPU runtime differences are sensitive to shared-machine
scheduling; do not generalise them to GPU speed.

For every metric report WLS minus OLS per image, mean, sample standard
deviation, median and WLS win rate. Use 10,000 fixed-seed paired
bootstrap samples for an approximate 95% mean-difference interval.
A positive PSNR/SSIM/HaarPSI or negative MSE/runtime difference
favours WLS. The validation set is not the final held-out test.

## Frozen input manifests and Step 6 output

Step 2 acquired A,y SHA256:
b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

Step 3 oracle weights SHA256:
50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1

Step 5 residual calibration SHA256:
fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a

Step 6 output on tuni:
/store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/step6_ols_vs_oracle_wls_validation_v1/

Step 6 independently pinned output manifest SHA256:
8e71d0977db94fde758895db24e193517aaec28d2a040e8d5ca55a2436e1c2e1

It includes 100 reconstructed float32 PNG-independent arrays in NPZ
containers, the 50-pair per_image_metrics.csv, pilot_summary.json,
summary.json, manifest.json and SHA256SUMS.

## Code and verification

- compare_ols_wls_step6.py: run pilot plus full validation, store both
  methods and paired metrics, preserve failures, fixed bootstrap statistics.
- verify_step6.py: read-only hash check of all 100 arrays, recalculate
  all 100 image quality metrics against original PNGs, verify 50/50
  same y/A, stages, category coverage and paired summaries.
- pytests/pcm/test_oracle_wls_step6.py: synthetic 256x256 end-to-end,
  paired bootstrap, WLS win-count logic, metric conventions and
  trust anchors.

Original SPGL1 and Wavelet2D sources and prior archives are unchanged.
The 4090 training remains independent. The free 4070 is not needed by
the present CPU-native solver adapter.

References: original LION/utils/metrics.py,
LION/pcm/spgl1_bpdn.py, LION/operators/Wavelet2D.py and Step 5 files.
SPGL1 documentation: https://spgl1.readthedocs.io/.
