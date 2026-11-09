# Expanded GS-PnP SSDU five-image tuning: completed experiment

**Status: all 27 candidates accounted for (24 valid, 3 invalid); winner
frozen before test; all 450 matched test reconstructions complete; report v2.**

## Hyperparameter selection and validity

The same five validation images and original four SSDU folds were reused,
without ground-truth images or test metrics for parameter selection.

The original gs_exp_001 (lambda=0.10, sigma=0.075, tau0=100/3) had SSDU
held-out MSE 0.004365433883375238. The expanded-grid winner is gs_exp_002
(lambda=0.10, sigma=0.075, tau0=50), SSDU MSE 0.004361004397883536,
a 0.1015% decrease. Three candidates were numerically invalid due to
GS-PnP failing to accept an update on at least one validation fold:
gs_exp_021, gs_exp_024, gs_exp_027. They were excluded by the
predeclared failure policy.

The 450-image comparison uses exact previously archived 20%-sampling,
NPL-default noise and pattern SHA256 fingerprints, with frozen parameters.

## Important evaluation finding

- Expanded GS-PnP mean PSNR: 32.5632 dB over 450 images
- Original GS-PnP SSDU-5: 32.5903 dB
- Supervised-validation-selected GS-PnP Step 6: 35.1098 dB
- Expanded minus original mean paired PSNR: -0.02713 dB
- Paired median PSNR: +0.12689 dB
- Expanded PSNR higher in 349/450 cases, but 43 cases lost >1 dB.
- A separate 8-case hardware-parity replay on RTX4090 matched old
  original RTX4070 PSNR to within 0.0001 dB in those cases.

**Lower held-out SSDU loss did not improve overall image PSNR.**
The 450-image evaluation is retrospective and must not be used for further
unrecorded hyperparameter selection. PCM noise can be correlated, and
unmeasured Hadamard directions can contain thin details.

## Canonical editable report

- report/GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex
- report/GS_PnP_Expanded_SSDU_Tuning_Full_Report.pdf
- report/GS_PnP_Expanded_SSDU_Tuning_Report.tex and PDF (latest copies)
- report/pretest_prepared_v1/ (original pre-GPU preparation report)
- report/figures/ (SSDU candidate loss and parameter-slice plots)
- test450/analysis/figures/ (10-method boxplots, category plots and five
  10-method visual comparisons with raw-image PSNR/SSIM/HaarPSI/MSE)

## Reproducibility sources and raw records

- frozen_selection.json: immutable final validation-only selection
- candidate_grid.json, prepared/manifest.json: fixed grid/inputs
- results/candidates/: all per-image per-fold JSON scores and diagnostics
- test450/records/: 450 reconstructed image metrics, parameter/hash provenance
- test450/raw/: 450 full-resolution float32 npy reconstructions, retained
  on tuni, NOT included in the downloadable ZIP to keep it small
- test450/same_gpu_probe.json: parity replay on 8 predeclared diagnostic cases
- test450/analysis/: merged CSV and JSON metrics, runtimes, paired differences
- logs/: numerical sweep and freeze logs
- Git branch ssdu-pnp-tuning-20261108: scripts/experiments/pcm/ssdu_pnp_tuning_20261108

To compile the final report on tuni:

    cd /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/next_gs_ssdu_sweep_v1/report
    latexmk -pdf -interaction=nonstopmode -halt-on-error GS_PnP_Expanded_SSDU_Tuning_Full_Report.tex

The pretest report and the older SSDU tuned-5, tuned-50 and
450-image benchmark experiments remain preserved.
