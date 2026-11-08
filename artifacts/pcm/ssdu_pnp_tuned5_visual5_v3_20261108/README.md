# SSDU PnP PCM report v3 - 8 November 2026

Main comparison: nine method labels = seven original baselines plus PnP-ADMM-tuned-5 and GS-PnP-tuned-5. The old 11-label v2 results are preserved, not discarded. Tuned-50 selected exactly the same configurations as tuned-5; both frozen 50-image records remain on tuni.

## Canonical editable version

- `SSDU_PnP_Tuning_Report.tex` and `SSDU_PnP_Tuning_Report.pdf`: latest version 3
- `SSDU_PnP_Tuning_Report_v3.tex/.pdf`: explicit version 3 filenames
- `archive_tuning_only_v1/`: v1 tuning-only TeX/PDF
- `archive_test450_11methods_v2/`: v2 full 11-label TeX/PDF

## Five visual cases

`../test450/v1/visual5/PCM_5_Cases_9_Methods_with_Metrics.pdf` (five pages) shows target and all 9 method labels with PSNR, SSIM, HaarPSI and MSE for CIGS 0279, Organic 0331, Perovskite 0786, Reference 0279 and Silicone 0370. The four non-CIGS cases were sampled reproducibly from the already randomly selected 10 test visualisation cases per category. Image intensity: [0,1]; colormap: AFM-hot capped at 0.8. All masks and measurements match the frozen 450-image benchmark.

## Expanded GS-PnP SSDU sweep

`../next_gs_ssdu_sweep_v1/PLAN.md` and `candidate_grid.json` define 27 validation-only configurations with lambda=0.1/0.2/0.4, sigma=0.075/0.10/0.15, tau0=33.33/50/75. This new sweep has NOT been executed. Do not select on test-image scores.

## Compile on tuni

    cd /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/report
    latexmk -pdf -interaction=nonstopmode -halt-on-error SSDU_PnP_Tuning_Report.tex

When working from the self-contained ZIP, `cd report` and run the same latexmk command. Preserve the relative `test450/v1/visual5/` PDF assets. Frozen parameter selection and 450-image data are unchanged.
