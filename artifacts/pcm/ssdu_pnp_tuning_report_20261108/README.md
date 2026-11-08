# SSDU PnP parameter tuning report (8 November 2026)

This directory is the canonical, editable experiment report on tuni.

## Current status

- Parameter searches complete: 15/15 PnP-ADMM candidates and 18/18 GS-PnP candidates, at both five and 50 validation images.
- Four separately frozen selection records, no failed candidates.
- The four choices happen to use the same parameters for five and 50 images, but their scores and provenance are stored independently.
- Evaluation on the 450-image test set is **not yet included**. Do not describe the report as the final test-set comparison.

## Files

- `SSDU_PnP_Tuning_Report.tex`: editable LaTeX, sections 1-6.
- `SSDU_PnP_Tuning_Report.pdf`: compiled 5-page report.
- `figures/candidate_losses.pdf`: complete search-grid measurements, both tuning sizes.
- `figures/category_losses.pdf`: held-out SSDU MSE by category.
- `regenerate_charts.py`: rebuilds vector figures from frozen candidate JSON records, without ground-truth access.
- `report_manifest.json`: source hashes, version and selections.

## Edit and recompile on tuni

    cd /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/report
    /home/t/micromamba/envs/lion_3/bin/python -B regenerate_charts.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error SSDU_PnP_Tuning_Report.tex

After the planned 450-image comparison, update the results and conclusion sections and save a new version. The original frozen tuning selections must not be altered after test-set inspection.
