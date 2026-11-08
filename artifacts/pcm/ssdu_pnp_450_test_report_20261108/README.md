# PCM SSDU PnP 450-case comparison - report version 2

Completed 8 November 2026. This version updates the original tuning-only report with the frozen 450-image matched test comparison across 11 labelled methods.

## Editable report

- `SSDU_PnP_Tuning_Report.tex`: latest editable LaTeX with 450-image results.
- `SSDU_PnP_Tuning_Report.pdf`: compiled eight-page latest report.
- `archive_tuning_only_v1/`: preserved tuning-only original TeX and PDF.
- `figures/`: original four-fold SSDU tuning curves.

## Separate test output

`../test450/v1/` includes the complete 450-case CSV and summary JSON, detailed timing, paired PSNR differences, numerical reconstructions and diagnostics for both new GS-PnP variants, and four comparison figures. The 14-page sample-image gallery and five-page category comparison are saved there.

## Method identity and test-data integrity

All 450 test measurement and pattern SHA256 hashes agree with archived matched 20% PCM acquisitions. Existing numerical baselines were reused after verification. The PnP-ADMM SSDU-5/50 selections equal the original validation-selected PnP-ADMM oracle exactly; the existing 450-test reconstructions are reused for all three labels. GS-PnP SSDU-5/50 selections also coincide and share one new 450-test run. A separate frozen GS-PnP Step6 oracle was newly reconstructed using paper-final-z output. No test-driven hyperparameter selection occurred.

## Recompile

```sh
cd /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/report
latexmk -pdf -interaction=nonstopmode -halt-on-error SSDU_PnP_Tuning_Report.tex
```

The LaTeX refers to `../test450/v1/figures/` and requires those companion PDFs to remain beside it in the current directory structure. Preserve the frozen selections and version 1 when making later revisions.
