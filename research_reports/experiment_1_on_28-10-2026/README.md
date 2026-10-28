# Experiment 1 on 28-10-2026

**Learned Post-Training Calibration Scaling of Spatial Regularisation Maps**

Private research reports from the LION PCM U-TV experiment, completed
28 October 2026 on tuni using the September 2026 U-TV checkpoint.

- `Experiment_1_on_28-10-2026_Exploratory_Sweep_Report.pdf`:
  41-page Stage I exploratory four-image scale sweep.
- `Experiment_1_on_28-10-2026_Full_Report.pdf`:
  38-page final report, with 50-image Approach A scale-head training,
  25-image validation and 4-CIGS-image evaluation.
- Matching `.tex` files: LaTeX sources.
- Matching ZIPs: source, PDF, figures and experiment data for reproduction.

The four test images were previously examined in Stage I; the final comparison
is independent of scale-head fitting but is not a completely untouched-image
evaluation. These reports are intended for private research use.

## Key result

With the September checkpoint, a validated fixed multiplier `c=8`
improved the four-CIGS-image PSNR over `c=1` by about 4 dB; the
selected near-optimal-set regression head did not beat this fixed multiplier
on the held-out CIGS tests. Global quality improvements can suppress narrow
dark lines, so inspect the feature-preservation figures as well.

The experimental source files and original results remain on tuni at:
`/store/LION/ttv22/pcm/results/experiment_1_on_28-10-2026/`.
