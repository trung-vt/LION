# PCM Tune-1 with bounded ternary search

Status on 9 November 2026: PREPARED. No GPU tune-1 parameter search started.

## Data and naming

test set 450 = original held-out PCM set, 90 images in each of five categories.
test set 45 = immutable random subset, nine images per category.
Validation calibrations = five independently selected noisy validation
acquisitions per category (25 calibration acquisitions total).

PnP-ADMM and GS-PnP each perform 25 independent tune-1 selections.
For each category, the five parameter sets are applied to the same nine
test45 images. Each method therefore requires 225 test reconstructions.

Selections and SHA256 identities are in:
- prepared/validation_calibrations.json
- prepared/test45.json
- prepared/ranges.json
- prepared/protocol.json

## Search

Bounded log-coordinate ternary, two passes with reversed coordinate order
and three ternary contractions per coordinate.

ADMM: rho [0.03, 4.0], denoiser sigma [0.015, 0.18].
GS-PnP: lambda [0.005, 0.40], sigma [0.015, 0.15],
initial tau0 [3.0, 75.0].

All four original SSDU folds are evaluated for each candidate. The 1D
unimodality assumption does not imply global minimisation of the joint
2D/3D PnP parameter objective; backtracking can cause discontinuities.
Invalid numerical candidates are recorded explicitly.

Upper budgets:
 ADMM 25 tune1 selections * 25 candidates * 4 folds = 2,500 solver runs.
 GS 25 tune1 selections * 37 candidates * 4 folds = 3,700 solver runs.

## Executing on tuni

Worktree: /home/t/Documents/GIT/LION_pcm_tune1_ternary_20261109
Results: /store/LION/ttv22/pcm/results/pcm/tune1_ternary_20261109_v1

Read-only status:
    cd /home/t/Documents/GIT/LION_pcm_tune1_ternary_20261109
    bash scripts/experiments/pcm/tune1_ternary_20261109/run_all.sh status

Numerical execution ONLY after an explicit later instruction:
    bash scripts/experiments/pcm/tune1_ternary_20261109/run_all.sh execute

The execute workflow runs independent validation-only ADMM tune-1 searches
on RTX 4070 and GS-PnP tune-1 searches on RTX 4090. Only after all 50
selections are frozen can the test45 runners open test images and metrics.

The final stage reruns missing historical baseline reconstructions, validates
their metrics and archives, writes per-image and calibration-average metrics,
creates category plots, composes 45 pages of complete visual comparisons
(target, 10 old method labels, 10 tune1 reconstructions per page), and
refreshes the report.

## Results interpretation

Five calibration choices sharing the same nine test cases are NOT
45 independent test images per category. Calibration means and SDs are
reported separately. The overall tune1 score averages five calibration
outcomes for each unique test image before averaging all 45 test cases.

The existing 450-image test set remains a separate benchmark and is not
used for tuning. All images use original 20%-sampling NPL-default noise,
normalised J=8 Hadamard operator, coarse_J=6, no changes to data or
pretrained PnP weights.

Original literature:
- Yaman et al. SSDU (2020): https://doi.org/10.1002/mrm.28378
- Berkeley unimodal interval optimisation: https://stat243.berkeley.edu/fall-2024/units/unit11-optim.html
- Hurault et al. GS-PnP (2022): https://openreview.net/pdf?id=fPhKeld3Okz
