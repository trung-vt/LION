# Step 6 independent reviewer follow-up (2 November 2026)

Reviewed inputs: external detailed Step 6 review, plus 12 reviewer Python tests.

## Confirmed findings and corrections

1. The historical paired PSNR figure used pilot-first run order but
   incorrectly labelled it as five consecutive ten-case category groups.
   Exactly 28 of 50 bar labels were in incorrect categories. The underlying
   recorded image-level results and category means were correct. A new
   deterministic plotting script explicitly sorts actual category and
   permutation position, then generates four versioned, corrected figures.
   The historical figure and results are preserved, not overwritten.
2. The original verifier imported the same metric and bootstrap functions
   as the experiment. A common PSNR data-range error could pass. A new
   independent numerical module verifies MSE/PSNR with NumPy formulas,
   SSIM by calling torchmetrics directly, HaarPSI by calling PIQ directly,
   and reimplements paired means, medians, win rates and bootstrap.
   It additionally calculates an independent scikit-image SSIM reference.
   The two SSIM implementations can differ by 0.017 on certain PCM images
   because of boundary handling. Do not require numerical identity.
   Original float32 torchmetrics PSNR differs from a float64 NumPy
   recomputation by at most 4.54e-6 dB on the 200 saved images.
   Therefore the independent verifier allows an explicit 1e-5 dB
   roundoff tolerance, while rejecting the deliberate +6.02 dB bug.
3. The tighter opt_tol=1e-6 run is the main numerical sensitivity; old
   opt_tol=1e-3 remains a historical baseline. Neither run establishes
   global optimality in the constrained L1 problem.
4. On the 50 validation images, the historical run showed WLS PSNR losses
   on 40/50, median delta -0.00606 dB, exact two-sided sign p 2.386e-5,
   and paired Wilcoxon p 1.425e-4. Tighter run: losses 39/50, median
   delta -0.00565 dB, sign p 9.02e-5, Wilcoxon p 1.049e-5.
   These describe the fixed validation images only. No held-out images
   or independent 450-image acquisition were touched.
5. Two pairs had solver failures at tighter tolerance. Median native SPGL1
   relative gaps were 0.022 OLS and 0.041 WLS; 8 and 11 had gaps above
   0.1. Robust sensitivity subsets have:
     all 50: mean delta PSNR -0.006711 dB
     48 excluding solver warnings: -0.006955 dB
     21 with both gaps under 0.05: -0.009920 dB
     2 with both gaps under 0.01: -0.005873 dB (inconclusive size)
   Correlation between difference in duality gaps and delta PSNR: +0.161.
   None of these checks certifies exact constrained-L1 solutions.
6. The independent verifier checked original acquisition/weight/epsilon
   SHA256 anchors and recalculated all four metrics on both immutable
   Step 6 archives (200 images in total).
   The ten pilot image pairs were also re-solved with the current code for
   each tolerance archive, a total of 40 reconstructions:
     - tighter 1e-6 archive: 20/20 reconstructed arrays byte-identical
     - historical 1e-3: 10/20 byte-identical, all 10 OLS; the 10 WLS
       have at most 2.19e-6 relative image L2 difference due to later
       reviewed float64 weighted-RHS formation at Step 5.
   This distinction is preserved rather than claiming byte identity for
   solver versions with different numerical expressions.
7. A new separate review provenance sidecar lists inspected code hashes,
   current library versions, original historical commits and the Step 6
   manifest anchors. These metadata are retrospective for historical
   runs: it is not claimed the current versions necessarily match the
   historical process environment.
8. The twelve reviewer-inspired regression tests now all pass, including
   the former strict expected failure for the shared PSNR metric bug.

## Frozen source manifests

Historical 1e-3 Step 6:
8e71d0977db94fde758895db24e193517aaec28d2a040e8d5ca55a2436e1c2e1

Tighter 1e-6 Step 6:
ff92d1f47bcf25fc1c410d164d3a5989c9d43f6b1810ffc9636fa94b607a4f4a

Original Step 5 calibration:
fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a

The Step 2 original acquisitions, Step 3 oracle weights, original Step 5
residual calibration, both Step 6 50-image archives, and untouched 450-image
held-out test partition are all unchanged.

## New implementation files

- scripts/experiments/pcm/oracle_wls/independent_step6_checks.py
- scripts/experiments/pcm/oracle_wls/verify_step6.py (strengthened)
- scripts/experiments/pcm/oracle_wls/review_step6_results.py
- scripts/experiments/pcm/oracle_wls/replay_step6_pilot_review.py
- pytests/pcm/test_oracle_wls_step6_review.py
- scripts/experiments/pcm/oracle_wls/README_STEP6_REVIEW_FOLLOWUP.md

Results (separate versioned archives on tuni):
/store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/step6_reviewer_analysis_v1/
/store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/step6_pilot_replay_review_v1/

Run reviewer tests from the LION root:

    CUDA_VISIBLE_DEVICES=-1 pytest -q \
      -o addopts='' -p no:cacheprovider \
      pytests/pcm/test_oracle_wls_step6_review.py

For independent replication of the figures, run:

    python -m scripts.experiments.pcm.oracle_wls.review_step6_results \
      --original [frozen Step6 1e-3 archive] \
      --tight [frozen Step6 1e-6 archive] \
      --output [new reviewer output folder]

The report source and figures are attached in the companion review ZIP.
The code is also on https://github.com/trung-vt/LION/tree/experiments_10.
