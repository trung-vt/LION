# Step 4: independent Monte Carlo check of oracle WLS variances

Status: validation experiment. This step does NOT tune, train or compare
reconstruction methods. It does not change the synthetic dataset, the stored
Step 2 acquisition, the Step 3 weights, or any shared LION module.

## Why this is independent

Step 3 computed the analytical v_k from the exact NPL noise model. Step 4
samples fresh realizations from the ORIGINAL PCM implementation:

- LION/pcm_tools/noise_generate.py::generate_noise
- LION/pcm_tools/noise_generate.py::add_noise

The full simulator uses the same two functions. A unit test checks that the
cached-clean-pattern Monte Carlo implementation gives byte-identical residuals
to LION/pcm/simulation.py::simulate_pcm_acquisition for three fixed seeds.

The RNGs use independent SHA-256-derived Step 4 streams and never reuse the
frozen Step 2 noise realizations. The mask A and clean ground truth x stay
unchanged for a given image. The saved oracle variance is only used as a
reference AFTER empirical noise has been generated.

## Prespecified experiment settings

Source: 50-image Step 2 reviewed validation archive, 10 per category.
Oracle reference: 50-image Step 3 frozen float64 weight archive.

Full NPL experiment:
- First two approved validation permutation positions (901 and 902) per
  category, ten images total, selected by a fixed rule before seeing results.
- 4096 independent fresh simulated acquisitions per image.
- For each Hadamard coefficient k, save the empirical mean and unbiased
  sample variance of y_k-(Ax)_k. Use float64 Welford moments online.

Four isolated noise controls (one image per category):
- Dark, independent bias, shared bias gradient, integrated ARMA beam.
- 1024 independent realizations for each control on each image.
- The same original noise-generation functions are used, with the other
  standard deviations set to zero. No new physical noise implementation.

Recorded per image/mode: variance ratio mean and median, median absolute
relative error, p95 relative variance error, residual-mean z-scores, and
individual empirical/analytical vectors, case seeds and hash identities.
Monte Carlo noise draws are independent within one coefficient across
realizations. Different coefficients in a single acquisition may be
correlated; never regard them as independent observations.

## Acceptance checks, locked before the full run

For full NPL, 4096 draws:
- Median absolute relative error <= 0.09.
- 95th percentile absolute relative error <= 0.18.
- Mean empirical/predicted variance ratio in [0.93, 1.07].
- At least 98% of pointwise residual-mean z-scores have |z| < 3.5.

For single-component controls, 1024 draws:
- Median absolute relative error <= 0.14.
- 95th percentile relative error <= 0.27.
- Mean variance ratio in [0.87, 1.13].
- At least 98% of pointwise residual-mean z-scores have |z| < 3.5.

These are deliberately conservative engineering diagnostic checks, NOT
simultaneous confidence intervals or proof that the full covariance is
diagonal. A Gaussian reference gives relative sample-variance standard
errors ~sqrt(2/(n-1)), or approximately 2.2% at 4096 draws and 4.4% at
1024 draws. The actual swap-marginal distribution is a mixture of Gaussians,
so this reference is only approximate.

A failure is still saved as evidence; it does not automatically trigger
new weights or a changed threshold. Investigate and version any corrections.

## Frozen sources and locations

Step 2 acquisition manifest SHA-256:
b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

Step 3 oracle-weight manifest SHA-256:
50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1

Base run directory:
    /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102

Step 2 input: acquisitions_v5_reviewed_final/
Step 3 input: oracle_weights_step3_v2_final/
Step 4 output: monte_carlo_step4_v1/

All new Step 4 files are in their own location. Source images, noisy
measurements and oracle weights are read-only in this workflow. The Monte
Carlo result archive includes protocol.json, manifest.json, summary.csv,
SHA256SUMS, and separate NPZ summaries. Large raw random traces are not
saved. The complete source, tests, data summaries, independent anchor hash,
and exact code-change list belong in the review ZIP.

## Commands

Use the LION repository root, configured LION_DATA_PATH and LION_TTV22_PATH,
and CPU. The existing 4090 training job is unaffected.

    python -m scripts.experiments.pcm.oracle_wls.monte_carlo_validate \
      --acquisitions /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final \
      --weights /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/oracle_weights_step3_v2_final \
      --output /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/monte_carlo_step4_v1 \
      --expected-acquisition-manifest-sha256 b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac \
      --expected-weight-manifest-sha256 50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1 \
      --repetitions 4096 --control-repetitions 1024 --cases-per-category 2

After the build, independently record EXTERNAL_STEP4_MANIFEST_SHA256 and
verify with the separately stored digest:

    python -m scripts.experiments.pcm.oracle_wls.verify_monte_carlo \
      --acquisitions [same Step 2 folder] --weights [same Step 3 folder] \
      --output [Step 4 output folder] \
      --expected-acquisition-manifest-sha256 [Step 2 digest] \
      --expected-weight-manifest-sha256 [Step 3 digest] \
      --expected-monte-carlo-manifest-sha256 [Step 4 digest] --replay

A strict replay recalculates the entire random simulation with the original
deterministic per-case seeds and compares the saved empirical mean and variance
byte-for-byte (same CPU/Python/dependency environment). Integrity-only verify
checks original data hashes, summary fields and file membership.

## Scientific limits

- The full NPL variance includes all noise sources and swap marginalization.
  The isolated component controls inspect each formula contribution.
- This verifies the *diagonal marginal variances*. It does not establish
  that the noise covariance is diagonal, nor that WLS is optimal compared to
  a full generalized least-squares solution.
- The evaluation intentionally uses ground truth for the oracle reference.
  This is a controlled upper-information benchmark, not a practical estimator.
- No SPGL1-BPDN modification, PSNR/SSIM comparison, training run or held-out
  test acquisition is performed in Step 4.

References:
- Original LION noise and simulation code above.
- Step 3 mathematical derivation: README_STEP3.md.
- SciPy lfilter:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
- statsmodels ARMA samples:
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html
