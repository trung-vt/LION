# Step 5: Oracle WLS-BPDN and calibrated 95% residual constraints

## Minimal code-change policy

Step 5 adds experiment-specific files only. Existing LION PCM noise
simulation, SPGL1-BPDN, spgl1_torch, Wavelet2D and PhotocurrentMapOp
remain UNCHANGED. The current PCM wavelet is the periodized orthogonal
Daubechies db4 wavelet. Debiasing is disabled in this initial comparison.
The same noisy measurement y and mask A are loaded from Step 2 for both
OLS and WLS, and all original images and archived data are read-only.

New files:
- weighted_bpdn.py: diagonal operator wrapper, protected loader and BPDN call.
- calibrate_residual_bounds.py: independent 95% noise-residual calibration.
- pytests/pcm/test_oracle_wls_step5.py: operator, solver and archive tests.
- README_STEP5.md: this documentation.

## Model

Let B=A Psi^{-1}, where Psi denotes the UNCHANGED db4 wavelet analysis
operator. The inverse-variance weights in Step 3 are w_m=1/v_m. Set
D=diag(1/sqrt(v_m)) using the stored, float64 inverse standard deviations.

OLS:  min ||z||_1 subject to ||Bz-y||_2 <= epsilon_OLS.
WLS:  min ||z||_1 subject to ||D Bz - D y||_2 <= epsilon_WLS.

The forward and matching adjoint are:
    weighted_forward(z) = D B z
    weighted_adjoint(r) = B^* D r.

Tests compare the correct adjoint dot products and verify that D=I
reduces to the original BPDN operator. Applying D only to the noisy
measurements, without weighting the operator, is incorrect.

## Precision and solver

Step 2 noisy measurement and Step 3 variance/whitener remain float64
in their protected NPZ archives. The existing Wavelet2D and SPGL1
adapter explicitly use float32, so only the SOLVER INPUTS are cast
to float32. The weighted wrapper uses one float32 representation of D
for BOTH forward and adjoint applications, thus preserving symmetry
at the solver precision. The SPGL1 implementation is reused directly.

The solver factor is fixed to 1.0. Same db4 wavelet, periodization,
iteration limits and optimality tolerance apply to both OLS and WLS.
No shared solver code changes and no optional post-debiasing.
All solver results must retain convergence status, iterations, original
OLS residual norm, standardized WLS residual norm, and float32 roundoff
information. Do not claim reconstruction quality benefits in Step 5.

## Relation to noise variances

Under the zero-mean model:
    E||y-Ax_true||_2^2 = sum_m v_m
    E||D(y-Ax_true)||_2^2 = M.

The square roots of these two expected squared residuals are reference
scales, NOT automatic 95% quantiles. The PCM noise is correlated across
measurements because of its integrated ARMA beam component, so a naive
chi-square quantile is not assumed.

## Step 5 95% empirical calibration

For EACH of the 50 approved validation images, hold the original x,A
and Step 3 variances fixed. Use original LION.generate_noise/add_noise
to create independent simulated noise realizations. They are NOT the
saved Step 2 noisy y, and not the existing Step 4 Monte Carlo seeds.

- 2048 independent draws per image estimate the upper empirical 95th
  percentile, using NumPy quantile(method='higher').
- 1024 different independent draws per image assess coverage, without
  changing or tuning the 95th-percentile thresholds.
- For each draw, use the SAME error vector e for OLS norm(e) and
  WLS norm(D e). This makes the calibration a matched comparison.
- Save per-image epsilon_ols_95 and epsilon_wls_95, the reference
  square-root-variance scales, independent RNG seeds and holdout coverage.
- Predeclared engineering coverage gate is 91% to 99% for each method
  and case. This is not a formal simultaneous statistical confidence
  interval. At 1024 independent realizations, the binomial reference
  sampling SD near 95% is about 0.68%, not including quantile error.

The ground truth is needed ONLY to generate the calibrated thresholds
under the oracle noise model. The SPGL1 solver never receives the ground
truth or uses it during reconstruction.

## Frozen input identities

Step 2 validated acquisition manifest SHA-256:
b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

Step 3 oracle weight manifest SHA-256:
50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1

Results on tuni:
    /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/residual_calibration_step5_v1/

The calibration prints a separately recorded SHA-256 of manifest.json.
The loader refuses any unanchored calibration or mismatched source hashes.
Any existing output directory is never overwritten.

## Reproducibility

Run from the LION repository root on CPU, with LION_DATA_PATH set:

    python -m scripts.experiments.pcm.oracle_wls.calibrate_residual_bounds \
      --acquisitions [Step 2 frozen archive] \
      --weights [Step 3 frozen weights] \
      --output [new Step 5 calibration directory] \
      --expected-acquisition-manifest-sha256 [Step 2 anchor] \
      --expected-weight-manifest-sha256 [Step 3 anchor] \
      --fit-draws 2048 --holdout-draws 1024

Then run a single image using:
    python -m scripts.experiments.pcm.oracle_wls.weighted_bpdn \
      --acquisitions [Step 2 folder] --weights [Step 3 folder] \
      --calibration [Step 5 folder] \
      --expected-calibration-manifest-sha256 [Step 5 external anchor] \
      --category CIGS --sample-id 733 --method wls --iters 120

Choose --method ols for the matched unweighted solve using the SAME
frozen acquisition. Validation and test-set generation remain separate.
Step 6 performs the actual PSNR/SSIM, detail and convergence comparison.
No improvement in quality is claimed here.

References:
- LION/pcm/spgl1_bpdn.py : spgl1_bpdn_torch
- LION/classical_algorithms/spgl1_torch.py : float32 SPGL1 adapter
- LION/operators/Wavelet2D.py : unchanged db4/periodized wavelet
- scripts/experiments/pcm/oracle_wls/oracle_variance.py : v_m
- scripts/experiments/pcm/oracle_wls/monte_carlo_validate.py : original noise draws
- https://spgl1.readthedocs.io/
