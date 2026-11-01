# SPGL1 Basis Pursuit Denoising (BPDN) for noisy PCM

Implementation branch: experiments_10

Files:
- LION/pcm/spgl1_bpdn.py: SPGL1 BPDN solver, configuration, diagnostic output.
- LION/pcm/spgl1_bpdn_debias.py: **separate optional support-based LS debias module**.
- LION/experiments/pcm_presets/presets.py: separate BPDN methods and test preset.
- LION/pcm/run_pcm_single_sample_experiment.py: method dispatcher and JSON diagnostics.
- pytests/pcm/test_pcm_spgl1_bpdn.py: original BPDN CPU tests.
- pytests/pcm/test_pcm_spgl1_bpdn_debias.py: debias tests and parity check with legacy BP.

## 1. Why debiasing is separate

BPDN, run by spgl1.spg_bpdn, approximately solves

    min_w ||w||_1  subject to ||A Psi^{-1} w - y||_2 <= epsilon.

Here w contains wavelet coefficients, Psi^{-1} is wavelet synthesis, A is the
normalised PCM sensing operator, and y is the DIFFERENTIAL noisy PCM measurement.

BPDN encourages sparse coefficients, often shrinking their magnitude.
The optional second stage does not change the support S chosen by BPDN.
Instead, it re-estimates coefficient values on that support using least squares:

    S = {j: |factor * w_j| > debias_support_tol}

    min_v 0.5 * ||A Psi^{-1} v - y||_2^2
    subject to v_j = 0 for j outside S.

This is **the same debias_ls implementation** already used by the legacy
SPGL1 Basis Pursuit method in LION/pcm/reconstruct.py. It uses the DebiasOp
support operator and runs
gradient descent up to debias_iters (or until the relative update reaches
debias_tol). The legacy BP call still uses spg_bp unchanged, although its shared torch wrapper now also accepts positive BPDN sigma.

Important: the post-debias result is NOT guaranteed to minimise the original
BPDN objective. It may increase ||w||_1, and feasibility and reconstruction
quality must be checked after the refit. No improvement is guaranteed.

## 2. Two selectable method IDs

The new NOISY_PCM_METHODS collection contains both:

- **spgl1_bpdn**: debias=False (default), original BPDN result.
- **spgl1_bpdn_debiased**: debias=True, same initial BPDN settings,
  then support-restricted LS debiasing.

The separate preset **synthetic_png_noisy_bpdn** now includes
the legacy exact-BP method "spgl1" and both BPDN variants. The standard
runner also supplies its shared zero-filled baseline. For each image,
all three reconstruction methods use the same A and y from one acquisition.

The historical QUICK_METHODS, GRID_METHODS and BOX_PLOT_METHODS are not
mutated. The shared BP torch wrapper now supports an optional BPDN sigma,
but sigma=0 still calls the original spg_bp implementation. No PCM data, noise generation,
existing checkpoints or GPU jobs are changed.

## 3. Configuration options

SPGL1BPDNParameter includes:

| Field | Default | Meaning |
| --- | --- | --- |
| noise_rms | 0.06 | Pilot measurement-noise RMS; epsilon = noise_rms * sqrt(M) |
| epsilon | None | If given, overrides the RMS-based bound |
| factor | 1.0 | Scales y and epsilon for SPGL1; final coefficients are unscaled |
| iters | 300 | Maximum SPGL1 iterations |
| wavelet_name | db4 | Wavelet synthesis basis |
| opt_tol | 1e-4 | SPGL1 optimality tolerance |
| verbosity | 0 | SPGL1 logging |
| **debias** | **False** | Toggle the separate least-squares stage |
| debias_iters | 10 | Maximum gradient descent iterations during debiasing |
| debias_support_tol | 1e-5 | Absolute coefficient support cutoff in scaled units |
| debias_tol | 1e-7 | Relative debias update stopping criterion |

For the synthetic 256x256 PCM data at 20% sampling, M=13107. The
initial epsilon using noise_rms=0.06 is approximately 6.87. This is
NOT a calibrated final tolerance, and it is NOT oracle WLS.

The default debiasing parameters are chosen to match the existing BP
post-processing settings. In particular, following the legacy BP pipeline,
support selection is made on the SPGL1-scaled coefficients
(factor * w), not automatically on the original image-unit coefficients.

No clamping, sign constraint, resampling, neural-network training, or
additional noise is introduced.

## 4. Execution order

1. Obtain a frozen or deterministic acquisition with A, y.
2. Build Wavelet2D and CompositeOp B=A Psi^{-1}.
3. Run spg_bpdn on the numerical input factor*y, with the bound
   factor*epsilon. Divide the recovered coefficients by factor.
4. Record the BPDN solver status, iteration count, and the original-unit
   residual ||Bw-y||_2. These metrics are always kept.
5. **If debias=False:** return the wavelet-synthesised BPDN solution.
6. **If debias=True:** call debias_bpdn_coefficients in the new module.
   It calls the existing debias_ls(B, factor*y, factor*w, ...), then
   divides the coefficients by factor. It computes the residual and L1
   norms before and after the LS step.
7. Reconstruct the final image by inverse wavelet transform.
8. The experiment runner saves reconstruction metrics and
   solver_diagnostics.json (when log.save_metrics is enabled). The diagnostics
are returned through SolverDiagnosticsReconOutput, not stored as mutable
state on the method object.

## 5. Diagnostics

solver_diagnostics.json contains all previous BPDN fields, plus:

- debias_enabled: True when debias=True was selected
- debias_applied: True only if at least one coefficient passed the threshold
- debias_support_size: size of the selected wavelet coefficient set
- pre_debias_l1_norm: coefficient L1 norm before LS
- post_debias_l1_norm: coefficient L1 norm after LS
- post_debias_residual_norm: measurement residual after LS
- post_debias_feasible_with_tolerance: separate check against BPDN's epsilon

For the no-debias method, debias_enabled and debias_applied are False;
optional post-debias values are None. The original residual_norm and
feasible_with_tolerance always refer to the original SPGL1 solution.
The method's PSNR/SSIM/MSE are computed on the image actually returned,
meaning after debiasing when enabled.

A new post-debias feasibility flag does not change the reported original
SPGL1 solver convergence status or iteration count.

## 6. Usage

### Change BPDN's debiasing option in Python

    import torch
    from LION.pcm.spgl1_bpdn import SPGL1BPDNParameter, SPGL1BPDNReconFn

    params = SPGL1BPDNParameter(
        noise_rms=0.06,
        iters=300,
        factor=1.0,
        wavelet_name="db4",
        debias=True,
        debias_iters=10,
        debias_support_tol=1e-5,
        debias_tol=1e-7,
    )
    method = SPGL1BPDNReconFn(params, device=torch.device("cpu"))
    result = method(recon_input)
    print(result.diagnostics[0])

When debias=False, the original BPDN behavior is preserved.

### Compare both options using the existing preset

    from LION.experiments.pcm_presets.presets import PRESETS

    config = PRESETS["synthetic_png_noisy_bpdn"]
    assert "spgl1_bpdn" in config.methods
    assert "spgl1_bpdn_debiased" in config.methods

Both methods use the identical generated measurement for each image inside
one run. The historical BP method in the same preset is an additional
reference, NOT an OLS/WLS matched control.

### CPU tests (no GPU required)

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
      python -m pytest -q -o addopts='' -p no:cacheprovider \
      pytests/pcm/test_pcm_spgl1_bpdn.py \
      pytests/pcm/test_pcm_spgl1_bpdn_debias.py

The tests include exact parity with LION's legacy debias_ls on a small
operator, empty support handling, numerical factor support selection,
the no-debias unchanged result, residual logging, and preset registration.

## 7. Fair comparison rules and caveats

Compare BPDN with/without debiasing using the same mask, measurements,
wavelet, epsilon, numerical factor, and first-stage SPGL1 parameters.
Do not select the winning iteration or noise tolerance using held-out test
ground truth. Calibrate first on validation acquisitions.

Running debias_ls can add work: it estimates an operator norm and executes
up to debias_iters support-constrained gradient updates. CPU and GPU
runtime differences depend on the number of selected coefficients and the
number of iterations, not only on the number of pixels.

Debiasing is still unweighted least squares. When oracle WLS is added
later, the optional post-processing should be weighted consistently
if it is enabled on the weighted branch. Otherwise, disable debiasing on
both OLS and WLS for the first strict fidelity-only comparison.

This does NOT implement oracle WLS, full covariance GLS, automatic
epsilon calibration, or a 450-image quality evaluation.

## 8. References

- Local source: LION/operators/DebiasOp.py, function debias_ls.
- Local source: LION/pcm/reconstruct.py, function spgl1_reconstruction
  (legacy BP followed by debias_ls).
- Official SPGL1 BPDN API:
  https://spgl1.readthedocs.io/en/latest/api/generated/spgl1.spg_bpdn.html
- SPGL1 original documentation:
  https://friedlander.io/spgl1/usage/
- G. Koutsourakis, A. Thompson, and J. C. Blakesley, "Toward Megapixel
  Resolution Compressed Sensing Current Mapping of Photovoltaic Devices
  Using Digital Light Processing", Solar RRL 5(11), 2100467 (2021),
  DOI: 10.1002/solr.202100467.


## 9. Review corrections implemented (1 November 2026)

These changes respond to SPGL1_BPDN_review.tex and to the supplied
test_pcm_spgl1_bpdn_extra.py. All tests are CPU-only.

1. **SPGL1-consistent feasibility**:
   within_bpdn_bound checks
   factor*residual <= factor*epsilon + opt_tol*max(1,factor*residual)
   with a small float32 recomputation allowance. The old hard-coded
   max(1e-5, 1e-3*epsilon) check was too restrictive at small epsilon.
   The reviewer's strict xfail for epsilon=0.01 now passes normally.
2. **Shared solver adapter**:
   spgl1_torch(op, y, sigma=...) calls spg_bpdn for positive sigma,
   reusing the exact SciPy LinearOperator adapter as the legacy BP
   call, which still uses spg_bp for sigma=0.
3. **Diagnostics output**:
   the BPDN method returns a SolverDiagnosticsReconOutput; the
   PCM trial runner saves its JSON if metrics are enabled.
   There is no last_diagnostics mutable field.
4. **Richer diagnostics**:
   solver_status_name, solver_converged, coefficient_l1_norm,
   coefficient_nnz_at_1e_minus_5 (threshold 1e-5 in image units),
   lambda_equivalent = info['gnorm'] / factor,
   post_debias_residual_to_epsilon, and the old fields.
   The post-debias residual ratio is especially important: a
   small ratio can indicate over-fitting measurement noise.
5. **Fast debiasing for normalized PCM**:
   debias_ls accepts an optional known safe step_size.
   A unit step is valid for normalized subsampled Hadamard A and
   periodized orthogonal wavelets, since ||A Psi^{-1} P_S|| <= 1.
   The original BP default (power_method norm estimation) is unchanged.
   For unnormalised operators, the BPDN implementation falls back to
   the original norm-estimation method.
6. **Numerical tolerance**:
   the dedicated noisy-BPDN method presets now select opt_tol=1e-6,
   rather than the class default opt_tol=1e-4. The default remains
   useful for quick tests, but cannot certify small duality gaps.
7. **Clear saved epsilon**:
   specifying absolute epsilon sets the inactive noise_rms to None.
   The two method collections no longer share the same mutable
   historical BP parameter object (deepcopy).
8. **Tests**:
   the existing tests now live under pytests/pcm/; the reviewer's
   supplied tests are included without the old expected-failure mark.
   Added checks cover exact identity BPDN, residual scaling, db4
   adjoint and row orthonormality, the BP adapter, solver statuses,
   empty batches, output diagnostics, and debiasing cost avoidance.

### Important remaining points

- The historical BP preset uses factor=1e5 and 100 iterations, and
  its solver status is not yet logged by the original BP pipeline.
  It must not be treated as a convergence-matched BPDN comparator.
  No paper claim should rely on that comparison without a separately
  converged, matched BP baseline.
- The initial RMS=0.06 is **not** a confidence bound or oracle value.
  Select epsilon using validation data or a validated noise model.
- The review's quoted PSNR values for the two artificial images are
  reviewer-generated findings; they are not a held-out benchmark.
- Oracle diagonal WLS and covariance-aware GLS are still future work.
- The new method accepts float64 inputs but the current LION/PCM/Wavelet
  operators compute using float32. Small images with db4 level=0 are
  effectively handled without a multiscale decomposition.

References:
SPGL1 usage: https://friedlander.io/spgl1/usage/
Python BPDN API: https://spgl1.readthedocs.io/en/latest/api/generated/spgl1.spg_bpdn.html
