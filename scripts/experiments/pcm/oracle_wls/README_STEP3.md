# Step 3: exact marginal oracle PCM variances (float64)

Status: analytical diagonal variances and weights implemented for 50 stored
validation acquisitions, with deterministic algebraic tests and a separate
50-image synthetic end-to-end test. Monte Carlo variance validation is Step 4.

## Model read from the existing simulator

Original source references:
- LION/pcm_tools/noise_generate.py: generate_noise, generate_bias_noise,
  generate_beam_noise, add_noise.
- LION/pcm/simulation.py: simulate_pcm_acquisition.
- LION/pcm/noise.py: PCMNoiseSimulation.npl_default.

For each Hadamard measurement k, let p_k and q_k be its clean normalized
positive and negative split currents. The noisy differential measurement is
(p_k+n_plus) - (q_k+n_minus). A separately drawn independent random swap
per pair assigns the adjacent noise-trace samples with probability 1/2 each.

The bias trace is I0*(1+t*g)+I0*sigma_bias*xi_t, with one Gaussian gradient g
per acquisition repeat. Dark noise is independent Gaussian of standard deviation
sigma_dark. Beam noise is an integrated zero-initial-state ARMA trace and
multiplies p_k and q_k, not the background current I0.

All source noises are independent of one another and of the swap draws.
E[y_k|x,A]=(Ax)_k mathematically, up to floating point rounding.

## Exact marginal variance

Let D=p_k-q_k and S=p_k+q_k. Let R_a,R_b be integrated beam trace values
at the adjacent zero-based exposure times a=2k, b=2k+1. Averaging the two
swaps removes the beam cross term:

    Var_beam(k) = (D**2/4)*Var(R_a+R_b)
                +(S**2/4)*Var(R_a-R_b).

The other contributions are all nonnegative:

    Var_dark = 2*sigma_dark**2
    Var_bias_white = 2*(I0*sigma_bias)**2
    Var_bias_drift = (I0*sigma_gradient)**2

The total marginal variance is the sum divided by the number of independent
acquisition repeats; the locked archive has one repeat.

For the integrated ARMA trace, use the exact finite impulse response h_j
of the simulator's ARMA filter. Put H_j=sum_{i=0}^j h_i, H_{-1}=0.
For b=2k+1:

    Var(R_a+R_b) = sigma_beam**2 *
                    sum_{j=0}^b (H_j + H_{j-1})**2
    Var(R_a-R_b) = sigma_beam**2 * sum_{j=0}^b h_j**2.

These are sums of nonnegative terms, avoiding cancellation in
Var(R_a)+Var(R_b)-2*Cov(R_a,R_b). The code calculates the impulse response
with scipy.signal.lfilter. The original PCM simulation uses statsmodels
arma_generate_sample and then np.cumsum with zero initial state.

Raw values are saved as variance (float64), weight=1/variance,
inverse_std=1/sqrt(variance), and four separate variance contributions.
There is no arbitrary scaling, clipping or learned weight map.
Nonpositive or nonfinite variances are rejected.

This is a diagonal marginal oracle. Integrated beam noise may correlate
different Hadamard measurements; the diagonal inverse variances are not the
inverse of the full covariance matrix. Ground-truth x and A enter the formula,
but noisy measurements y and realized swap flags do NOT.

## Input, output and reproducibility

Input: frozen Step 2 reviewed archive:

    /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final

Independently pinned source manifest SHA-256:

    b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

Output: separate directory oracle_weights_step3_v2_final containing 50
per-image compressed float64 NPZ files, manifest.json, protocol.json,
SHA256SUMS. Output is atomic and existing directories are never overwritten.
Save the printed Step 3 manifest hash outside the output folder, to anchor it.
No original images or 450 held-out test acquisitions were touched.

From the LION root, with LION_DATA_PATH and LION_TTV22_PATH configured:

~~~bash
python -m scripts.experiments.pcm.oracle_wls.build_oracle_weights build \
  --acquisitions /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final \
  --output /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/oracle_weights_step3_v2_final \
  --expected-acquisition-manifest-sha256 b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

python -m scripts.experiments.pcm.oracle_wls.build_oracle_weights verify \
  --acquisitions /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final \
  --output /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/oracle_weights_step3_v2_final \
  --expected-acquisition-manifest-sha256 b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac \
  --recompute
~~~

## Tests and limits

- pytests/pcm/test_oracle_wls_variance.py tests limiting cases, independent
  repeat scaling, direct finite-matrix covariance, true marginalization
  over swaps, and agreement with the actual simulator's ARMA implementation.
- pytests/pcm/test_oracle_wls_step3_e2e.py builds a disposable synthetic
  50-image archive, reproduces every weight file, tests corruption, and
  verifies that changes to realized y and swaps do not change oracle weights.
- The actual stored 50 validation cases are processed without re-simulating
  any noisy measurement.
- Step 4, independent Monte Carlo verification against repeated stochastic
  acquisitions, has not yet been run.
- No weighted SPGL1-BPDN integration, PSNR claim, or held-out test run is part
  of Step 3.

External references:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html


## Independent Step 3 reviewer follow-up (2 November 2026)

The external reviewer supplied eleven mathematical stress tests and four
weight-archive tampering tests. All eleven mathematical tests passed without
changes to oracle_variance.py. The original integrity tests reported two
strict expected failures: (1) successful verification without an independent
weight-manifest anchor or analytical recomputation; (2) a protocol.json
change with a correspondingly updated internal SHA256SUMS line.

The two gaps were fixed only in build_oracle_weights.py. A successful
verify_weights call now requires either:

- expected_weight_manifest_sha256 obtained and retained **outside** the
  weight archive; or
- recompute=True, which recomputes the analytical arrays from ground truth.

The check for this requirement runs after the per-file source binding checks
so that swapped NPZ files still produce the more specific source-measurement
identity error. The verifier also compares the actual protocol.json digest
against manifest['protocol_sha256'], confirms the step's declared diagonal
covariance convention and compares the recorded input protocol hash to the
frozen Step 2 protocol. It does not silently update any manifest or checksum.

Both reviewer expected-failure marks have been removed after fixing them.
The expanded tests live in:

- pytests/pcm/test_oracle_wls_step3_reviewer_math.py
- pytests/pcm/test_oracle_wls_step3_reviewer_integrity.py

The original 50 weight files are preserved byte-for-byte, including
their original external manifest SHA256. The source file
oracle_variance.py is unchanged (SHA256
e4c6d14c68f15d3bee1ed39d169da858799ef5b736f9f8f3e2c85a69c2224cde).
The Step 4 Monte Carlo implementation is also unchanged. After these
verification-only changes, rechecking the original Step 4 archive is
appropriate; the stochastic experiment itself does not need repetition.

The earlier text in this README describes historical Step 3 status when
the weights were first created. Independent Step 4 Monte Carlo validation
was subsequently completed (see README_STEP4.md).
