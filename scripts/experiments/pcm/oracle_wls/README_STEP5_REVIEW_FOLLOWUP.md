# Step 5 independent review: corrections and Step 6 sensitivity

Reviewer materials received 2026-11-02: a detailed LaTeX review and nine extra
pytest cases. Two previously xfail tests documented reproducible gaps.

## Corrections

1. SPGL1 root_found is NOT proof of L1 optimality. Its residual can hit
   epsilon while the relative duality gap is large. The existing shared
   LION/pcm/spgl1_bpdn.py now exposes the native SPGL1 info["rgap"] and
   info["tau"] as two optional backward-compatible diagnostic fields.
   The original solver and Wavelet2D/db4 implementation are unchanged.
2. load_step5_case checks both independent source manifest SHA256 anchors
   EVEN when verify_all=False. With verify_all=True it additionally checks
   every original archived case. Coherent archive editing is rejected.
3. The Step 5 calibration loader enforces status=complete, case-count
   consistency and passing recorded coverage gates. The real frozen
   production archive additionally requires all 50 validation cases.
   Synthetic one-case fixtures remain supported with their distinct hashes.
4. Newly generated calibration archives record source SHA256s, Python and
   package version metadata, and require a single acquisition repeat.
   Original v1 output was not rewritten.
5. The weighted solver forms D*y in float64 and converts once at the
   boundary of existing float32 SPGL1. Final OLS and WLS residual checks
   use original frozen float64 y and weights.
6. The WLS report's one stale Step 3 weight digest was corrected to the
   independently verified true archive ending in 94d1.
7. The nine review cases were added to pytest and both xfails are removed.
   Additional tests cover native SPGL1 gap parity and calibration tampering.

## Frozen noise and recalibration

Original Step 5 calibration manifest SHA256:
    fe4202ffc1eb42054c11624848c33f55d183936b6c73722c8a4d95e8553e108a

A new, provenance-complete v2 calibration was created separately:
    residual_calibration_step5_v2_reviewed_provenance/

Its manifest SHA256:
    6a0108cec36423d278c521f722d9b750779ba1fe0191babb89b0f6dfc2df630a

All 50 per-image numerical records (calibrated epsilon, coverage, random
seeds, counts, source ids) match the original v1 EXACTLY. The original
archived v1 manifest and all original noisy y and weights remain immutable.

Library version caveat: the installed spgl1 distribution metadata reports
0.0.0 while the imported SPGL1 package __version__ reports 0.0.3.
The actual solver implementation is version 0.0.3; the provenance captures
the distribution metadata as provided, without silently reconciling them.

Saved Step 2 noise was compared with 256 fresh Step 5 simulation draws per
image for ALL 50 validation images. Inside the original 95% bound:
OLS 47/50, WLS 47/50 (nominal expectation 47.5).
Mean standardised score about -0.0923 and sample SD 1.018.
No |z| exceeded 2.41; the stored original acquisition arrays are intact.

## Step 6 historical and tighter tolerance

Historical Step 6 v1, opt_tol=1e-3, 120 max SPGL1 iterations:
    8e71d0977db94fde758895db24e193517aaec28d2a040e8d5ca55a2436e1c2e1

Separate, immutable Step 6 reviewer follow-up, opt_tol=1e-6,
1500 max SPGL1 iterations, SAME frozen y, A, db4 wavelet and epsilon:
    step6_opt_tol_1e_minus6_followup_v1/
New manifest SHA256:
    ff92d1f47bcf25fc1c410d164d3a5989c9d43f6b1810ffc9636fa94b607a4f4a

Mean PSNR 29.31456 OLS versus 29.30785 oracle WLS.
Mean WLS-OLS PSNR -0.006711 dB; WLS wins 11/50.
Paired bootstrap 95% interval about [-0.009334, -0.004241] dB.

These results do NOT certify optimisation convergence. Two of fifty
pairs had line-search failures. Median native SPGL1 relative duality gaps
were 0.022 OLS and 0.041 WLS. Eight OLS and eleven WLS gaps exceed 0.1.
Four opt_tol=1e-7 sensitivity checks revealed further line-search failures
and, for some images, non-negligible changes from the 1e-6 solution.

Interpretation: No aggregate advantage from diagonal oracle WLS was
observed in these comparisons, but full constrained-L1 optimality remains
unproven with the existing float32 SPGL1. A dedicated high-precision or
optimality-certified subset is necessary for a stronger theorem-like claim.

## Scope and source references

No stored Step 2, 3, 4, 5 v1 or Step 6 v1 file was edited.
The 450-image independent test split was not accessed.
No modification to the db4 wavelet implementation or underlying solver
numerical algorithm was made. The GPU training on RTX 4090 was untouched.

Primary source code:
- LION/pcm/spgl1_bpdn.py
- LION/classical_algorithms/spgl1_torch.py
- LION/operators/Wavelet2D.py
- scripts/experiments/pcm/oracle_wls/weighted_bpdn.py
- scripts/experiments/pcm/oracle_wls/calibrate_residual_bounds.py
- scripts/experiments/pcm/oracle_wls/review_saved_noise_step5.py
- scripts/experiments/pcm/oracle_wls/run_step6_tighter_tolerance.py

Official SPGL1: https://spgl1.readthedocs.io/
Git fork: https://github.com/trung-vt/LION/tree/experiments_10
