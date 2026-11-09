# Perovskite 0292: scalar TGV vs PnP-ADMM-tuned-5 and best scaled UTV C60

## Objective and solver

Second-order scalar TGV with a componentwise (anisotropic) L1 penalty:

min over u,v:
    0.5 ||A u - y||_2^2 + alpha1 ||grad(u)-v||_1 + alpha0 ||E(v)||_1

- Implementation: existing LION AdaptiveTGVPCMNetwork, using the MRPro-backed LION PDHG, rather than a newly written TGV algorithm
- LION PCM operator: normalized Hadamard, 20% multilevel, j_order=8, coarse_J=6, frozen NPL-default differential measurement
- Ground truth: held-out Perovskite image0292.png, 256x256
- Frozen measurement and sampled-pattern SHA256 verified
- PDHG tau=sigma=0.18, theta=1.0, conservative stability upper bound tau*sigma*||K||^2 <= 0.5508 < 1
- Fixed zero initialization and matching OLS fidelity; no learned parameter map in scalar TGV

## Ground-truth oracle parameter search

- 35 coarse (alpha1, alpha0) pairs at 512 iterations
- 39 refinements around six promising coarse candidates at 512 iterations
- 30 local grid pairs at 512 iterations
- Top candidate checks at 2048, 4096 and 8192 iterations
- Final selected alpha1=0.028284271247461905, alpha0=0.01, 8192 PDHG iterations
- Final PSNR 30.436258316 dB, SSIM 0.740215063, HaarPSI 0.734537542, MSE 0.000904427958
- Final chosen setting is stable in PSNR from 2048 to 8192 iterations (<0.001 dB change)
- This is one-image, ground-truth-based oracle tuning, not an unbiased test-set or population comparison
- The scalar TGV model has two weights; using another definition of TGV (e.g. isotropic Frobenius norm) may need different weights

## Matched comparisons (same acquisition)

- PnP-ADMM-tuned-5: PSNR 32.369354 dB, SSIM 0.926871, HaarPSI 0.817647
- UTV C60 at oracle rho=0.38344238: PSNR 29.408306 dB, SSIM 0.841750, HaarPSI 0.681442
- Scalar TGV tuned: PSNR 30.436258 dB, SSIM 0.740215, HaarPSI 0.734538

TGV has a higher PSNR and HaarPSI than tuned C60 here, but lower SSIM.
PnP-ADMM-tuned-5 has the highest metrics of all three.

Note that the iteration numbers are not directly equivalent as measures of speed:
TGV runs 8192 PDHG steps; C60 runs 512 PDHG steps; PnP-ADMM uses its own
solver/denoiser iterations with different per-iteration computational costs.

## Deliverables

- Perovskite_0292_TGV_vs_PnP_ADMM_vs_UTV_Best.pdf: 2-page figure with whole-image comparison, zoom, parameter sweep and methods
- comparison_metrics.csv: three methods, PSNR/SSIM/HaarPSI/MSE
- four_way_raw_arrays.npz: exact GT, noisy measurement, indices, PnP, UTV, TGV and parameters
- best_tgv.npy: raw scalar TGV reconstruction
- all_tuning_metrics.csv, coarse_512.csv, fine_512.csv, local_grid_512.csv, validate_*.csv: full parameter scan and convergence
- selected.json, protocol.json: selection and run provenance
- Frozen inputs and source data in frozen_case.npz, pnp_admm_tuned5.npy, utv_c60_optimised_rho.npz
- Source scripts: scripts/experiments/pcm/perovskite0292_scalar_tgv_mrpro_20261109.py and perovskite0292_scalar_tgv_report_20261109.py

## References

- Bredies, Kunisch and Pock (2010), Total Generalized Variation: https://doi.org/10.1137/090769521
- MRPro documentation: https://docs.mrpro.rocks/
- MRPro source: https://github.com/PTB-MR/mrpro
