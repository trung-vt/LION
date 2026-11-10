# Perovskite 0292: scalar TGV-3, MRPro-backed LION PDHG

## Data and solver

- Same frozen 20% multilevel Hadamard PCM acquisition, NPL-default noise.
- Measurement SHA256: f8ffea66daeedaef8df0b1f711d609266d654d86af5dd6da34f6b13b2c287dfa.
- Pattern SHA256: 6a81317079c12733583d014d158fe0ecb46d3739f6d67cc027edfad0960cc9f5.
- Scalar TGV-3 objective:
  0.5||Au-y||^2 + alpha2||grad(u)-v||_1
  + alpha1||E1(v)-w||_1 + alpha0||E2(w)||_1.
- Full tensor componentwise L1 (anisotropic), MRPro block PDHG,
  zero initial image, tau=sigma=0.18; TGV-2 solver unchanged.

## Search protocol and result

A finite oracle grid selected scalar parameters using this image's GT PSNR.

- Coarse: 125 configurations at 512 iterations.
- Expanded coarse near the alpha0 boundary: 27 at 512 iterations.
- First shortlist: 13 at 2048 iterations.
- Local search: 81 at 1024 iterations.
- Second shortlist: 6 at 4096 iterations.
- Finalists: 3 at 8192 iterations.
- Total 255 parameter/iteration evaluations, plus one repeated finalist.
- Best tested alpha2=0.032, alpha1=0.010, alpha0=0.024.
- Final 8192-step PSNR 30.4643287659 dB, SSIM 0.7402219772,
  HaarPSI 0.7335723639, MSE 0.0008986011380.
- Same tuple at 4096 steps: PSNR 30.4646472931 dB,
  a difference of just 0.00032 dB.

## Exact-acquisition method comparisons

| Method | PSNR (dB) | SSIM | HaarPSI |
|---|---:|---:|---:|
| PnP-ADMM-tuned-5 | 32.369354 | 0.926871 | 0.817647 |
| Best C60 UTV (rho=0.3834424) | 29.408306 | 0.841750 | 0.681442 |
| Best scalar TV (lambda=0.0240982) | 30.369715 | 0.814062 | 0.725788 |
| Best scalar TGV-2 (alpha1=0.0282843, alpha0=0.01) | 30.436258 | 0.740215 | 0.734538 |
| Best tested scalar TGV-3 | 30.464329 | 0.740222 | 0.733572 |

TGV-3 gains 0.02807 dB PSNR over TGV-2 with nearly identical
SSIM and slightly lower HaarPSI. The visual result is also similar.
This does not establish that TGV-3 generally outperforms TGV-2.
PnP-ADMM-tuned-5 is still ahead on all three quality metrics.

## Files and reproduction

- Perovskite_0292_TGV3_vs_TGV2_TV_UTV_PnP.pdf: 2-page comparison
- six_way_raw_arrays.npz: GT, measurement, indices, all reconstructions
- comparison_metrics.csv: all method PSNR, SSIM, HaarPSI, MSE
- coarse_512.csv, coarse_shortlist_2048.csv, local_1024.csv,
  shortlist_4096.csv, finalists_8192.csv: full search history
- summary.json, report_protocol.json: run audit
- tgv3_best_scalar.npy: TGV-3 final output
- Scripts in scripts/experiments/pcm, run on branch
  feature/pcm-tgv3-mrpro-20261109, with stages:
  coarse, expand, refine, validate; then the comparison script.

## Cautions and references

GT-guided single-image oracle tuning, not an unbiased generalisation
result. These are the best tested weights, not a certified global optimum.
The shared LION PDHG interpolation is not the textbook Chambolle-Pock
extrapolation. No formal convergence theorem for this exact version is
claimed. TGV-3 costs more computation per iteration than TGV-2.

- Bredies, Kunisch and Pock (2010), TGV, Equation 3.6:
  https://doi.org/10.1137/090769521
- Chambolle and Pock (2011):
  https://doi.org/10.1007/s10851-010-0251-1
- MRPro docs: https://docs.mrpro.rocks/
