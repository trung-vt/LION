# C60 learned regularisation versus scaled zero-fill parameter maps

## Protocol
- 10 frozen validation images, 2/category; RNG seed 20261110.
- Categories: CIGS, Organic, Perovskite, Reference, Silicone.
- The same frozen 20% multilevel Hadamard measurement, NPL-default noise and GT for all compared methods in each case.
- Fixed 60-epoch C60 checkpoint and 512 PDHG iterations, with 20-step CG input.
- Learned C60 reference: original post-softplus learned positive map, no scaling.
- Replacement regularisation map: lambda(rho,x) = rho * max(zero_filled(x),0).
  Negative zero-fill pixel values were clamped to zero because L1 weights
  cannot be negative. No upper clipping or map normalisation was applied.
  The CNN is not used to derive the replacement parameter map.
- Ternary search in linear rho [0.0001,10], target final interval width 0.01,
  extra logarithmic diagnostic values, and final interval probes.
- Per-case selected rho is the best ground-truth PSNR among evaluated values.
- PnP-ADMM-tuned-5: previously validation-selected rho=0.5, sigma=0.075,
  unchanged DRUNet with residual stopping. Not tuned using per-image GT.
- PSNR, SSIM and HaarPSI use original raw (unclipped) reconstructed values.

## Results
Mean PSNR over 10 cases: PnP 35.89562 dB;
learned C60 32.42729 dB;
zero-fill-map C60 24.78978 dB.
Mean zero-fill-map minus learned C60: -7.63751 dB.
Zero-fill map improves over learned in 0/10 cases.
Selected rho at lower boundary: 9/10.
Best evaluated rho outside final ternary bracket: 3/10.

Ternary search presumes unimodal PSNR. Unimodality is not proven for these
nonlinear, 512-step reconstructions. The best *observed* rho is reported
even if it falls outside the final bracket. Frequent lower-bound choices
suggest the given search range may omit a better smaller rho.

The CNN checkpoint and PnP baseline had been selected using validation data,
and per-image rho uses validation GT (oracle). This is exploratory, not
held-out population-level evidence.

## Deliverables
- 13-page comparison PDF (3 summary/protocol pages, 10 visual pages).
- all_10_metrics.csv: metrics and selected rho per case.
- all10_raw_reconstructions.npz: GT, measurement, indices, zero-fill, PnP,
  learned C60, zero-fill C60, and both parameter maps for all 10 cases.
- Complete_C60_ZeroFillMap_Validation10_Experiment.zip: frozen validation
  acquisitions, model inputs, results, search traces and reproducibility scripts.
- report_summary.json and frozen_10_manifest.json: provenance and summaries.

No changes made to LION code. Everything was run and generated on tuni.

Original sources:
- LION PCM acquired validation archive:
  /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final
- Fixed C60 checkpoint:
  /store/LION/ttv22/pcm/results/pcm/pcm_utv_4090_3x56_winner_snapshot_20261103/checkpoints/best_val_epoch60.pt
- PnP-ADMM tuned validation algorithm:
  scripts/experiments/pcm/pnp_admm_tuning/step3_final_heldout.py
