# C60 UTV: learned, clipped and positive-shifted zero-fill maps

## Protocol

- Same 10 frozen PCM validation images, two per category (seed 20261110).
- Same original 20% multilevel Hadamard acquisitions and NPL-default noise.
- Unchanged C60 epoch 60 checkpoint, 512 PDHG steps, 20-iteration CG input, normalized PCM operator and zero image initialization.
- Previous clipping: Lambda_clipped(rho,x) = rho * max(zero_fill(x),0).
- New positive shift: Lambda_shifted(rho,x) = rho * (zero_fill(x) - min(zero_fill) + 1e-5), minimum computed separately for each image; no clipping, no upper cap.
- Original learned C60 map (CNN post-softplus) and PnP-ADMM-tuned-5 are fixed baselines.
- Ternary search on linear rho in [0.0001,10], target final interval width <= 0.01. Additional logarithmic diagnostics and bracket probes.
- The best observed PSNR over all tested rho values is selected, even outside the final ternary bracket.
- PSNR/SSIM/HaarPSI calculated from raw reconstruction arrays.
- All implementation and execution on tuni; no LION source modifications.

## Results

Mean PSNR (dB), n=10:
- PnP-ADMM-tuned-5: 35.895618
- Original learned-map C60: 32.427293
- Previous clipped zero-fill C60: 24.789782
- New shifted zero-fill C60: 26.800192
- Shifted minus clipped: +2.010410 dB.
- Shifted minus learned: -5.627101 dB.
- Shifted better than clipped: 10/10.
- Shifted better than learned: 0/10.
- Shifted best at rho=0.0001: 0/10.
- Best observed candidate outside final ternary bracket: 1/10.

Interpretation: positive shifting is better than clipping negative pixel values,
but C60 with the learned map remains clearly better for these 10 images.
The C60 model and PnP parameters were chosen on validation data, and rho
is tuned here with each image ground truth. This is exploratory oracle
analysis, not an unbiased generalisation assessment. Ternary search assumes
PSNR unimodality, which is not guaranteed.

## Deliverables

- PCM_C60_Shifted_vs_Clipped_vs_Learned_Validation10.pdf: 13 pages,
  executive summary, full metrics table, method description and 10 image pages.
- comparison_all_10_metrics.csv: per-image rho, PSNR, SSIM, HaarPSI,
  improvements, timing and search warnings.
- all10_raw_comparisons.npz: original measurements, GT, indices,
  original zero-fill, PnP, learned C60, clipped C60, shifted C60 and maps.
- Complete_C60_Shifted_ZeroFill_Validation10.zip: all full case outputs,
  previous clipped experiment and current shifted experiment, complete
  parameter-search histories, selection and reproduction scripts.

C60 original source:
  /store/LION/ttv22/pcm/results/pcm/pcm_utv_4090_3x56_winner_snapshot_20261103
Frozen validation archive:
  /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final
Previous clipped study:
  /store/LION/ttv22/pcm/results/pcm/utv_c60_zero_fill_map_val10_20261110_v1
