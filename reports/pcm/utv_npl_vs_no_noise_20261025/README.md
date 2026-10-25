# U-TV PCM benchmark: fixed NPL default vs no simulated noise

[**View the 12-page PDF with graphs and paired images**](npl_default_vs_no_noise.pdf)

- Frozen best-validation supervised U-TV checkpoint: epoch 5, step 22500
  from the 1000-epoch variable-noise run.
- Same **50 held-out test images** (10 per category), 6 fixed sampling ratios:
  5%, 10%, 20%, 30%, 40%, 50%, with coarse J=6.
- 300 reconstructions with the unchanged PCMNoiseSimulation.npl_default
  and 300 **clean** reconstructions using noise=None in the existing
  LION acquisition simulator.
- **Sampling masks are matched** between noisy and noiseless conditions
  for every image/ratio (same acquisition seed). There is one noise
  realisation per image/ratio.
- **Single shared parameter-map scale-factor key** (no rate-specific
  learned scale factors).
- Metrics: untouched LION MSE, PSNR, SSIM, HaarPSI, Pearson; calculated
  from full reconstructed tensors, not from display PNGs.
- [Per-image metrics](comparison_per_image.csv): 600 rows.
- [Summary metrics](comparison_summary.csv): mean and SD by category
  (plus overall), condition, sampling ratio.
- The PDF includes **two test images per category** at every sampling ratio,
  with NPL-default and clean reconstructions aligned.
- Images were rendered through the **existing** PCMResultWriter with
  common afmhot limits [0, 1] from the original typed render config.
  The report reuses those PNGs without changing colour normalization.

The previous independent-variable-noise sweep is a different experiment
and cannot be interpreted as a mask-paired controlled noise comparison.
The immutable checkpoint snapshot (not included in GitHub) is in
/store/LION/ttv22/pcm/results/20261025_utv_ratio_sweep_snapshot/.
The original RTX 4090 training job was not changed.
