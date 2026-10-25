# PCM U-TV: matched zero-filled and in-order zero-filled baselines

[**Open the 41-page A3 PDF**](utv_zero_filled_clim_0_1p2_individual_metrics.pdf)

The PDF reports 10 distinct **held-out TEST images** (two per category,
CIGS, Organic, Perovskite, Reference, Silicone) at 5, 10, 20, 30, 40
and 50 percent Hadamard sampling, coarse J=6. Every displayed reconstruction
has **MSE, PSNR, SSIM, and HaarPSI** computed using the original LION
compute_metrics on the full-precision tensors, not on display PNGs.

Each example shows:
- Original grayscale ground-truth PNG, read as 8-bit L and divided by 255;
- Zero-filled *matched multilevel*: LION's op.pseudo_inv(y) with **the
  exact same operator and noisy/noiseless measurements as learned U-TV**;
- Zero-filled *in-order*: a new acquisition using the first m Hadamard
  patterns in coarse-to-fine order, at the same sampling ratio and
  noise preset as the U-TV case; no different learned weights;
- U-TV from the **frozen epoch-5 / step-22,500 checkpoint**, one learned
  scale-factor key for all ratios, 512 unrolled PDHG iterations.

Two acquisition conditions are included: the unchanged fixed NPL-default
noise preset and the *truly noiseless* simulator (noise=None). In-order
measurements are separately simulated and, when noisy, have their own
noise realisation. The U-TV / multilevel zero-filled pair uses identical
measurements within each condition.

The **entire image display has fixed afmhot limits [0, 1.2]**, applied
by the existing PCM result renderer. Colorbars use the same limits.
No per-image autoscaling. Color limits do not affect MSE/PSNR/SSIM/HaarPSI.

- [Individual metrics for all 360 reconstructions](individual_metrics_all_methods.csv)
- [Method averages/SD by noise condition and sampling rate](method_comparison_summary.csv)
- [Typed evaluation configuration used](evaluation_config_used.yaml)

**Quantitative scope:** 120 U-TV, 120 matched multilevel zero-filled,
120 in-order zero-filled reconstructions, across 10 held-out examples.
The first page reports means over these **10**, not over the 50-image
quantitative benchmark in the earlier report. Data is descriptive.

No separate scripts were added to LION. The original library tester
was extended with optional baseline output controls on the development
branch at
[trung-vt/LION commit d0e3bff](https://github.com/trung-vt/LION/commit/d0e3bff4c323187ce84f1854cc44ce165e08a55d).
The presentation assembler is kept outside the source tree with
the evaluation results on tuni. The immutable best-checkpoint copy
is likewise not committed; original long training remains unaffected.
