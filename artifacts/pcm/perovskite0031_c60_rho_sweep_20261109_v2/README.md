# Perovskite 0031 UTV C60 parameter-map sweep, revised

The same frozen 20% multilevel Hadamard measurement with NPL-default noise,
the original C60 epoch-60 checkpoint and 512 PDHG iterations are used.

The post-softplus predicted parameter map is multiplied by rho.
The ordered grid is: 0.01, 0.05, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55,
0.7, 0.9, 0.95, 1.0. Values above 1 are excluded from this PDF.

The rho=1 output exactly matches the saved C60 baseline.
The figures show GT and all 12 reconstructions, each with PSNR, SSIM
and HaarPSI. The metrics are computed from unclipped outputs.

This is a single-image post-hoc oracle analysis, not population-level tuning.
