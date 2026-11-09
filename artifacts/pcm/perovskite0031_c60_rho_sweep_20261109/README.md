# Perovskite 0031: C60 global parameter-map scaling

One stored noisy measurement: PCM synthetic held-out Perovskite image 0031,
20% multilevel Hadamard sampling (J=8, coarse_J=6), NPL-default noise.

UTV C60 checkpoint epoch 60, CG-input with 20 iterations and zero initial
image, normalized operator, 512 PDHG iterations. Lambda map predicted once,
from the frozen measurement. For each reconstruction, only the final
positive lambda map is multiplied by rho. No new measurement or noise.

rho = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1, 1.1, 1.3, 1.5].
Metrics PSNR, SSIM, HaarPSI, MSE use raw unclipped reconstructions and
ground truth data range 1. Image displays share [0,1] limits.

The rho=1 run reproduces the archived C60 reconstruction exactly
(max absolute pixel difference = 0.0) and PSNR 30.4574909 dB.

Files: one-page comparison PDF, large PNG, CSV metrics, protocol JSON,
raw reconstructions/ground truth/map in npz and reproduction script
`scripts/experiments/pcm/perovskite0031_c60_rho_sweep_20261109.py`.

This is per-image post-hoc analysis using ground-truth metrics, not
unbiased held-out parameter tuning or a population-level performance result.
