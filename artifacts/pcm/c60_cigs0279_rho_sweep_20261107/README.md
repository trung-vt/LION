# C60 parameter-map scalar multiplier sweep: CIGS 0279

Status: completed; 100 rho values 0.1, 0.2, ..., 10.0.
One fixed 20% multilevel Hadamard acquisition (NPL-default noise), 512 PDHG iterations.
C60 checkpoint epoch 60. Parameter map predicted once and multiplied by rho after softplus.

PSNR-selected rho: 1.1
PSNR at best rho: 33.453880 dB
PSNR at rho=1: 33.421795 dB
Scalar TV one-image oracle (512 iterations): 33.513805 dB

IMPORTANT: CIGS 0279 belongs to TEST split. Tuning rho by ground-truth PSNR
on this test image is an *oracle sensitivity experiment*, NOT a cross-validated
or unbiased estimate of reconstruction generalisation.

Key files:
- report.pdf and report.tex: one-page overview
- rho_sweep.csv: 100 metric rows (PSNR, SSIM, HaarPSI, MSE, runtime)
- figures/rho_metrics.pdf: all four metrics across rho
- figures/comparison.pdf: GT, zero-filled, TV oracle 512, C60 original, C60 tuned
- figures/parameter_maps.pdf: original, best-scaled and difference maps
- png_reconstructions/: 100 separate 256x256 AFM-hot PNG images
- reconstructions/: all 100 *raw float32* reconstructions as NPZ
- original_lambda_map.npy and best_scaled_lambda_map.npy: learned map and scaled map
- comparison_reference/: archived 512-iteration TV oracle reference
- source/: run and render code, plus run_on_tuni.sh

The full ZIP includes all 100 raw NPZ reconstructions.
The compact ZIP includes all 100 display PNGs but omits the 100 individual NPZs.
Results are saved in the path above on tuni.
