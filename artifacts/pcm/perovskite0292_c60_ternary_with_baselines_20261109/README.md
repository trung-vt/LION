# Perovskite 0292: C60 ternary search and matched comparison

## Dataset and reproducibility
- PCM synthetic held-out Perovskite_0292; 20% multilevel Hadamard sampling (J=8, coarse_J=6), fixed NPL-default noise
- Its previous 450-test acquisition was deterministically replayed; frozen measurement and sampling-index SHA256 values match exactly
- C60 and zero-fill reconstruction MSEs match the original 450-image benchmark record
- C60 epoch-60 checkpoint with 20-iteration CG input, zero initial image, normalized operator and 512 PDHG iterations
- Positive C60 map predicted once; only the final parameter map is scaled
- Common [0,1] image display; metrics use unmodified reconstructions

## Sweep and search
- 12 displayed values: 0.01, 0.05, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 0.9, 0.95, 1.0
- Ternary search on linear rho in [0.0001, 10], 26 iterations, 67 unique evaluations including diagnostics and final probes
- Final bracket [0.38340938216863885, 0.3834753850553407], width 0.00006600288670183252
- Selected midpoint rho 0.3834423836119898
- PSNR 29.4083061218 dB, SSIM 0.8417503834, HaarPSI 0.6814417839
- PSNR is numerically tied within the final bracket at the available floating-point precision
- This is a single-image, ground-truth-guided oracle study, not unbiased test-set parameter tuning
- Ternary search assumes unimodality. Diagnostics do not prove global optimality; 1e-4 refers to bracket width, not PSNR error

## Deliverables
- One-page PDF with 18 panels: GT, four archived baseline methods, 12 C60 scales and ternary-selected C60
- metrics.csv with PSNR, SSIM, HaarPSI and MSE for all 17 reconstructions
- ternary_selected_reconstruction.npz with selected reconstruction, GT, measurement, mask and C60 map
- rho_sweep_reconstructions.npz and rho_sweep_metrics.csv for the 12-scale experiment
- frozen_case.npz and frozen_case_provenance.json for exact SHA256 acquisition replay
- summary.json, evaluations.csv, ternary_iterations.csv and ternary_selection.json for the full search trace
- Four reproducibility scripts stored under scripts/experiments/pcm/

All four baseline reconstructions are reused from the frozen benchmark. None is newly tuned or reconstructed.
