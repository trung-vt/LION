# Perovskite 0031: C60 ternary search and matched visual comparison

- Frozen synthetic PCM test case Perovskite_0031, 20% multilevel Hadamard acquisition, NPL-default noise
- C60 epoch-60 checkpoint, same CG input, fixed learnt positive TV parameter map, 512 PDHG iterations
- Search linear rho in [0.0001, 10] using standard ternary search
- Stop when final interval width in rho is <= 0.0001, assuming PSNR is unimodal
- Diagnostic evaluations at 13 rho values show no obvious competing peak, but do not prove global unimodality
- 25 ternary iterations, 66 unique rho evaluations including diagnostics and probes
- Final interval: [0.4325622022727993, 0.43266120660285196], width 0.00009900433005266551
- Selected image uses midpoint rho=0.4326117044378256 (inside final bracket)
- PSNR 31.1550159454 dB, SSIM 0.7925548553, HaarPSI 0.6995180249
- PSNR is numerically tied within the final bracket at the available floating point precision
- Distinguish rho interval accuracy from a mathematical global-optimum or PSNR error guarantee
- Compared with the same archived Zero-fill, SPGL1-BPDN, PnP-ADMM-tuned-5, GS-PnP-tuned-5 and the previous 12 C60 scales
- Display [0,1] for all panels; metrics use raw unmodified reconstructions
- This is per-image, ground-truth oracle tuning, not unbiased population-level performance

Contents:
- PDF: one-page 18-panel visual comparison
- metrics.csv: all 17 reconstruction metrics
- ternary_selected_reconstruction.npz: selected reconstruction, GT, mask, measurement, lambda map
- ternary_selection.json: selected rho, PSNR, final interval
- summary.json, evaluations.csv, ternary_iterations.csv: full search audit
- Search and comparison source scripts in scripts/experiments/pcm/
