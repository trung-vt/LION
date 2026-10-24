# Bregman experiments on three curved / branching crack-like PCM samples

The goal is to test whether five Bregman outer iterations improve recovery
of **non-straight, narrow crack-like defects**, rather than only the straight
lines examined previously.

## Selected synthetic ground-truth images

All three come directly from the synthetic PCM **Organic** dataset
(256 × 256 grayscale PNGs):

| Sample | Visually identified feature | Analysis ROI [x0,y0,x1,y1), pixels |
|---|---|---|
| Organic/image0497.png | Irregular diagonal thin feature near upper-left corner | [8,15,143,124] |
| Organic/image0455.png | Curved thin feature near lower-left corner | [14,157,101,250] |
| Organic/image0587.png | Irregular/branching thin features near lower-right corner | [134,88,247,246] |

These are image-analysis and visual **crack-like candidate** selections. The
dataset does not supply pixel-exact crack annotations for these samples.
The ROI boxes include surrounding material and are *not* crack segmentation
masks. In particular, patch PSNR/SSIM/HaarPSI should not be described as
crack-specific structural-recovery scores.

## Protocol

Use the original synthetic PNG dataset box-plot preset:

- Normalised PCM Hadamard operator, 20% multilevel sampling, coarse J=6.
- Simulated noise: NPL default, with existing sampling/noise seed logic.
- Existing supervised PCM U-TV checkpoint
  `/store/LION/ttv22/pcm/results/pcm/pcm_supervised/20260920_025657_TV/checkpoints/best_val_loss_s0.20_cJ6.pt`.
- Same `resolved_config.yaml`, configured CG model input, zero initial image,
  256 PDHG inner iterations, no altered model weights or solver settings.
- Also reconstruct zero-filled, SPGL1, and GS-PnP DRUNet using the same
  measurements and existing PCM method configurations.

Five outer iterations are compared, both using
`d_k=y+b_{k-1}` and `b_k=b_{k-1}+y-Ax_k`:

1. **Fixed TV map**: estimate from `CG(y)` once and reuse for all five steps.
2. **Re-estimated TV map**: recompute `CG(d_k)` and the learned spatial TV
   map at the *beginning of every step*, before the PDHG reconstruction.

The implementations assert exact equality of the first reconstruction
and TV map between the two variants.

Metrics: full-image and selected-ROI PSNR/SSIM/HaarPSI/MSE; original noisy
measurement residual, realised synthetic oracle noise norm, lambda-map
statistics, and synchronized GPU timings. Diagnostic plots have consistent
intensity/colorbar scales and include crack-region crops, TV maps, signed
differences, and error images.

**Caveat**: Each five-step run is deliberately diagnostic, not automatically
discrepancy-stopped; the oracle discrepancy threshold is satisfied at step 1.
Re-estimated TV maps change the regulariser, so the fixed-objective
Bregman convergence theory does not directly apply. The network also performs
a finite number of PDHG steps rather than certified exact inner minimisation.

## Reproduce on tuni

From the LION repository on `experiments_10`:

```bash
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for id in 497 455 587; do
  python scripts/experiments/pcm/bregman_curved_cracks/run.py --sample-id "$id"
  python scripts/experiments/pcm/bregman_curved_cracks/render.py --sample-id "$id"
done
python scripts/experiments/pcm/bregman_curved_cracks/overview.py
```

Results:

```text
/store/LION/ttv22/pcm/results/20261024_pcm_curved_cracks_three_samples/
  Organic_image0497/
  Organic_image0455/
  Organic_image0587/
  three_cracks_full_comparison.png
  three_cracks_roi_comparison.png
  three_crack_ground_truths.png
  three_cracks_summary.csv
```

Each image directory contains fixed/re-estimated 5-step reconstructions,
ground truth, matched baselines, the five TV weight maps, region comparisons,
shared-scale error plots, `metrics.csv`, and `protocol.json`.

The repository's PCM library code and existing experiment results are untouched.
The experiment scripts are intended to remain uncommitted until reviewed.
