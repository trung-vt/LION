# Three synthetic CIGS images with long, thin lines

This experiment repeats the five-step PCM Bregman study on three additional
**ground-truth PNG images** from the synthetic dataset. The lines are longer
than the 33-pixel one-pixel-wide line in `CIGS/image0674.png`.

## Samples and line locations

| File | Feature | Approximate full extent | Width |
| --- | --- | --- | --- |
| `CIGS/image0985.png` | Vertical dark line centred around x = 167 | y ≈ 24–229 (205 px) | ~3–4 px |
| `CIGS/image0917.png` | Vertical dark line centred around x = 88 | y ≈ 27–232 (205 px) | ~3–4 px |
| `CIGS/image0864.png` | Horizontal dark line centred around y = 167 | x ≈ 27–231 (204 px) | ~3–4 px |

The locations and approximate widths were checked against the *original
8-bit PNG pixel values*. To exclude border artefacts, the line-contrast
metric uses the 173-pixel interior interval `[40, 213)` and compares the
mean of a three-pixel-wide core against flanks 7 pixels away. This
contrast is only a **local feature metric**, not a complete measure of
line preservation.

## Reconstruction protocol

The experiment uses the existing
`SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS` pipeline, restricted to one
CIGS sample and one trial at a time, with:

- Image size: 256 × 256, imported with the original
  `Image.convert("L") / 255` scaling.
- Operator: 20% multilevel subsampled, **normalised Hadamard**;
  coarse `J = 6`.
- Noise: `PCMNoiseSimulation.npl_default` for synthetic acquisition.
- U-TV model: supervised PCM checkpoint
  `20260920_025657_TV/checkpoints/best_val_loss_s0.20_cJ6.pt`,
  loaded with its `resolved_config.yaml`.
- Inner solve: original 256 PDHG iterations; zero initial image,
  configured CG input, unchanged network weights and PDHG parameters.
- Existing comparison methods: zero-filled, SPGL1, GS-PnP DRUNet.

Two versions of **five outer iterations** are compared using the same
measurement `y` and sampling mask for each sample:

1. **Fixed map:** estimate the learned TV parameter map once from
   `CG(y)`, then reuse it.
2. **Re-estimated map:** at each outer iteration form
   `d_k = y + b_(k-1)`, compute `CG(d_k)`, estimate the TV map anew,
   solve with `PDHG(d_k, lambda_k)`, and update
   `b_k = b_(k-1) + y - A x_k`.

The first iteration was verified to match exactly between the two
variants for each sample.

These five steps are **diagnostic**, rather than discrepancy-stopped.
Using the realised synthetic noise norm, the discrepancy rule with
`tau = 1.1` would stop at iteration 1 in all three experiments.
Re-estimating the map changes the regulariser between iterations;
fixed-regulariser Bregman convergence theory does not directly apply.

## Reproduce on tuni

From the `LION` repository on `experiments_10`:

```bash
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for id in 985 917 864; do
    python scripts/experiments/pcm/bregman_cigs_long_lines/run.py --sample-id "$id"
    python scripts/experiments/pcm/bregman_cigs_long_lines/render.py --sample-id "$id"
done
python scripts/experiments/pcm/bregman_cigs_long_lines/overview.py
```

The original experiments and the `experiments_10` library code are
unchanged. New result files are saved under:

```text
/store/LION/ttv22/pcm/results/20261024_cigs_long_lines_three_samples/
├── image0985/
├── image0917/
├── image0864/
├── three_samples_full_comparison.png
├── three_samples_line_crops.png
└── three_samples_summary.csv
```

Each image's directory contains `fixed/` and `reestimated/`
iterations 01–05, ground truth, baselines, full-resolution NumPy arrays,
TV maps, shared-scale error plots, cross-sectional and longitudinal line
profiles, `metrics.csv`, and `protocol.json`.

The learned checkpoint was trained on noiseless synthetic measurements,
whereas these tests use simulated NPL measurement noise. PSNR, SSIM and
HaarPSI should be interpreted accordingly.
