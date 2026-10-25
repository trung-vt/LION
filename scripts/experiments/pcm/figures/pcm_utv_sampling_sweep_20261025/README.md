# Noisy U-TV: PCM sampling sweep (5%, 10%, 20%, 30%, 40%, 50%)

This is a **frozen checkpoint evaluation**, **not** a new training run.

## Deliverables

- [36-page A3 PDF report](utv_sampling_5_to_50_percent_report.pdf):
  metric summary and plots; full-resolution image panels and matched,
  enlarged crops for **three held-out test images per category**, plus two
  separately labelled CIGS thin-line examples.
- [Category and overall summary CSV](metrics_summary.csv):
  means and standard deviations across **10 held-out test PNGs per category**
  for MSE, PSNR, SSIM and HaarPSI.
- [Per-image metrics CSV](metrics_per_image.csv): exactly 300 records
  (5 categories x 10 test images x 6 ratios).
- [Individual PNG images](individual_png/): 119 individual panels, all
  rendered with common intensity scale 0 to 1.
- [Evaluation manifest](evaluation_manifest.json), plus the
  [random noise ranges](noise_variation_ranges.yaml) used by the test.
- [Supplementary line-example scores](supplementary_line_metrics_NOT_HELDOUT.csv)
  **excluded from all held-out averages**.

## Evaluation protocol and limitations

- Snapshot checkpoint: best_val_loss_s0.20_cJ6.pt, **epoch 5 / step 22500**,
  SHA-256: 94144e75ab63ecc7356dd0f9ccbc555dc213ab7796db971c547e7f387b49eea4.
  A verified **private** copy was used (the live checkpoint was *not* read during
  evaluation). Checkpoint weights are not included in GitHub.
- Evaluation GPU: RTX 4070 (12 GB); training on RTX 4090 was not changed.
- Network: 4-level U-Net, 128 initial filters, 512 PDHG iterations,
  shared map scale-factor key across sampling ratios, CG input.
- 5 synthetic PCM categories; test IDs corresponding to permutation
  positions 911, 920, 930, 940, 950, 960, 970, 980, 990, 1000:
  sample IDs 611, 377, 357, 331, 935, 249, 294, 617, 95, 288.
- Acquisition: existing normalised Hadamard PCM simulator; multilevel
  coarse_j=6 with sampling ratios from 5% to 50%.
  At 5%, less than 6.25% is sampled, so the existing sampler
  retains the J=5 low-frequency block and draws part of J=6 randomly.
- Noise parameters: each image and sampling ratio receives an **independently
  randomised** NPL-style parameter draw from the ranges in the YAML file.
  Noise is simulated in the measurement domain, one deterministic draw
  per image/ratio (see seed in per-image CSV). **This sweep is not a
  noise-controlled isolation of the sampling effect.** In particular,
  40% need not be better than 30% under independent noise settings.
- Metrics: untouched LION compute_metrics implementation; PSNR/SSIM/HaarPSI,
  MSE. Means are per-image arithmetic means.
- Three visually illustrated *TEST* examples per category: image IDs
  611, 249, 288. The PDF adds CIGS/image0674 (TRAIN, known 1-pixel line)
  and CIGS/image0917 (VALIDATION, long line) for **qualitative** comparison.
  The latter two are not part of the quantitative TEST aggregation.
- A larger/full held-out set and repeated independent noise draws would be
  needed for high-confidence or controlled comparisons.

## Storage on tuni

Full arrays and the immutable copied checkpoint remain under
/store/LION/ttv22/pcm/results/20261025_utv_ratio_sweep_snapshot.
Training remains in its original tmux session. No library source changes,
checkpoint commits, or changes to the working training branch are part of
this results publication.

## Changing plot settings without rerunning GPU reconstructions

The report's figure settings are now independent of the training config:

- Plot-only YAML: [config/pcm/pcm_utv_ratio_sweep_render.yaml](../../../../../config/pcm/pcm_utv_ratio_sweep_render.yaml)
- Re-rendering script: [scripts/experiments/pcm/utv_ratio_sweep_20261025/report.py](../../utv_ratio_sweep_20261025/report.py)
- Exact settings used to render this publication: [render_config_used.yaml](render_config_used.yaml)

The YAML uses LION's existing log.final_result structure. In particular,
the reconstruction fields cmap, cmap_max, vmin, vmax, and cbar_tick_format
control the **ground truth and reconstructed PNGs, PDF panels, and PDF
colourbars together**. Current defaults are afmhot and common limits [0, 1].
Changing the display scale does not recalculate any reconstruction or metric.

On tuni, the evaluation arrays are stored outside GitHub in
/store/LION/ttv22/pcm/results/20261025_utv_ratio_sweep_snapshot/arrays.
With those arrays present, run:

    python scripts/experiments/pcm/utv_ratio_sweep_20261025/report.py \
      --render-config config/pcm/pcm_utv_ratio_sweep_render.yaml \
      --output-dir /store/LION/ttv22/pcm/results/20261025_utv_ratio_sweep_snapshot/rendered_configurable

This recreates the PDF and the 119 individual PNGs from saved arrays;
the evaluation GPUs and checkpoint are not needed. Choose another
output directory if you want to preserve earlier render variants.
