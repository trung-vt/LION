# Fixed20 NPL default: AFM hot visual comparisons

The corrected 450-image synthetic PCM experiment is rendered with the shared LION AFM hot reconstruction colormap, not grayscale.

- [All 50 selected images (PDF)](Fixed20_NPL_50_Visual_Comparisons.pdf): 16 A3 pages, six views per image, PSNR, SSIM, and HaarPSI.
- [Full 450-image and five per-category boxplots (PDF)](Fixed20_NPL_Boxplots_By_Category.pdf): seven A3 pages.
- [15 separate AFM visual sheets (PNG)](visual_pages_afm/): three per category, ten images per category.
- boxplots_all_450.png and boxplots_[category].png: box-plot previews, not image reconstructions.
- all_450_metrics.csv, summary.json, protocol.json: unchanged full-set numerical data and settings.
- AFM_render_manifest.json: file hashes and publication metadata.

All images use image_style('reconstruction') from [LION's PCM figure style](../../LION/pcm/randomized/figure_style.py), configured as afmhot, vmin=0, vmax=1. The change affects image display only: source arrays and metrics are unchanged. Before-and-after SHA-256 hashes of the CSV, summary, and protocol matched.

Rendering code: [plots.py](../../scripts/experiments/pcm/fixed_utv_test_20261103/plots.py).

The test uses fixed 20% multilevel Hadamard sampling and fixed NPL default noise parameters. Masks and noise realisations are independently seeded by image. All methods see identical measurements for the same image.

These are simulated measurements of synthetic PCM images, not physical measured PCM data. Full-model comparisons do not isolate the effect of noise training.
