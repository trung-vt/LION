# Full comparison of 4070 and 4090 trainings — 50-image repeat

**28 October 2026 · LION supervised PCM U-TV reconstruction**

This report compares two 4-block × 128-filter U-TV models using 10 randomly selected held-out test images from each of five categories (50 images total). Each image is evaluated under the two prescribed PCM noise regimes, for 100 paired acquisitions. Measurements and sampling patterns match the earlier 450-image evaluation exactly.

The current validation-selected checkpoints are the **RTX 4070 model at epoch 17** and the **RTX 4090 model at epoch 23**. Their performance is compared on the same 100 acquisitions with the frozen previous checkpoints (4070 epoch 7 and 4090 epoch 22). The 4070-side experiment produced 200 new reconstructions; earlier results were reused, not recomputed.

## Files

- **[Complete 21-page report (PDF)](Full_comparison_4070_4090_50_images.pdf)** — interpretation, aggregate and per-category tables, bootstrap intervals, reconstruction examples, and error maps.
- **[LaTeX source](Full_comparison_4070_4090_50_images.tex)** — report source, using the nine zoomable PDFs in [figures/](figures/).
- **[Complete source, figures, and data bundle (ZIP)](Full_comparison_4070_4090_50_images_source_bundle.zip)** — includes CSV results, scripts, figures, and ten preselected qualitative examples in compressed NumPy format, but excludes model checkpoints.
- **[Per-acquisition metrics (CSV)](all_paired_results_old_new.csv)** — complete old and current metrics, 100 rows.
- **[Statistical summary (JSON)](summary_50.json)** — paired category-stratified bootstrap statistics, both noise regimes, sampling-ratio strata and win counts.
- **[Protocol and checkpoint hashes (JSON)](protocol_manifest_50.json)** — the exact random sample IDs, acquisition seed namespace and frozen checkpoint metadata.
- **[Previous checkpoint metrics, 50-image subset (CSV)](old_reference_metrics_subset.csv)** and five current_CATEGORY.csv files — supporting evaluation outputs.
- **Evaluation and reporting scripts:** evaluate_current_50.py, summarize_50.py, make_figures.py, build_report_50.py and launch_evaluation.sh. These are the actual run scripts, with tuni-specific absolute paths; adjust those paths and provide the pinned training code before rerunning elsewhere.

## Key results

| Checkpoint | MSE ↓ | PSNR ↑ | SSIM ↑ | HaarPSI ↑ |
|---|---:|---:|---:|---:|
| 4070 epoch 7 | 0.002717 | 29.597 | 0.7524 | 0.6615 |
| 4070 epoch 17 | **0.002445** | **29.975** | **0.7732** | **0.6803** |
| 4090 epoch 22 | 0.002594 | 29.497 | 0.7550 | 0.6446 |
| 4090 epoch 23 | 0.002549 | 29.131 | 0.7424 | 0.6454 |

These are averages across the same 50 test images and two equally weighted noise conditions. The 4070 checkpoint improved MSE by approximately 10%, while the 4090 checkpoint changed only modestly in MSE and declined in average PSNR and SSIM. The current MSE head-to-head 95% bootstrap confidence interval includes zero; see the report for nuance and per-category results.

**Test-set reuse qualification:** Images were unused for *training and validation*, but **not newly untouched test images**: all 450 held-out images were already evaluated in the first comparison on 27 October 2026. This repeat measures progress on fixed paired measurements, and must not be used for further test-driven model selection.

## Reproduction and provenance

- Experiments performed on tuni, with new inference on its RTX 4070.
- Original pinned source revision: 9188b0735843c96526537b6b9bfd834fd089f7cd.
- Full on-host experiment directory: /store/LION/ttv22/pcm/results/pcm/pcm_full_comparison_50_20261028/.
- Old checkpoints and current checkpoints are retained there, but **not committed to Git** (approximately 392 MiB each). The manifest contains SHA-256 hashes.
- Compile the report from this directory with latexmk -pdf Full_comparison_4070_4090_50_images.tex. The individual PDFs under figures/ are used as external images by LaTeX; the archive contains all assets needed to rebuild it.
- Figures use shared intensity and absolute-error scales, as explained in the report.
