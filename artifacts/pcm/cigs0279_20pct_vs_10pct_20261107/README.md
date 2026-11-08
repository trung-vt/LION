# CIGS0279 20% versus 10%: corrected colour-rendered experiment

Correction 8 November 2026: same original PCM observations, reconstructions, metrics and methods, with corrected AFM-hot colormap range. The previous PDF used cmap_max=1.0, mapping bright values to white; this version uses cmap_max=0.8 and maps maximum intensities to yellow (RGB approximately 255,255,153), as shown in the supplied reference image. vmin=0 and vmax=1 in both versions. This correction affects display only.

`comparison_2pages.pdf`: repaired PDF, two pages, eight panels per page.

`source/03_render.py`: corrected reproducible renderer.

`source/03_render_previous_cmap1p0.py`: preserved earlier renderer for audit only.

`reconstructions/`: 16 raw numpy arrays, copied bitwise from v1.

`page2_10pct_measurement.npz`: preserved 10% acquisition from v1.

`page1_20pct_metrics.json`, `page2_10pct_metrics.json`: unchanged original metrics.

`protocol.json`, `verification.json`: correction and integrity metadata.

The other source scripts are the original v1 acquisition/reconstruction scripts; use `source/03_render.py` in this v2 folder to render the corrected PDF. The parameters for 10% were not retuned.
