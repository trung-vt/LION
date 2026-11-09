# Bad images for UTV - frozen 23-case subset (2026-11-09)

This is the current **bad images for UTV** dataset. All **23** cases belong
to the frozen 450-image held-out PCM test set.

| Category | Image identifiers | Count |
| --- | --- | ---: |
| Reference | 0021, 0066, 0188, 0199, 0260, 0357, 0727, 0793 | 8 |
| CIGS | 0254, 0710, 0727, 0805, 0821, 0873 | 6 |
| Perovskite | 0021, 0031, 0079, 0098, 0187, 0260, 0288, 0292 | 8 |
| Silicone | 0767 | 1 |
| **Total** | | **23** |

Perovskite_0391 was excluded because it is outside the frozen held-out test
set and does not have a stored 20% NPL-default acquisition.

## Measurements, methods and outputs

All cases use existing 20% multilevel Hadamard acquisitions with NPL-default
PCM noise. Fourteen cases have archived acquisitions and reconstructions.
Nine additional cases reuse verified baseline and GS arrays together with
SHA256-matched deterministic acquisition replay and cached ADMM tuned-5
reconstructions. No fresh noise was sampled, and no solver was rerun to make
this 23-case revision.

The 23-page PDF compares ground truth, zero-fill, SPGL1-BPDN,
PnP-ADMM-tuned-5, GS-PnP-tuned-5, scalar TV and original UTV C60.
The display uses a common [0,1] range. PSNR, SSIM and HaarPSI are
calculated on the unmodified reconstruction arrays.

The CSV has 138 rows (23 cases x 6 reconstructed methods) and is unchanged
from the verified 23 cases in the previous comparison. The manifest
contains all case identifiers, measurement/pattern SHA256 hashes,
parameters and acquisition provenance.

## User-specified failure observations

- Reference_0260: all methods fail to recover lines.
- Reference_0793: all methods fail.
- Silicone_0767: zero-fill retains vertical lines that other methods lose.

These are observations for further investigation, not verified causal claims.
The purpose is to investigate UTV C60 detail and contrast loss, including
whether under-scaling the parameter map recovers fine structures.

## Reproducibility

- Generator: scripts/experiments/pcm/bad_utv_23_20261109.py
- tuni results: /store/LION/ttv22/pcm/results/pcm/bad_images_for_utv_20261109_v3_23
- PDF: PCM_Bad_UTV_23_20pct_NPL_7_Methods.pdf
- Metrics CSV: PCM_Bad_UTV_23_20pct_NPL_7_Methods_metrics.csv
- Manifest: bad_images_for_utv_dataset_manifest_23.json

This comparison is a baseline. It is not the UTV multiplier sweep.
