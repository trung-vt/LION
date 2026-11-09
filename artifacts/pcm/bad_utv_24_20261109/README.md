# Bad images for UTV - 24 nominated identifiers

The PDF is one page per nominated identifier. It compares ground truth,
zero-fill, SPGL1-BPDN, PnP-ADMM-tuned-5, GS-PnP-tuned-5, scalar TV,
and the original UTV C60 checkpoint, with a shared [0,1] display range.

For 23 cases, all six reconstruction methods use the exact frozen 20%
Hadamard / NPL-default benchmark acquisition. Fourteen raw cases were
already archived; nine new cases reuse validated test45 baseline arrays,
GS tuned-5 raw arrays and an SHA256-verified exact deterministic replay for
ADMM tuned-5. The PDF reports the original frozen metrics.

**Perovskite_0391 is NOT in the frozen 450-image held-out test** and has no
stored 20% NPL-default measurement. It remains in the named 24-case dataset.
Page 23 shows its actual ground truth, and all six reconstruction panels
clearly say that the acquisitions/reconstructions are not available. No
fake metrics or new noise realisations were created.

User-specified failure observations are annotated for Reference_0260,
Reference_0793 and Silicone_0767.

Source generator: `scripts/experiments/pcm/bad_utv_24_20261109.py`.
Data and experiment output on tuni under:
`/store/LION/ttv22/pcm/results/pcm/bad_images_for_utv_20261109_v2_24`.
