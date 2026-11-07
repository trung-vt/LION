# CIGS 0279: 20% versus 10% NPL-noisy PCM comparison

Completed 7 November 2026 on tuni RTX 4090.

- `comparison_2pages.pdf`: final landscape PDF, exactly two pages and eight panels each.
- `page1_20pct_metrics.json`, `page2_10pct_metrics.json`: PSNR, SSIM, HaarPSI and MSE on raw reconstruction arrays.
- `page2_10pct_measurement.npz`: one newly simulated 10% acquisition, ground truth, mask, noise realisation (plus swap flags), and RNG seeds.
- `reconstructions/20pct`, `reconstructions/10pct`: eight raw arrays per sampling rate.
- `protocol.json`: frozen parameters, numerical sources, provenance, hashes and seeds.
- `verification.json`: 20% replay checks against historical 450-image metrics.
- `source/`: exact scripts used for preflight, classical/PnP runs, pinned epoch60 C60, and PDF rendering.

All methods on each page use one shared acquisition. At 20%, the measurement is reused bitwise from the original fixed-NPL experiment. At 10%, the acquisition is newly generated and frozen. The 20%-selected PnP and TV parameters are applied unchanged at 10%. TV uses an image-specific test oracle at 20%; this is not an unbiased validation-tuned baseline. The GS-PnP selected configuration is the frozen Step6 method with paper final z output (not the earlier separate GS-v2 450-image tune). No lambda-map multiplier is applied to C60.
