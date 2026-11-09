# 23 bad images for UTV: C60 learned map multiplier sweep

## Protocol
- Exactly the 23 images selected in the final "bad images for UTV" manifest (not 24, no Perovskite 0391).
- 8 Reference, 6 CIGS, 8 Perovskite, 1 Silicone.
- Frozen 20% multilevel Hadamard measurement and NPL-default PCM noise for each image. 14 raw stored acquisition files, 9 exact deterministic replays verified against both original measurement and pattern SHA256.
- UTV C60 checkpoint epoch 60, normalized PCM operator, 20-step CG input, zero initial image, 512 PDHG iterations.
- Predict positive C60 map once, then multiply it by global rho in [0.0001,10].
- Per-image, ground-truth PSNR oracle tuning with standard ternary search on linear rho.
- Stop when final rho bracket width <= 0.01 (not PSNR error). Around 50 reconstruction evaluations per image including diagnostics and bracket probes.
- Unimodality is needed for global-optimum guarantee; diagnostics do not prove it.
- Select best observed bracket PSNR unless a candidate outside performs better, then flag it. None of these 23 selected candidates lay outside its final bracket.
- PnP-ADMM-tuned-5 reuses frozen 5-sample tuned selections, without GT-based retuning for each image.
- C60 rho=1 uses stored reconstruction, verified against same checkpoint and frozen metrics.
- All panels share [0,1] display; PSNR/SSIM/HaarPSI computed on raw images.

## Results
- Mean C60 PSNR across 23: original 30.27625 dB; oracle scaled 30.58894 dB (+0.31269 dB).
- Mean PnP-ADMM-tuned-5 PSNR 33.78142 dB. It beats oracle-scaled C60 on all 23 examples.
- Mean scaling PSNR gain by category: Reference +0.13252 dB, CIGS +0.01540 dB, Perovskite +0.73349 dB, Silicone +0.17146 dB (1 case).
- This deliberately selected bad-image subset is not representative for generalisation evaluation.

## Deliverables
- Comparison PDF: 23 pages, each with GT, PnP-ADMM-tuned-5, C60 rho=1, C60 best rho.
- all_23_summary.csv: all selected rho values, final brackets, PSNR/SSIM/HaarPSI and improvements.
- summary.json: aggregated stats and audit flags.
- optimal_reconstructions_23.npz: all selected images and multipliers by case.
- Full results ZIP: all per-image evaluations, ternary iterations, GT, ADMM, C60 base and optimum, maps, summaries and scripts.
- Source scripts in scripts/experiments/pcm/bad_utv23_c60_ternary_width0p01_20261109.py and bad_utv23_c60_ternary_width0p01_render_20261109.py.

All per-case raw files remain on tuni under the root above in the cases/ directory.
