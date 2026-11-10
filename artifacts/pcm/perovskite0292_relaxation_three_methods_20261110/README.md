# Perovskite 0292: TV, TGV-2 and TGV-3 relaxation comparison

Six-page PDF with actual raw reconstructions under both relaxation rules.
The only algorithmic difference is the primal interpolation vs canonical
Chambolle-Pock extrapolation. The LION/MRPro code is NOT changed.

Frozen 20% NPL-default measurement; fixed GT-selected scalar weights.
Step sizes tau=sigma=0.18. All final metrics at 8192 iterations, all
nine checkpoints also supplied in the CSV and archive.

At 8192 iterations, objective decreases under canonical CP:
TV 0.0055%, TGV2 1.43%, TGV3 4.27%.
Absolute PSNR changes are below 0.00005 dB in all methods.

For TGV2 and TGV3, canonical CP at 512 iterations achieves lower
objectives than LION at 8192 iterations. For TV this happens
at 1024 canonical iterations, not 512.

Per-step GPU runtime:
TV ~0.96 ms, TGV2 ~1.49 ms, TGV3 ~2.03 ms.
The extrapolation difference adds negligible per-step overhead.
Different methods have different objectives, so cross-method
objective values should not be compared.

The ZIP contains all 54 per-method/rule/checkpoint measurements,
all six final raw reconstructions, experiment code and one-line
local solver changes. No generalisation claims for this
single-image, GT-selected experiment.
