# Perovskite 0292 relaxation checkpoints: TV, TGV-2, TGV-3

Frozen synthetic 20% multilevel Hadamard PCM acquisition, NPL-default noise.
One GT image, same sampling indices, fixed GT-tuned scalar weights,
same zero initial primal and dual fields, tau=sigma=0.18.
Existing LION PDHG: lerp(old,new,1)=new.
Canonical Chambolle-Pock PDHG: new+(new-old) with theta=1.
Only the relaxation expression changes. LION/MRPro code is unchanged.

All 18 reconstructions at 256/512/1024 iterations reproduce the archived
PSNR, SSIM, HaarPSI, MSE and objective figures. Each new independent
run includes measured GPU wall-time, image signed error (x-GT),
and measurement-domain residual (A x-y).

The 17-page updated PDF consists of 6 original experiment-report pages,
1 checkpoint metrics page, then 3 pages for each checkpoint
(full images with matched crops, signed spatial errors and measured
Hadamard-index residuals), ending with objective vs runtime comparisons.

PSNR at 1024 iterations, existing LION -> canonical CP:
TV 30.369192 -> 30.369751 dB;
TGV-2 30.439955 -> 30.441168 dB;
TGV-3 30.471500 -> 30.471724 dB.

Run time at 1024 iterations, existing LION -> canonical CP:
TV 0.989 -> 0.996 s;
TGV-2 1.538 -> 1.544 s;
TGV-3 2.089 -> 2.098 s.

Only sampled Hadamard coefficients are shown in measurement residual maps.
Their 256x256 index-grid arrangement is NOT a spatial image.

Artifacts: updated PDF, metrics CSV, compact NPZ of raw arrays,
protocol JSON, reconstruction replay script, PDF rendering script and full ZIP.

References:
Chambolle and Pock (2011): https://doi.org/10.1007/s10851-010-0251-1
Bredies, Kunisch, Pock (2010): https://doi.org/10.1137/090769521
MRPro PDHG: https://docs.mrpro.rocks/_autosummary/mrpro.algorithms.optimizers.pdhg.html

No LION source files were modified.
