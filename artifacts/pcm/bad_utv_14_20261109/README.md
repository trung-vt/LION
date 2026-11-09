# Bad images for UTV (14 cases), fixed 20% NPL PCM

Images from the existing 450-case held-out test: Reference 0021, 0066,
0188, 0727; CIGS 0710, 0727, 0805, 0821, 0873; Perovskite 0031,
0079, 0098, 0260, 0288.

One page per case shows exactly seven panels: ground truth, zero-fill,
SPGL1-BPDN, PnP-ADMM-tuned-5, GS-PnP-tuned-5, scalar TV, UTV C60.

This is a visual comparison of pre-existing reconstructions, not a
regularisation-parameter sweep. No images, masks or noisy measurements were
regenerated. Measurement and pattern SHA256 match the archived acquisitions
and the frozen 450-case metrics. Each raw reconstruction has its original
archived MSE. Common display scale [0,1] with AFM hot; other numerical
metrics refer to unmodified (unclipped) reconstructions.

PnP-ADMM-tuned-5 uses the five-image SSDU selection, which happened to
match the earlier validation-selected ADMM parameters. This is not a new
test-oracle tuning. GS-PnP-tuned-5 uses the frozen SSDU-5 selection, not
an oracle/expanded GS selection. Scalar TV uses lambda=0.011 for 1024
iterations. UTV C60 is the original network at checkpoint epoch 60.

Source: scripts/experiments/pcm/bad_utv_14_20261109.py
Data and output stored on tuni under:
/store/LION/ttv22/pcm/results/pcm/bad_images_for_utv_20261109_v1/
