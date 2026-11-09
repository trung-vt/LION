# Frozen PCM tune-1 ternary search plan, 9 November 2026

State: prepared only. No PnP calibration or test evaluation launched.

1. Sample selection: 5 random validations and 9 random test images in each
   of five PCM categories. Save a fixed seed, source paths and hashes. DONE.
   Test set 45 is a subset of the original test set 450; neither changes.
2. Score candidate parameters from the four original held-out Hadamard folds.
   No image GT and no test quality metrics are available during tuning.
3. Bounded ternary search on log parameters, with 3 contractions per
   coordinate and 2 coordinate passes in opposite order.
   ADMM: rho [.03,4], sigma [.015,.18].
   GS: lambda [.005,.4], sigma [.015,.15], tau0 [3,75].
4. For each category independently tune 5 validation images, resulting
   in 25 distinct parameter searches per method, 50 searches total.
   Max unique candidate runs per calibration: 25 ADMM, 37 GS.
   Max SSDU fold solver runs: 2,500 ADMM and 3,700 GS.
5. Freeze ALL 50 parameter-selection files before test45 image access.
   Save all candidate trials, values, heldout losses and failure flags.
6. For each selected configuration reconstruct nine category-matched test
   images, producing 225 reconstructions per PnP method. Use all original
   sampled measurements of each test image, without withholding folds.
7. Compare against 10 old method labels: zero fill, SPGL1-BP, BPDN,
   PnP-ADMM supervised-validation oracle, GS-PnP Step6 oracle,
   TV validation-tuned, UTV-C60 original, ADMM SSDU-5,
   GS SSDU-5 and expanded GS SSDU-5.
   Reuse old scores only after checking measurement and pattern SHA256.
   Replay missing raw arrays under identical original method settings.
8. Report PSNR/SSIM/HaarPSI/MSE and wall time per calibration and per
   test image; show mean and sample SD over nine distinct cases/category.
   For the overall tune1 score, average five calibration outcomes for each
   unique test image before averaging 45 test images.
9. Visualise each of 45 test cases on a 21-panel page: target, ten old
   method labels, five ADMM tune1 and five GS-PnP tune1 calibrations.
   Use AFM-hot capped 0.8 and fixed intensity [0,1], raw-image metrics.
10. Update the prepared six-section TeX report and compile PDF; archive
    selections, code, metrics, raw reconstructions, plots and the 45-page
    gallery in the new dedicated result directory.

Assumptions: one-dimensional coordinate sections of noisy SSDU risk are
unimodal. This does not guarantee a joint optimum. No test-driven tuning.
All evaluation is retrospective because prior 450-test metrics have been
inspected in the broader PCM research project.

Execute only after explicit instruction, using run_all.sh execute.
