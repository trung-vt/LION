# PCM plug-and-play tuning using SSDU measurement holdout
Prepared: 2026-11-08. Status: preparation only; no SSDU hyperparameter sweep or 450-image evaluation has run.

## Purpose and honest terminology

Select global reconstruction hyperparameters using noisy acquired PCM measurements only, without ground-truth images. Save separate 5-image and 50-image parameter selections for each of PnP-ADMM-DRUNet and GS-PnP-GS-DRUNet. Evaluate them alongside seven existing baselines on the same 450 held-out-test acquisitions.

The previous oracle-tuned PnP settings were selected using ground-truth quality on VALIDATION images, then frozen before test. They are supervised-validation references, not per-test-image oracle fits. The 20% image-specific scalar-TV oracle used in a separate CIGS0279 experiment is NOT the dataset-wide TV baseline here. Use TV tuned on validation data, lambda=0.011, 1024 iterations, for the 450-image comparison.

The older 450-case test results have already been examined elsewhere in the research project. This is therefore a retrospective comparison; prevent any additional test-driven method selection. For a genuinely untouched prospective confirmation, collect new measurements or reserve a new blind evaluation set.

## Source protocol and controls

Image size 256 x 256, J=8; 65,536 possible Hadamard patterns. Every validation acquisition is already archived at 20% (13,107 measurements), normalized PCM operator, multilevel sampling with coarse_J=6, one NPL-default noisy acquisition, float64 archive. Validation permutation positions 901 to 910, 10 images in each of CIGS, Organic, Perovskite, Reference, Silicone.

Frozen source:
  /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final

Verified: the referenced source archive contains the 50 validation acquisitions only (test_count=0); this preparation therefore does not even require opening the old test archive.

Only load the measurement and sampled_indices arrays in SSDU code; do not read targets, PSNR, or test images. Cast the measurement to float32 only for solvers if required, retaining the full precision for the validation residual. The measurements and sampled row IDs must remain unchanged and checksum-verified.

Pilot5: one image per category, selected solely from preset permutation positions:
  CIGS 901 -> 0733; Organic 903 -> 0008; Perovskite 905 -> 0283; Reference 907 -> 0768; Silicone 909 -> 0589.
Full50: all validation positions 901..910 in each category. Pilot5 is a subset of Full50.

Both stages use exactly the same predeclared candidate configurations and fold assignment. The 5-image result is frozen BEFORE the 50-image result is selected and is never overwritten.

## SSDU measurement splitting

For each image, split only the 13,107 ACQUIRED measurements. Retain the first 4096 low-frequency patterns in the reconstruction input in every fold. Split the remaining 9011 acquired fine-scale patterns into four held-out folds. Use deterministic, level-stratified contiguous blocks of 128 indices, assigned among the four folds. The partition uses mask indices and fixed seeds, never image quality.

Each fine-scale measurement appears in one and only one held-out fold. In fold k, the fit operator sees the 4096 coarse patterns plus three quarters of the fine patterns. The validation operator sees only the remaining fine patterns; fit and heldout measurements are disjoint.

For candidate parameters theta on validation case i and fold k, compute:
  xhat(theta,i,k) = R_theta(A_fit(i,k), y_fit(i,k))
  l(theta,i,k) = mean((A_hold(i,k) xhat(theta,i,k) - y_hold(i,k))**2)

Average the four fold losses for an image, then average images equally within each of five categories and average the five category means equally. This is the sole primary hyperparameter selection criterion. Make ties deterministic using a predeclared tolerance and optionally solver runtime only when losses are effectively equal.

The loss must be computed in the same normalized PCM units as the original forward model. It must be computed only after reconstruction; validation rows must never be passed to the reconstruction operator.

## Caveats and required audits BEFORE tuning

1. PCM NPL noise can be temporally correlated. Training and held-out noise are not guaranteed independent. Measurement holdout is therefore a useful heuristic, not an unbiased image-MSE estimator. Examine residual correlations using noise calibration or repeat acquisitions if available, without using image GT. Block-stratified folds reduce adjacent-row leakage relative to fully interleaved row splits, but do not remove global bias/drift correlations.
2. Each fit fold has about 16.6% of full Hadamard rows rather than the final 20%. PnP data-fidelity and proximal scaling must be audited. With a summed half-squared L2 loss, changing the number of rows changes the relative regularisation strength; do not silently rescale rho, lambda or tau. Decide and record one fixed fold-to-full transfer convention before tuning. The prepared first version declares unscaled normalized fidelity and requires this audit before experiment execution.
3. The tuned-5 and tuned-50 parameter grids and stopping-rule definitions must remain identical. Do not build the full50 search space by choosing only the 5-image winner's neighborhood; that would confound the comparison.
4. GS-PnP must use the paper final z output for all SSDU and oracle baselines. The separate historical GS v2 method returning native x is NOT interchangeable. Keep lambda, sigma and tau0 independently configurable. Verify forward and gradient consistency with the normalized PCM operator.
5. No end-to-end denoiser weights are learned. This is hyperparameter-only measurement cross-validation, inspired by SSDU, not network training or strict equivalence to MRI SSDU.
6. Prior GT-based tuning is known to us. A fresh unsupervised study cannot erase those prior insights. Predeclare this broad parameter grid and never rank/select using GT.

## Frozen candidate grids

PnP-ADMM: rho in {0.1,0.25,0.5,1,2}, denoiser sigma in {0.03,0.05,0.075}; 15 combinations. Fix 20 minimum iterations, 100 maximum, residual abs_tol=1e-4, rel_tol=1e-3, five consecutive passes. Use pretrained DRUNet and unchanged normalized PCM data proximal.

GS-PnP: lambda in {0.01,0.03,0.1}; sigma in {0.03,0.05,0.075}; tau0 in {10,33.3333333333}; 18 combinations, independent Cartesian grid. Fix up to 100 accepted steps, 20 backtracking attempts, gamma=0.1, eta=0.9, DeepInv stopping eps=1e-6. Use pretrained GS-DRUNet and final paper z output.

Both grids are initial, frozen predeclared choices. Expanding a grid after reviewing test performance is prohibited. Any revised protocol must be separately versioned and must not silently replace tuned-5 or tuned-50.

## Step-by-step execution

Step 0: Verify source archive and freeze candidate grid, pilot case list, acquisition checksums, and four fold masks for all 50 validation images. Save as immutable prepared files, with no reconstruction. DONE when preparation tests pass.

Step 1: Implement measurement-only reconstruction adapters for ADMM and GS-PnP. Perform tiny-sample correctness tests, including data-fit/holdout disjointness, correct split operators and the GS final z output. Audit summed-fidelity scale and correlated-noise leakage. Fix the solver normalization convention BEFORE any SSDU search. Require deterministic CuBLAS and cuDNN algorithms: otherwise GS-PnP backtracking decisions change between nominally identical runs. Freeze the numerical code hashes.

Step 2: Pilot5, for each of 15 ADMM and 18 GS configurations: reconstruct across 5 validation images and 4 heldout folds. Save per-fold MSE, per-category summary, stopping diagnostics and run times. Select one theta for ADMM and one for GS using only the predefined measurement loss.

Step 3: Freeze pilot5 choices permanently as:
  PnP-ADMM-tuned-5 and GS-PnP-tuned-5.
Save selection.json, config and scores, candidate grid and source hashes. No later 50-image result may modify these selections.

Step 4: Full50, evaluate the SAME 15+18 grid across all 50 validation images with the SAME frozen folds and selection loss. Do not use pilot5 winner to narrow the candidate set. Save all per-case/per-fold scores and per-category diagnostics.

Step 5: Freeze final50 choices independently as:
  PnP-ADMM-tuned-50 and GS-PnP-tuned-50.
Record selections, rank tables, ambiguity/ties and all checksums before touching test-set reconstruction metrics.

Step 6: Reconstruct every one of 450 test images (90/category), at the same FIXED 20% NPL-default acquisition. For each selected SSDU config, run once using ALL acquired measurements; there is no holdout at test. Verify measurement and pattern hashes against the existing matched 450-image benchmark. Reuse numerical baseline results only when same measurements, method variants and outputs are independently verified; otherwise rerun methods.

Step 7: Compare all ELEVEN methods in the SAME figures and tables:
  1 zero-filled
  2 SPGL1-BP
  3 SPGL1-BPDN (frozen non-debiased preset)
  4 PnP-ADMM-tuned-oracle (GT-selected on validation, not test oracle)
  5 GS-PnP-tuned-oracle (frozen Step6 paper-final-z method)
  6 scalar TV validation-tuned (lambda=0.011, 1024 iterations)
  7 original UTV-C60 (epoch 60; no map multiplier)
  8 PnP-ADMM-tuned-5
  9 PnP-ADMM-tuned-50
 10 GS-PnP-tuned-5
 11 GS-PnP-tuned-50

Compute PSNR, SSIM, HaarPSI, MSE, mean and standard deviation for each category and all 450 images; plot comparable 11-method box plots and per-category score comparisons. Show paired gains relative to the two oracle-tuned baselines. Also report reconstruction runtime per image, and distinguish offline parameter search time from inference time.

Step 8: Prepare full reproducibility archive: exact source code, selected configs, fold hashes, measurement fingerprints, candidate losses, per-image scores, figures, visual reconstructions and report. State limitations: correlated noise; null-space unobservable detail; evaluation test set previously inspected in prior project research.

## Suggested output directories

Persistent git worktree:
  /home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108
Experiment scripts inside it:
  scripts/experiments/pcm/ssdu_pnp_tuning_20261108

Data/results outside git:
  /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1
  prepared/             immutable 50 validation case folds, candidate grid and pilot subset
  pilot5/               to be populated after Step 2; frozen selections after Step 3
  final50/              to be populated after Step 4; frozen selections after Step 5
  test450/              only after BOTH selections are frozen
  figures/              final 11-method comparisons

This preparation DOES NOT initiate expensive GPU runs and DOES NOT open the test set.

## Original reference

Yaman et al., Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data, Magnetic Resonance in Medicine (2020), DOI 10.1002/mrm.28378. See the Methods section on partitioning acquired measurements into disjoint reconstruction and loss sets, and the discussion of test-time reconstruction using all acquired measurements.
  https://onlinelibrary.wiley.com/doi/10.1002/mrm.28378
