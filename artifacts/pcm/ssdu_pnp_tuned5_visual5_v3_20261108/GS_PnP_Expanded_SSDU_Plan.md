# Follow-up: GS-PnP SSDU sweep beyond the previous upper bounds

Prepared: 8 November 2026. Status: **plan only**. No new parameter sweep has been run.

## Objective

The initial *blind SSDU* 5-image selection was lambda=0.1, DRUNet sigma=0.075, initial tau0=100/3 (candidate gs_018). These were the **largest values of all three parameters in the original candidate grid**, so there is no evidence that the minimum of SSDU held-out loss lies within the interior. We therefore propose a *predeclared, validation-only* parameter expansion above these limits.

The first SSDU GS-PnP configuration did NOT match the Step-6 supervised-validation-tuned oracle in the 450-image retrospective test. This difference motivates careful method analysis, but **the 450 test-set reconstruction quality must not be used to choose new parameters**.

Important distinction: Lower held-out *measurement* MSE need not imply higher *image* PSNR. Correlated PCM noise, poor sensitivity to missing Hadamard directions, and denoiser over-smoothing are plausible limitations. A greater parameter value can also *worsen* image quality.

## Frozen test-independent data

Use precisely the original 5 validation images (one each from CIGS, Organic, Perovskite, Reference and Silicone), and their original 4 x fixed multilevel held-out Hadamard partitions, with all 4,096 coarse rows in the fit set.

Original SSDU archive:
  /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/prepared

The only selection criterion is the same category-balanced mean of per-fold held-out MSE as in the original SSDU experiment. Do not access GT, test metrics, or visual test-example results during selection. Save a fresh protocol, a separate study identifier and all source SHA256 hashes.

## Step 0. Baseline and numerical check

Reproduce the original gs_018 configuration under the same frozen-fold scoring and deterministic CUDA mode (CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN). Confirm matching held-out MSE and final paper-z output. Pin the same pretrained GS-DRUNet and the same solver implementation, maximum accepted steps = 100, max backtracking attempts = 20, gamma=0.1, eta=0.9, epsilon=1e-6. Verify no implicit coupling between lambda and tau0.

Check the actual effective gradient step lambda * tau at accepted iterates. Increasing both can make this product much larger, so track line-search rejection counts, final accepted tau, denoiser evaluations, and fraction of runs that terminate early.

## Step 1. Predeclared expanded sweep (27 configurations)

A three-dimensional Cartesian grid, with the previously selected boundary point retained as a control:

| Parameter | Values |
| --- | --- |
| lambda | **0.10, 0.20, 0.40** |
| denoiser sigma | **0.075, 0.10, 0.15** |
| initial tau0 | **100/3, 50, 75** |

27 candidate triples, including 26 previously untested combinations. Run all 27 on five validation images and four frozen SSDU folds: 540 fold reconstructions (20 per candidate). All configurations use **the same operator normalization and code**.

Record: category, image ID, fold, candidate, held-out prediction MSE, accepted iterations, stopping reason, line-search rejections/exhaustions, actual final tau, effective lambda * tau, and wall-clock time.

If a configuration is numerically invalid or repeatedly fails backtracking, record failure explicitly and define a *predeclared* invalid-candidate policy (exclude it with reasons; do not silently substitute another result). Never adjust parameters after seeing test-image metrics.

## Step 2. Parameter selection and edge analysis

Choose the minimum category-balanced SSDU risk, with the original fixed tie rule. Produce 2D slices through parameter space and marginal plots showing the effect of increasing **each** of lambda, sigma and tau0. This separates an actual monotonic improvement from interactions and unstable step sizes.

Show the top several candidates and their per-category SSDU losses. Save the selected configuration and whole candidate list with SHA256 under a new name such as GS-PnP-SSDU-expanded-5 (not replacing GS-PnP-tuned-5).

If a winner again lies at one or more **new upper boundaries**, then **the search has still not established an interior optimum**. A new higher grid can be preregistered based on validation measurements alone. This is not automatic proof that ever-larger values will give better image results.

## Step 3. Freeze and prospective evaluation policy

Freeze all candidate scores, selected parameters, stopping rules, source code and model hashes **before any further test evaluation**. If there are independent future real PCM acquisitions, measure reconstruction quality and thin-line recovery there as the preferred prospective test. If we re-evaluate the already seen 450-image dataset, label it explicitly **retrospective exploratory analysis**, not untouched validation.

Compare expanded-5 against the original SSDU-5 and supervised-validation GS oracle only *after* freezing, with PSNR, SSIM, HaarPSI, MSE and detailed visual examples. Do not tune to those test metrics.

## Computational cost

27 x 5 x 4 = 540 GS solver reconstructions. A solver may run up to 100 accepted iterations, each involving network computations and possibly repeated backtracking. Total cost depends on **both** accepted iterations and the work within each iteration; not just the number of parameter triples. The earlier grid's 5-image work is saved, but only gs_018 is exactly reused by this expanded grid.

## Scientific caveat

The original 5 and 50 image studies selected identical parameter values. This supports *selection stability within that finite grid*, not a general statement that five validation images always suffice. Similarly, the ADMM SSDU-5 configuration matching the supervised validation-tuned ADMM baseline means it reaches that **specific benchmark**, not a proof of globally optimal reconstruction.
