# Sequential PCM U-TV hyperparameter search (full 256x256)

All trials start from scratch with common seeds on 10 training + 2
validation examples per category (50 / 10 images), 512 PDHG steps,
random sampling 5-50% and new 0-inclusive shifted-log sigma ranges.
One shared scale factor only, across all sampling rates.
All short trials disable checkpoint writes; scores are mean validation
MSE over the two end-of-epoch evaluations. Comparisons remain noisy.

1. Base optimizer LR: run Adam with 0.0002, 0.0006, 0.00007;
   keep scale-factor LR=0.0002, initial raw scale=0.1.
2. Choose best base LR, compare separate scale LR 0.1x vs 5x of base,
   with a control retained from stage 1 (do not cross with all base LRs).
3. Choose best pair, test initial raw scale -1.0 vs +1.0,
   using the current value 0.1 as control.
4. Compare AdamW vs Adam **once** using the current best; consider
   weight_decay 1e-4 separately if AdamW's baseline effect is negligible.
5. Check architectures at most: 4 stages x 256 filters and
   5 stages x 128 filters; compare against 4 x 128 baseline.
   Do not attempt 6 x 512 (8.85B params, Adam states alone ~132GiB).
   Run memory smoke test on available GPU before changing architecture.
6. Recheck top 1-2 settings with 4-5 epochs and possibly 20 training
   images/category, 4 validation images/category before full training.

The ranking of 2-epoch runs is a screening heuristic only.
Use common RNG seeds/splits for paired fair comparisons. The ongoing
1000-epoch RTX 4090 training run is left untouched.


## Verified implementation and reproducibility

The original 2026-10-25 eight-trial screen and first five-epoch trials
should **NOT** be used for the new noise-range selection. An entry-point
imported the original installed LION source instead of the isolated worktree;
the saved *resolved_config.yaml* showed a fixed 20% acquisition and did
not include the zero-inclusive noise variation. Their scores were therefore
computed under the wrong acquisition distribution.

The replacement experiments under
`/store/LION/ttv22/pcm/results/20261025_utv_zero_inclusive_hpo/verified_hpo_20261025`
use the current LION source through **python -m** and explicit PYTHONPATH
and verify the resolved configuration before accepting trial scores.

The intended long-training YAML uses one authoritative configuration:

- `sampling.sampling_ratios: []` and `sampling.sampling_ratio_range` for
  uniformly sampled acquisition ratios in [0.05, 0.5].
  A nonempty discrete list takes precedence over a range.
- `validation_sampling` explicitly supplies the independent validation
  conditions. Acquisitions are seeded identically across epochs, reducing
  noise in comparisons and giving correct checkpoint names, e.g.
  `best_val_loss_s0.05-0.50_cJ6.pt`.
- `test_sampling.sampling_ratios` defines the independent fixed test
  percentages [0.05, 0.1, 0.2, 0.3, 0.4, 0.5].
- `acquisition_variation.noise_sigma_ranges` covers the four sigma
  distributions with a zero-inclusive shifted-log law. It no longer
  duplicates the sampling ratio.
- `sampling.share_scale_factor_across_ratios: true` requests one single
  learned scalar for all ratios.
- The legacy `acquisition_variation.sampling_ratio` remains loadable for
  old experiments, but is deprecated and has an explicit compatibility path.

For launches inside a worktree, invoke LION as a module from that worktree,
not as a file-path script that could import an editable installation elsewhere:

```bash
cd /path/to/checked-out/LION
PYTHONPATH="$PWD" python -m scripts.experiments.pcm.train.pcm_supervised_sapm_train --config config/pcm/pcm_utv_zero_inclusive.yaml
```

No long training should be started from this config until the verified
hyperparameter finalist results have been compared. The running old RTX
4090 training job is independent and must not be interrupted.
