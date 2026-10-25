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
