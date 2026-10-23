# Five-step Bregman experiment: synthetic CIGS image0674

This experiment uses the original synthetic PCM dataset sample
`Synthetic Data/CIGS/image0674.png`. It contains a one-pixel-wide,
33-pixel-long horizontal dark line in row 115, columns 106–138.

The experiment compares **fixed** and **re-estimated** spatial TV parameter
maps in five Bregman outer iterations, using otherwise identical
measurements and reconstruction settings.

## Experiment configuration

- Branch: `experiments_10`.
- Input: 256 × 256 grayscale PNG, loaded with the existing PCM dataset
  scaling (`Image.convert("L") / 255`).
- Acquisition: the existing synthetic PNG box-plot preset
  (`SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS`), one CIGS sample, 20%
  multilevel subsampled *normalised* Hadamard operator, coarse J = 6,
  NPL default simulated measurement noise, and trial 1.
- The existing seed construction is used with
  `sample_name="artifacts/CIGS/sample_0674"`,
  `category="CIGS"`, `sample_id=674`. No custom acquisition or
  noise generator is introduced.
- Checkpoint:
  `/store/LION/ttv22/pcm/results/pcm/pcm_supervised/20260920_025657_TV/checkpoints/best_val_loss_s0.20_cJ6.pt`,
  with its `resolved_config.yaml`. This model was trained with
  noiseless synthetic measurements; the current experiment uses noisy
  measurements.
- TV model input: CG reconstruction from the current measurement target.
  Inner solution: original 256-iteration `AdaptiveTVPCMNetwork`,
  `backend="new"`, zero initial image, original PDHG parameters.
- Additional baselines: original zero-filled, SPGL1 and
  `pnp_gs_deepinv_drunet` PCM reconstruction paths and parameters.

The Bregman iteration uses

```text
d_k = y + b_(k-1)
x_k = PDHG(d_k, lambda_k)
b_k = b_(k-1) + y - A(x_k)
```

In the **fixed** case, the learned TV map is obtained once from `y`.
In the **re-estimated** case, `make_pcm_forward_data` recalculates the
CG model input and the learned TV map from `d_k` at every outer iteration.
The forward operator and observed `y` remain unchanged.

## Run on tuni

From the repository root:

```bash
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/experiments/pcm/bregman_cigs_0674/run.py
python scripts/experiments/pcm/bregman_cigs_0674/render.py
```

Output directory:

```text
/store/LION/ttv22/pcm/results/20261024_cigs_synthetic_0674_bregman_5steps
```

The run is **five diagnostic iterations**, even if the discrepancy
principle is satisfied earlier. The script logs the oracle noise norm
(using known synthetic ground truth), residuals against original `y`,
pixel-line contrast, PSNR, SSIM, HaarPSI, MSE, TV map statistics and
timings. Each variant saves the five reconstruction PNGs and arrays;
the re-estimated variant also saves five spatial TV maps.

Plotting provides comparisons on shared intensity/colorbar scales,
absolute-error images, a zoom near the one-pixel line, TV maps, and a
pixel-level line profile.

## Interpretation

Recomputing the learned weights changes the TV regulariser across outer
iterations. The fixed-convex-regulariser Bregman convergence theorem in
Appendix B does **not** apply directly. Moreover, the trained U-TV
network's 256 PDHG steps are not established to be exact inner
minimisers. Re-estimated maps can also encounter input distributions
that differ from training. This is an empirical comparison only.
