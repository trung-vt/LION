# Original CIGS example: fixed-TV-map Bregman, 1024 PDHG iterations

This is the **first experiment in the conversation**, re-run with
**1024 rather than 256 PDHG inner iterations**, using the *same ground truth*
as [the original GitHub PNG](https://github.com/trung-vt/LION/blob/1fe389f714edf48fb6de6a1f61b493149186fafa/scripts/experiments/pcm/figures/cigs_utv_bregman_20261023/ground_truth.png).

Ground-truth image source:
`/store/LION/ttv22/pcm/data/photocurrent_data/example_CIGS_256x256.npy`.

The PNG copied into these results has Git blob SHA
`f8e0953a4a1be2d151190629b814d1c19f8ef993`, which exactly matches
the user-linked GitHub ground truth.

## Protocol

- Original experiment: `/store/LION/ttv22/pcm/results/20261023_cigs_utv_bregman_five_outer_exp10/`.
- Original acquisition: 20% normalised subsampled Hadamard PCM forward
  operator, multilevel sampling, coarse J = 6, NPL default simulated noise,
  seed **1045075180**.
- Unchanged supervised checkpoint:
  `/store/LION/ttv22/pcm/results/pcm/pcm_supervised/20260920_025657_TV/checkpoints/best_val_loss_s0.20_cJ6.pt`;
  unchanged `resolved_config.yaml`.
- Learned spatial TV map: estimated **once** from the checkpoint's
  `CG(y)` input, and **kept fixed** across five outer iterations.
- Inner PDHG: original LION/MRPro solver, checkpoint's zero initial image and
  step sizes; the **only solver change** is `max_iterations=1024` rather
  than 256.
- Five outer Bregman iterations:
  `d_k=y+b_(k-1)`;
  `x_k=PDHG(d_k, fixed_lambda, max_iterations=1024)`;
  `b_k=b_(k-1)+y-Ax_k`.

This experiment uses a fixed map, **not** the later variant that re-estimates
the map at every Bregman step.

The script checks that its ground-truth array, sampled indices and noisy
measurements match the original experiment **exactly**. It also checks the
original fixed lambda map exactly, and that a one-off 256-step reconstruction
reproduces the earlier first iteration with zero difference.

## Reproduce on tuni

From the LION repository (`experiments_10`):

```bash
export LION_DATA_PATH=/store/LION/ttv22/pcm/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

python scripts/experiments/pcm/bregman_example_cigs_1024/run.py
python scripts/experiments/pcm/bregman_example_cigs_1024/render.py
```

The run intentionally refuses to overwrite an existing results directory.
Use a new destination if rerunning.

Results directory:
`/store/LION/ttv22/pcm/results/20261024_cigs_example_bregman_fixed_map_pdhg1024/`.

Outputs include all five 1024-step reconstructions (PNG and NPY), the fixed
TV weight map, original noisy measurements and mask, metrics, timing,
matched original zero-filled/SPGL1/GS-PnP baselines, common-scale image and
error comparisons with the original 256-step results, signed difference
images, and protocol metadata.

This is a **five-iteration diagnostic** run; it does not use automatic
discrepancy stopping. The finite 1024-step PDHG solve is not claimed to
produce an exact minimiser, nor is monotonic PSNR improvement guaranteed.
