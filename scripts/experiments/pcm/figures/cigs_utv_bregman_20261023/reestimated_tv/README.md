# CIGS 256×256: Bregman iteration with TV map re-estimated each step

These are results of the **earlier standalone CIGS image experiment**, not
synthetic dataset image0674.  They are stored separately from the fixed-weight
`iteration_*.png` images in the parent directory.

The 5 outer iterations use the same CIGS ground truth, 20% multilevel
normalised Hadamard sampling (coarse J=6), NPL default simulated PCM noise,
and checkpoint `20260920_025657_TV/best_val_loss_s0.20_cJ6.pt`
as the existing fixed-weight experiment.

Each outer step constructs `d_k = y + b_(k-1)`, recomputes the
checkpoint-configured CG input and the learned spatial TV map from `d_k`,
then solves with the unchanged U-TV/PDHG network (256 inner iterations,
zero initial image). Finally `b_k = b_(k-1) + y - A x_k`.

The first iteration and its TV map were verified to match the previous
fixed-weight experiment exactly.

Contents:

- `iteration_01.png` ... `iteration_05.png`: reconstructions
- `lambda_map_01.png` ... `lambda_map_05.png`: learned TV maps
- `lambda_maps_shared_colorbar.png`: comparison of all 5 maps
- `comparison_full_shared_colorbar.png`: ground truth + 5 reconstructions
- `comparison_bottom_right_shared_colorbar.png`: zoom
- `comparison_preset_roi_shared_colorbar.png`: original preset zoom
- `comparison_all_methods_shared_colorbar.png`: zero-filled, SPGL1, GS-PnP and reconstructions
- `comparison_fixed_vs_reestimated.png`: both Bregman variants
- `difference_reestimated_minus_fixed.png`: signed differences
- `absolute_error_all_shared_colorbar.png`: shared-scale absolute errors
- `metrics.csv`, `comparison_fixed_vs_reestimated.csv`, `protocol.json`

This is a diagnostic five-iteration run, not a discrepancy-stopped result.
Re-estimating the TV map changes the regulariser between iterations, so
the fixed-J theoretical convergence result does not apply directly.

Source script on tuni (not committed):
`scripts/experiments/pcm/bregman_reestimated_tv/run.py`.
