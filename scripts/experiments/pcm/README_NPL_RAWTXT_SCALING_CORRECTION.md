# NPL unseen raw-map scaling correction, 21 October 2026

This note records an important correction to the experiments and exports.
**Use the corrected raw-text-map results**, not the earlier normalized-image
results, for the current NPL visual slides and any NPL-facing exports.

## Why the previous presets were wrong

The first unseen-sample presets used `derived/*_active_full.npy`.
Those inputs had already undergone:

```text
x_old = clip(max(-original_current, 0) / source_p995, 0, 1)
```

That preprocessing discards positive original currents and clips some negative
current magnitudes. The first presets also omitted `r_high`, so
`PCMScalingParameter.scale()` did not scale those images inside the solver.
Multiplying reconstructed images by `-source_p995` later could not reproduce
the intended raw-current inference process, especially for nonlinear priors.

The original NPL maps also have an outer **768 x 1024** canvas. The PCM
Hadamard operator is configured for **256 x 256** (CIGS) or **512 x 512**
(perovskite). The existing `image_txt_map` loader did not select the active
rectangle. A small, validated helper was needed to crop it without a flip.

## Current presets

| Parameter | CIGS | Perovskite |
| --- | --- | --- |
| Data type | `image_txt_map` | `image_txt_map` |
| Data source | original CIGS `.txt.map` | original perovskite `.txt.map` |
| `r_high` | **1.8e-9** | **4.0e-9** |
| `r_low` | 0.0 | 0.0 |
| `divide_by` | 1.0 | 1.0 |
| `data_sign` | -1.0 | -1.0 |
| `is_out_of_distribution` | True | True |
| `tests_scale_ground_truth` | False | False |
| `noise` | None | None |
| `j_order` | 8 | 9 |
| `PlotHelper.color.clim` | **(0,1.8e-9)** | **(0,4.0e-9)** |

The image loader selects the tight nonzero rectangle only when
`data_type="image_txt_map"` and the source is not already the expected
power-of-two square. An exact shape assertion rejects a wrong ROI.
E27-style full-square text maps remain unchanged.

For the normalized PCM forward operator with `r_low=0`:

```text
x_physical = -raw_active / divide_by
x_model = (x_physical - r_low) / (r_high - r_low)
simulated_measurement = A(x_model)
recon_model = method(simulated_measurement)
recon_physical = recon_model * (r_high - r_low) + r_low
```

The output `recon.npy` now contains **sign-corrected current scale**, not
dimensionless normalized values. To return to the original NPL signed
`.txt.map` convention, the export writes `-recon_physical` inside the
original raw canvas active region. There is **no p995 scaling after the
reconstruction**. There is no additional simulated noise.

## Verified test and run coverage

- Four data loading regression tests passed for cropped and already-square
  text maps, preservation of sign/orientation, and invalid ROI errors.
- Both actual NPL input tensors matched raw-map active rectangles after a
  single sign correction to within float32 rounding (less than 2e-16 A).
- A zero-fill 100% forward/inverse preflight passed for both specimens:
  physical-scale outputs with matching shapes and signs.
- **Fourteen full QUICK-method cases completed:** both specimens at
  100%, 80%, 50%, 20%, **10%, 5%, and 2%**, with strictly ordered pattern
  selection and no added noise. Each case saves all eight methods plus
  zero-fill, its `config.yaml`, result CSVs and `recon.npy` arrays.
  The 10%, 5% and 2% cases were added on 22 October 2026; the existing
  100%-20% cases were preserved.
  At 10%, `coarseJ=6` (CIGS) and `7` (perovskite). At 5% and 2%, the
  respective values are `5` and `6`, keeping the fully included
  coarse block within the sampling budget. Since the scheme is ordered,
  the selected pattern indices are exactly `0, ..., floor(p*N)-1`.
- This rerun covers the NPL visual baseline, **not** a new 384-case
  method-parameter sweep. The earlier 384-case sweep used the old normalized
  inputs and must not be mixed with these corrected raw-current results.

Results:

```text
/mnt/store_ext4/store/LION/ttv22/pcm/results/20261021_npl_rawtxt_scaled_1p8e9_4p0e9_v1/
```

Corrected NPL text exports (primary output, v2):

```text
/mnt/store_ext4/store/LION/ttv22/pcm/results/20261021_npl_rawtxt_scaled_1p8e9_4p0e9_v1/exports_for_npl_v2/
```

Use `NPL_corrected_rawtxt_scaled_100pct_CIGS_perovskite.zip`. It contains
one ZIP per specimen, retaining the original `.txt.map` filename as a
prefix, the original 768 x 1024 canvas and row order, and both original
signed-current-scale maps and dimensionless normalized-model maps.
The text files use tab-delimited scientific notation. NPL's GUI may display
y increasing upwards; **no exported array was flipped**.

Previous `exports_for_npl/` under either the original normalized run or
the initial draft corrected run are superseded and should **not** be sent
to NPL. The current `exports_for_npl_v2/` was independently reloaded and
verified against its numeric `recon.npy` sources and SHA256s.

## Scientific caveat

The original NPL map is a **measured full inversion**, not a noiseless
physical ground truth. All subsampled and 100% Hadamard acquisition vectors
in these tests are **simulated from that reference**, not independently
measured. The scaling `r_high` is an explicit user-chosen model convention;
the reconstructions should not be described as independently calibrated
physical measurements.

## Source files

- `LION/experiments/pcm_presets/presets.py`: corrected specimen presets and
  PlotHelper color limits.
- `LION/pcm/data.py`: strict active-region crop from padded `image_txt_map`,
  then sign correction and division.
- `LION/data_loaders/pcm/data_primitives.py`: `PCMScalingParameter.scale`
  and `rescale`.
- `LION/pcm/run_pcm_single_sample_experiment.py`: forward data scaling.
- `LION/pcm/run_pcm_single_sample_trial.py`: reconstruction output rescaling.
- `tests/pcm/test_raw_txt_map_scaling.py`: regression tests.
- `scripts/experiments/pcm/run_unseen_rawtxt_scaled_noiseless_ordered.sh`:
  corrected baseline reruns.
- `scripts/experiments/pcm/export_unseen_rawtxt_scaled_100pct.py`: corrected
  exports in original NPL `.txt.map` convention.
