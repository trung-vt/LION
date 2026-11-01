# Step 2: float64 validation acquisition archive

Completed on tuni, branch `experiments_10`, 1 November 2026 (UTC).

## Scope

This is acquisition storage, not a reconstruction benchmark. There are 50 validation acquisitions (10 per category) and no held-out test acquisitions. All image loading, split-Hadamard arithmetic, noise additions, differencing and master measurements use float64. The original images, previous protocols/pilot, training configuration, solver defaults and existing 4090 job were not changed.

## Minimal changes to shared code

- `LION/pcm_tools/noise_generate.py`: five added lines. An optional `swap_flags_out` list receives the flags from the existing random draw. No new random draw and no change to the three returned tensors.
- `LION/pcm/simulation.py`: thirteen added lines. An optional `record_swaps=False` argument and optional `PCMAcquisition.swap_flags` field. Recording is off for existing callers; the float64 experiment enables it.
- The experiment-specific `float64_acquisition.py` extends the earlier wrapper with strict archive validation, packed flags, atomic no-overwrite writes, loading and independent solver copies. Existing `test_oracle_wls_float64_acquisition.py` is unchanged.
- New `acquisition_archive.py` builds and verifies validation only. New `test_oracle_wls_storage.py` adds 24 tests. No dependencies were added.

The review ZIP provides exact before/after sources and a patch against the working tree immediately before this task, not a patch combining earlier uncommitted BPDN work.

## Stored NPZ schema: pcm-float64-swaps-v1

- `measurement`: float64, `(1, 1, 13107)`, normalised differential measurements.
- `sampled_indices`: uint16, `(13107,)`, ascending external dyadic indices. The loader converts these to int64 for indexing.
- `i0`: float64, `(1, 1)`.
- `swap_flags_packed`: uint8, `(1, 1639)` for one repeat. Little-endian bit order; the five unused bits at the end are zero.
- `metadata_json`: Unicode scalar, not a pickle. Includes source image/hash, full numeric noise parameters, mask/noise seeds, protocol ID, measurement order/normalisation and schema.

For zero-based pair k, flag 0 assigns noise times `(2k, 2k+1)` to `(plus, minus)`; flag 1 assigns `(2k+1, 2k)`. These are noise assignments, not a reordering of the stored Hadamard rows. Runtime flags have shape `(*image_batch_shape, repeats, measurements)` and are CPU booleans. All repeats are recorded separately before averaging.

Primary oracle weights remain planned as `Var(y-Ax | x,A)`, marginal over random swaps. Saved flags are audit/replay information, not additional information supplied to the primary WLS weights. No weights or full covariance whitening are implemented here.

## On-disk location

`/store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v2/`

This holds protocol.json, the unchanged parent precision protocol and input manifest, 50 validation NPZs, manifest.json, verification.json and SHA256SUMS. The test directory is intentionally empty. Read-only permissions prevent accidental edits; they are not an OS immutable flag or cryptographic signature.

## Commands from the LION repository

Use the existing lion_3 environment. Set `LION_DATA_PATH=/store/LION/ttv22/pcm/data`, `LION_TTV22_PATH=/store/LION/ttv22`, and `PYTHONPATH=.`. For GPU runs expose only the approved 4070 UUID: `CUDA_VISIBLE_DEVICES=GPU-05267a12-f71f-101d-59e5-05e5c430d8ef`. Check that it has no active compute process first; do not start concurrent CUDA processes on it. Use CPU with `CUDA_VISIBLE_DEVICES=-1` and `--device cpu`.

Verify stored files without simulating anything:

```bash
python -m scripts.experiments.pcm.oracle_wls.acquisition_archive verify \
  --root /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v2
```

Replay on the 4070 and cross-check CPU:

```bash
python -m scripts.experiments.pcm.oracle_wls.acquisition_archive verify \
  --root /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v2 \
  --device cuda:0 --replay --cpu-crosscheck
```

To regenerate, use `build --protocol <parent_float64_protocol.json> --manifest <step1_image_and_seed_manifest.csv> --root <new-empty-output-path> --device cuda:0`. The build command refuses any existing output directory. It does not support test-set generation or silent resume of partial archives.

## Checked evidence

- Baseline before edits: 85 tests passed.
- Final CPU suite: 109 tests passed; new tests with 4070 enabled: 24 passed.
- Exact pre-change versus post-change source comparison: 48 configurations, 96 byte-identical measurement comparisons; RNG states also identical.
- All 50 stored files round-trip without numerical loss and replay byte-for-byte on the 4070.
- All 50 CPU cross-checks passed a predeclared absolute tolerance of 2e-12. Largest measurement/I0 difference: 2.2737367544323206e-13. CPU/GPU masks and swap flags were identical.
- Corruption, missing cases, unexpected test files, changed protocol and existing output directories were rejected on a disposable copy. The original archive still verified afterwards.
- NPZ files total 5,770,589 bytes. The 4090 training process (PID 301833) remained running. No reconstruction result or WLS quality claim follows from these checks.

Checksums identify the saved compressed files. Replay compares raw array bytes; a newly compressed ZIP container need not have the same file hash. Keep the decimal 64-bit seeds as exact integers; do not round them through a spreadsheet or floating-point JSON parser.

## Exact source provenance after Git commits

The 50 frozen validation acquisitions contain SHA-256 hashes of the exact Python
source that generated them. The repository Black hook reformatted three Python
files during Git commits without changing their syntax trees. Those three files
were restored byte-for-byte from the audited `review_export.tar.gz` snapshot
and committed with the Black hook skipped once. This preserves strict source
hash checks during replay. Do not reformat these three frozen-source files
without creating a new, explicitly versioned acquisition protocol.
