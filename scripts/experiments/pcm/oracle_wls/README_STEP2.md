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


## Reviewer follow-up (2 November 2026)

The review of Step 2 identified five main guard gaps. These are now addressed
in experiment-specific acquisition code. Shared PCM numerical code is unchanged.
The new focused regression module is test_oracle_wls_step2_review_followup.py.

1. Manifest integrity: build writes SHA256SUMS covering manifest, protocol, source
   manifest and all NPZ files. Plain verify checks each digest. Build prints
   EXTERNAL_MANIFEST_SHA256. Keep that hash outside the archive and supply the
   --expected-manifest-sha256 option when independent anchoring is required.
   Internal checks alone cannot detect a coordinated edit to the manifest and
   the sidecar.
2. Hadamard identity: CODE_FILES includes LION/pcm_tools/ordering.py, while
   execution metadata records the spyrit version. Each new NPZ records the SHA
   of actual selected Hadamard row indices; loading verifies the same rows and
   a fixed SHA of the canonical dyadic j=8 ordering. Historical NPZ files have
   no per-image row digest, but the canonical row check still applies.
3. Noise consistency: verify compares the entire locked noise-parameter
   dictionary to the stored metadata, including expected keys.
4. Locked parameters: even single-case creation rejects missing NPL numeric
   noise keys before simulating.
5. Held-out isolation: the one-case CLI accepts validation only; the archive
   verifier rejects all files or directories inside the test folder.

Additional protections: same-bytes hash/load, fsync of directory publication,
decimal-string seed copies, and a loader by category and sample ID that checks
the checksum sidecar plus the individual file digest.

### Version history and frozen outputs

- Original immutable archive: acquisitions_v2/; original manifest SHA:
  96ae46bd96572cdd2d62523b43365cb32ca3b290d9a174203230c78a6785671f.
- Intermediate reviewer preflight: acquisitions_v4_reviewed/ (separate).
- Final reviewed archive: acquisitions_v5_reviewed_final/; manifest SHA:
  b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac.

The saved measurement y, indices, bias i0 and swap bits of the final archive
are checked byte-for-byte against all 50 original validation acquisitions.
Metadata and compressed NPZ file hashes are different because guards were added.
Neither the original source image dataset nor the frozen acquisition archive
was modified.

The original archive's strict replay checks source SHA-256. Running that replay
with newer, updated source code can fail the intended provenance check. Use
the historical source snapshot to replay old data; plain checksums continue to
work with the historical archive.

### Verification

From the LION root with existing LION_DATA_PATH and LION_TTV22_PATH:

~~~bash
python -m scripts.experiments.pcm.oracle_wls.acquisition_archive verify   --root /store/LION/ttv22/pcm/results/pcm/pcm_oracle_wls_vs_ols_20261102/acquisitions_v5_reviewed_final   --expected-manifest-sha256 b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac

python -m pytest -q -o addopts='' -p no:cacheprovider   pytests/pcm/test_oracle_wls_step2_review_followup.py   pytests/pcm/test_oracle_wls_storage.py   pytests/pcm/test_oracle_wls_float64_acquisition.py
~~~

126 tests passed across the 11-file relevant CPU regression suite.
50/50 GPU replay and CPU comparison passed with max difference
2.2737367544323206e-13. No full-repository test was claimed.

The optional chmod-0555 directory rule was not added: directory modes do not
give cryptographic authenticity and would complicate verification writebacks.
The checksums and separately pinned manifest are a stronger audit control.
The 4070 and 50-case split remain intentionally locked for this study.
No oracle weights or held-out test measurements were generated here.
