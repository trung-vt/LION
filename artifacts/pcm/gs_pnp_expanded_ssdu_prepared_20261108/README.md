# Expanded GS-PnP SSDU five-image tuning: preparation

Status: PREPARED, NO NEW PARAMETER SWEEP STARTED.

The original gs_018 was independently verified and reused as gs_exp_001.
Its SSDU MSE is 0.004365433883375238. The remaining 26 candidate
parameter combinations are still untested.

Original validation set: five PCM images (one per category);
four disjoint held-out Hadamard measurement folds per image.

Frozen 27-candidate grid:
- lambda: 0.10, 0.20, 0.40
- denoiser sigma: 0.075, 0.10, 0.15
- initial tau0: 33.3333, 50, 75

The grid comprises 540 total fold records, of which 20 are a verified
old control and 520 still need GPU reconstruction.

Report/ contains the editable GS_PnP_Expanded_SSDU_Tuning_Report.tex,
compiled PDF, candidate_rows.tex, current_results.tex, and status.json.
The separate code uses the earlier validation-only acquisitions and
pretrained GS-DRUNet, with the original paper-final-z output and
deterministic CUDA settings.

To start the expanded GPU sweep only after explicit approval:

  cd /home/t/Documents/GIT/LION_pcm_ssdu_tuning_20261108
  bash scripts/experiments/pcm/ssdu_pnp_tuning_20261108/run_expanded_gs.sh execute

Without running the sweep, a status check is:

  bash scripts/experiments/pcm/ssdu_pnp_tuning_20261108/run_expanded_gs.sh status

The execute command records case-level scores, resumes valid records,
excludes invalid configurations, freezes a validation-only selection,
creates numerical figures, regenerates table fragments and recompiles
the report. No test image quality is used to select parameters.
Previous tuned-5 and tuned-50 results and the matched test benchmark
remain intact in the parent SSDU experiment folder.
