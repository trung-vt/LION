# PCM PnP-ADMM and GS-PnP: SSDU hyperparameter tuning (5 vs 50 validation images)

Preparation-only branch. Read PLAN.md for exact steps and caveats.

Frozen input: original 50 stored FLOAT64 validation acquisitions, 10 per category, 20% multilevel Hadamard J=8, coarse_J=6, NPL default.

Run preparation from this worktree with the lion_3 Python environment:

    /home/t/micromamba/envs/lion_3/bin/python -B scripts/experiments/pcm/ssdu_pnp_tuning_20261108/prepare.py

Run tests with:

    /home/t/micromamba/envs/lion_3/bin/python -m pytest -q scripts/experiments/pcm/ssdu_pnp_tuning_20261108/test_prepare.py

Preparation generates immutable pilot case names, the same frozen grids for both stages, checksum-backed folds across all 50 validation images, and a protocol manifest in:

    /store/LION/ttv22/pcm/results/pcm/ssdu_pnp_tuning_20261108_v1/prepared

The prepared source contains NO hyperparameter sweep runner yet. Step 1 of PLAN.md requires measurement-only solver adapters and a data-fidelity scaling audit before tuning can start. No test-set access is made during preparation.

Method names:
- PnP-ADMM-tuned-5
- PnP-ADMM-tuned-50
- GS-PnP-tuned-5
- GS-PnP-tuned-50

Comparison uses seven existing reference methods plus these four (11 methods), for the fixed 450-image test set.

The operator-only dry run is:

    /home/t/micromamba/envs/lion_3/bin/python -B -m scripts.experiments.pcm.ssdu_pnp_tuning_20261108.preflight_operator

The snapshot of the frozen pilot subset and candidate grids is under prepared_snapshot. The 11-method evaluation roster and ground-truth provenance guards are in comparison_methods.json.
