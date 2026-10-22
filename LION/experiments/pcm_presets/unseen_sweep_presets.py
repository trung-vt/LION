"""Independent inference-parameter variants for the unseen PCM sampling study.

The original QUICK_METHODS and the two base sample presets are never mutated.
Changing parameters against unseen examples is exploratory, not a valid way
to choose final hyperparameters for held-out paper testing.
"""

from __future__ import annotations

from copy import deepcopy

from LION.experiments.pcm_experiments import PCMExperimentParameter


ADMM = ("pnp_admm_drunet", "pnp_admm_dwtdnet_tiny")
GS = ("pnp_gs_deepinv_drunet", "pnp_gs_deepinv_dwtdnet_tiny")
PNP = ADMM + GS

# Human-readable list for the two standalone GPU scripts and documentation.
PROFILE_NAMES = (
    "admm_rho_003",
    "admm_rho_100",
    "pnp_sigma_020",
    "pnp_sigma_100",
    "gs_lambda_030",
    "gs_lambda_300",
    "tv_lambda_005",
    "tv_lambda_050",
    "spgl1_db2",
    "iteration_cap_x2",
)


def make_unseen_sweep_presets(
    base: PCMExperimentParameter, quick_methods: dict
) -> list[PCMExperimentParameter]:
    """Create new named presets with one controlled parameter axis per profile."""

    def select(suffix: str, method_ids: tuple[str, ...]) -> PCMExperimentParameter:
        result = deepcopy(base)
        result.id = f"{base.id}_{suffix}"
        result.methods = {name: deepcopy(quick_methods[name]) for name in method_ids}
        return result

    variants: list[PCMExperimentParameter] = []

    # ADMM rho weights proximity to the previous iterate in the PCM data step.
    # This is not an image noise standard deviation or an L2 data precision.
    for suffix, rho in (("admm_rho_003", 0.003), ("admm_rho_100", 0.1)):
        cfg = select(suffix, ADMM)
        for name in ADMM:
            cfg.methods[name].prox.rho = rho
        variants.append(cfg)

    # Denoiser sigma is the denoiser's assumed noise level. Hold rho, GS
    # lambda, and all trained network weights at the original QUICK values.
    for suffix, sigma in (("pnp_sigma_020", 0.02), ("pnp_sigma_100", 0.1)):
        cfg = select(suffix, PNP)
        for name in PNP:
            cfg.methods[name].sigma = sigma
        variants.append(cfg)

    # GS-PnP lambda weights the explicit GS potential. Leave its dynamic
    # backtracking and initial stepsize policy unchanged.
    for suffix, lam in (("gs_lambda_030", 0.03), ("gs_lambda_300", 0.3)):
        cfg = select(suffix, GS)
        for name in GS:
            cfg.methods[name].lambd = lam
        variants.append(cfg)

    # Scalar TV lambda multiplies the anisotropic image-gradient L1 penalty.
    for suffix, lam in (("tv_lambda_005", 0.005), ("tv_lambda_050", 0.05)):
        cfg = select(suffix, ("scalar_tv",))
        cfg.methods["scalar_tv"].lambda_reg = lam
        variants.append(cfg)

    # SPGL1 uses a wavelet-domain L1 prior. A different wavelet is a different
    # basis, unlike changing its numerical data-scaling factor.
    cfg = select("spgl1_db2", ("spgl1",))
    cfg.methods["spgl1"].wavelet_name = "db2"
    variants.append(cfg)

    # Separate numerical iteration-budget experiment. Some GS methods may
    # still stop early under their original tolerance and backtracking rules.
    cfg = select("iteration_cap_x2", PNP + ("spgl1", "scalar_tv"))
    for name in PNP:
        cfg.methods[name].iters = 100
    cfg.methods["scalar_tv"].iters = 2048
    cfg.methods["scalar_tv"].metric_every = 2048
    cfg.methods["spgl1"].iters = 200
    variants.append(cfg)

    assert len(variants) == len(PROFILE_NAMES)
    assert [preset.id.rsplit("_", 3)[-1] for preset in variants]  # Keep IDs nonempty.
    assert len({preset.id for preset in variants}) == len(variants)
    return variants
