# PCM fidelity scaling and PnP penalty conventions

This note documents the conventions in the LION PCM reconstruction code.
It is a warning about **parameter interpretation**, not evidence that the
historical DRUNet-ADMM reconstructions used a wrong scale.

## Raw versus normalized Hadamard OLS

Let `n = (2**j_order)**2`, `A_raw = S P H`, where `H H.T = n I`,
and `A_norm = A_raw / sqrt(n)`. For the corresponding measurements,

```
y_norm = y_raw / sqrt(n)
f_raw(x) = 0.5 * ||A_raw x - y_raw||**2
f_norm(x) = 0.5 * ||A_norm x - y_norm||**2 = f_raw(x) / n
```

To preserve the data proximal mapping

```
prox_f_rho(q) = argmin_x [f(x) + rho/2 * ||x-q||**2]
```

when converting from raw to normalized OLS, use
`rho_norm = rho_raw / n` (and conversely `rho_raw = n * rho_norm`).
This also preserves the entire fixed-denoiser ADMM iteration if its starting
state and all other parameters stay unchanged. Keeping the same numeric
`rho` after the conversion changes the reconstruction.

`pcm_prox_transform_rho(op, rho)` performs a **different operation**:
it converts the penalty within the closed-form solve for the selected
operator's own fidelity, using `rho_transform = rho / kappa`,
`kappa = 1` for normalized rows and `kappa = n` for raw rows.
It does **not** automatically convert an experiment's configured `rho`
between the two physical operator conventions.

The current PCM simulator and single-sample reconstruction pipeline use
normalized Hadamard rows. An operator normalization change also requires
rescaling simulated or real measurements, not just the operator.

## Additional scaling by measurement count

Let `m` be the number of selected measurements. These are independent
normalizations: Hadamard normalization uses `n`; mean-squared fidelity
uses `m`.

For the objective `f(x) = c/2 * ||A x-y||**2`, the closed-form PCM
prox with `rho_pcm` is equivalent to a CG step with penalty
`eta = c * rho_pcm`. When `c=1/m`, `eta = rho_pcm/m`.
Changing `f` to `d*f` requires changing the penalty to `d*rho`
to preserve the same proximal map.

- `PnPADMM` + `PCMProxParameter`: `c=1` for the chosen `A,y`.
- `PnPADMM` + `CGProxParameter`:
  `c=data_weight/m` when `normalize=True`, otherwise
  `c=data_weight`. The CG `eta` is defined relative to this fidelity.
- Native `GSPnP`: `NormParameter(ord=2, factor=a, normalize=flag)`
  gives `c=2*a/m` when normalized and `c=2*a` otherwise.
  Its CG and PCM proximal branches use the same `c` as the objective.
  When an isolated CG proximal helper is called without an objective,
  its own `data_weight` and `normalize` options define the scale.
- DeepInv GS-PnP with `L2()`: uses `c=1`. The adapter requires
  compatible `NormParameter` settings rather than silently ignoring
  conflicting scaling choices.

For GS-PnP of the form

```
x_next = prox_(tau*f)(x - tau*lambda*grad_g(x))
```

if `f_new = d*f`, preserving the *whole* iteration also requires
`lambda_new = d*lambda` and `tau_new = tau/d`. Merely adjusting
the proximal penalty without adjusting the gradient-step coefficient may
give a different algorithm.

## Weighted fidelity caution

The simple `rho_norm = rho_raw/n` rule assumes OLS (or unchanged
numerical weights) under `A,y -> A,y / sqrt(n)`. If the measurement
covariance is transformed consistently,
`Sigma_norm = Sigma_raw/n`, then inverse-variance WLS satisfies

```
0.5 * r_norm.T @ inv(Sigma_norm) @ r_norm
 = 0.5 * r_raw.T @ inv(Sigma_raw) @ r_raw.
```

That fidelity does *not* acquire the OLS `1/n` factor, so no `rho`
rescaling is required solely for this operator-unit conversion. The
covariance, residual, and data units must all agree. Similarly, changing
image-intensity units can require changes to denoiser noise levels and
regularization parameters, not only to the fidelity penalty.

## Relevant source locations and tests

- `LION/pcm_tools/pcm_prox.py`: closed-form fidelity/penalty convention.
- `LION/reconstructors/PnPADMM.py`: exact PCM and CG data steps.
- `LION/reconstructors/GSPnP.py`: native objective/prox and DeepInv L2.
- `LION/reconstructors/PnP.py`: generic unnormalized OLS CG data step.
- `LION/classical_algorithms/conjugate_gradient.py`: CG configuration.
- `LION/pcm/data.py` and `LION/pcm/run_pcm_single_sample_experiment.py`:
  real measurement conversion and normalized acquisition setting.
- `LION/experiments/pcm_presets/presets.py`:
  configured penalty and fidelity choices.
- `pytests/pcm_tools/test_pcm_fidelity_scaling.py`:
  raw/normalized equivalence, faithful objective/prox scaling,
  and alternative solver regression tests.

The existing `PhotocurrentMapOp.normalized` parameter docstring says
"dividing by the number of elements", but the actual implementation
divides the raw WHT by `sqrt(n) = 2**j_order`, not by `n`. The
formulas and tests in this note follow the implementation.

The acquisition archive's strict `executed_source_sha256` lists
`LION/operators/PhotocurrentMapOp.py` and `LION/pcm/simulation.py`.
These source files were deliberately not changed to add warnings. Even
comment-only edits would invalidate their frozen source-hash checks.
