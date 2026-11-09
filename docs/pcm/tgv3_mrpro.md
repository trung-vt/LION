# TGV-3 with the existing LION/MRPro block PDHG

## Discrete model

For a frozen PCM acquisition y and normalized forward operator A, the model
minimizes

    0.5 ||A u-y||_2^2
    + alpha2 ||grad(u)-v||_1
    + alpha1 ||E1(v)-w||_1
    + alpha0 ||E2(w)||_1.

The primal variables are a scalar image u, a 2-vector v, and a full
symmetric 2x2 field w. E2 returns a full 2x2x2 tensor. Its components
are (D_i w_jk + D_j w_ik + D_k w_ij) / 3.

The 4x3 block operator has rows:
    (A, 0, 0)
    (grad, -I, 0)
    (0, E1, -I)
    (0, 0, E2).

The full tensor entries are used in the componentwise L1 penalty, including
repeated off-diagonal terms. This is an anisotropic discrete TGV-3 variant,
not a pixelwise Frobenius-norm formulation.

## Code

- LION/operators/SymmetrizedSecondOrderGradientOp.py: rank-three symmetrized
  derivative and exact discrete MRPro adjoint.
- LION/models/AdaptiveTGV3Network.py: four block rows, three primal fields,
  separable quadratic/L1 terms and zero primal proximal terms.
- LION/models/AdaptiveTGV3PCMNetwork.py: PCM model wrapper using inherited PDHG.
- LION/config/sapm_config.py: accepts the TGV3 identifier.
- LION/pcm/pcm_sapm.py: creates the TGV3 model in the PCM model loader and
  requires three output channels for a CNN map.
- pytests/pcm/test_tgv3_mrpro.py: symmetry, adjoints, dual projections,
  scalar/maps equivalence, checkpoint/autograd and solver validation.

There is NO separate PDHG iteration loop. BasePdhgNet.forward delegates to
LION/classical_algorithms/pdhg_mrpro.py.

## Usage

Construct SAPMModelConfig with backend="new",
pdhg.regularisation="TGV3", fd_dim=[-2,-1], and positive
primal_stepsize/dual_stepsize. The PCM-specific forward operator must be a
PhotocurrentMapMRProOp wrapping an already configured normalized PCM operator.
For classical TGV3, construct AdaptiveTGV3PCMNetwork(config, None) and call
forward with lambda_reg=(alpha2, alpha1, alpha0), e.g.
(0.02, 0.01, 0.005). Three spatial positive maps can be passed as a tensor
of shape (batch,3,height,width).

## Validation

- Nine unit tests passed in the LION lion_3 environment.
- Model factory explicit-map smoke test with three channels passed.
- A frozen Perovskite_0292 20% default-NPL PCM measurement matched both
  acquisition and sampling-pattern SHA256 records.
- Illustrative non-tuned scalar weights (0.02, 0.01, 0.005), 64 / 256 /
  512 / 1024 iterations: PSNR 28.76574 / 29.52464 / 29.95551 /
  30.21832 dB. All outputs finite.

## Important limitation

LION's current shared PDHG computes relaxed primal variables using
torch.lerp(old,new,relaxation). With relaxation=1, this is the new iterate
rather than textbook Chambolle-Pock extrapolation 2*new-old. The existing
solver was deliberately NOT changed, so the standard Chambolle-Pock
convergence statement should not be asserted for these exact iterates
without independent analysis.

Scalar TGV3 and three-map model construction are supported. Fully validated
end-to-end training/zero-shot TGV3 workflows and per-image optimal parameter
search are not included in this initial integration.

## References

Bredies, Kunisch and Pock, Total Generalized Variation (2010),
Eq. (3.6), p.505: https://doi.org/10.1137/090769521

Chambolle and Pock, A First-Order Primal-Dual Algorithm for Convex
Problems with Applications to Imaging (2011):
https://doi.org/10.1007/s10851-010-0251-1

MRPro operator documentation: https://docs.mrpro.rocks/
