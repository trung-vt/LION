"""Measurement-only validation for SSDU checkpoint selection."""

from __future__ import annotations

import numpy as np
import torch

from LION.pcm.randomized.models import regularized_reconstruction
from LION.pcm.randomized.noise import (
    NoiseRange,
    randomized_acquisition,
    sample_parameters,
    split_ssdu,
)


@torch.no_grad()
def ssdu_validation_score(model, config, corpus) -> float:
    """Score predicted held-out noisy coefficients on fixed validation images.

    Clean images enter only the measurement simulator. They are not reconstruction
    targets and do not determine the checkpoint ranking. Residual correlations
    across the split still limit the statistical interpretation of this loss.
    """
    model.eval()
    losses = []
    for i, (image, _) in enumerate(
        corpus.fixed_cases("validation", config.validation_per_domain)
    ):
        clean = image.to(config.device)
        for dark, beam in ((0.05, 1.2e-4), (0.2, 6e-4), (0.8, 1e-3)):
            for ratio in (0.01, 0.2, 1.0):
                rng = np.random.default_rng(1_200_000 + i * 100)
                parameters = sample_parameters(rng, NoiseRange())
                parameters.dark_sigma, parameters.beam_sigma = dark, beam
                acquisition, _ = randomized_acquisition(
                    clean,
                    rng,
                    NoiseRange(),
                    ratio=ratio,
                    parameters=parameters,
                )
                input_op, input_y, loss_op, loss_y, _ = split_ssdu(acquisition, rng)
                reconstruction = regularized_reconstruction(model, input_op, input_y)
                loss = (loss_op(reconstruction) - loss_y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite held-out validation loss")
                losses.append(float(loss))
    return float(np.mean(losses))
