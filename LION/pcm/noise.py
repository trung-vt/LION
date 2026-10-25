"""PCM noise-model configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from pathlib import Path

from LION.utils.parameter import LIONParameter
from LION.data_loaders.pcm.data_primitives import PCMRandomRange
from LION.config.config_utils import make_dataclass


@dataclass(kw_only=True)
class PCMNoiseParameter(LIONParameter):
    """Parameters of the canonical PCM noise model.

    These parameters use the model convention assumed by PCM reconstruction:
    image values are already in model units and the forward operator is normalized.

    ``beam_sigma``, ``bias_sigma`` and ``bias_gradient_sigma`` are relative parameters. ``dark_sigma`` is
    an absolute additive-noise standard deviation already expressed in normalized measurement units.
    """

    beam_ar: tuple[float, float, float, float] | None
    beam_ma: tuple[float, float] | None
    beam_sigma: float | None
    bias_sigma: float | None
    bias_gradient_sigma: float | None
    dark_sigma: float | None


@dataclass(kw_only=True)
class PCMNoiseSimulation(LIONParameter):
    """Canonical PCM noise model used by synthetic acquisition simulation."""

    name: str
    noise_params: PCMNoiseParameter
    signal_to_bias_ratio: float
    noisy_measurement_filename: str
    ground_truth_filename: str
    rng_seed: int = 0
    n_repeat_acquisitions: int = 1

    @staticmethod
    def npl_default(
        *, current_to_model_scale: float, j_order: int, n_repeat_acquisitions: int = 1
    ) -> PCMNoiseSimulation:
        """Build the canonical default model from the NPL reference values."""
        dark_sigma = 3.61e-7 * current_to_model_scale / (1 << j_order)
        return PCMNoiseSimulation(
            name="PCM noise simulation - NPL default",
            ground_truth_filename="E27-full-map-default-settings-average-of-4",
            noisy_measurement_filename="E27-full-map-1-default-settings",
            noise_params=PCMNoiseParameter(
                beam_ar=(-0.9221, -0.1709, -0.0769, -0.0354),
                beam_ma=(0.0377, -0.3669),
                beam_sigma=1.2e-4,
                bias_sigma=2.9e-5,
                bias_gradient_sigma=1.6e-7,
                dark_sigma=dark_sigma,
            ),
            signal_to_bias_ratio=0.3,
            rng_seed=0,
            n_repeat_acquisitions=n_repeat_acquisitions,
        )

    @staticmethod
    def npl_extrapolate(
        *, current_to_model_scale: float, j_order: int, n_repeat_acquisitions: int = 1
    ) -> PCMNoiseSimulation:
        """Build the canonical short-aperture model from the NPL reference values."""
        dark_sigma = 2.07e-6 * current_to_model_scale / (1 << j_order)
        return PCMNoiseSimulation(
            name="PCM noise simulation - NPL short-aperture extrapolation",
            ground_truth_filename="E27-full-map-1b-default-settings",
            noisy_measurement_filename="E27-full-map-1b-Ap100us",
            noise_params=PCMNoiseParameter(
                beam_ar=(-0.9221, -0.1709, -0.0769, -0.0354),
                beam_ma=(0.0377, -0.3669),
                beam_sigma=6e-4,
                bias_sigma=2.9e-5,
                bias_gradient_sigma=1.6e-7,
                dark_sigma=dark_sigma,
            ),
            signal_to_bias_ratio=0.3,
            rng_seed=5,
            n_repeat_acquisitions=n_repeat_acquisitions,
        )

    def ground_truth_filepath(self, data_folder: Path) -> Path:
        """Get the file path for the ground-truth current map."""
        return data_folder / f"{self.ground_truth_filename}.txt.map"

    def noisy_measurement_filepath(self, data_folder: Path) -> Path:
        """Get the file path for the noisy PCM measurement."""
        return data_folder / f"{self.noisy_measurement_filename}.txt"


def load_pcm_noise_simulation(
    data: dict | None, *, j_order: int
) -> PCMNoiseSimulation | None:
    """Load a canonical PCM noise model from YAML data."""
    if data is None:
        return None

    if "preset" not in data:
        return PCMNoiseSimulation(
            name=data["name"],
            noise_params=PCMNoiseParameter(**data["noise_params"]),
            signal_to_bias_ratio=data["signal_to_bias_ratio"],
            noisy_measurement_filename=data["noisy_measurement_filename"],
            ground_truth_filename=data["ground_truth_filename"],
            rng_seed=data["rng_seed"],
            n_repeat_acquisitions=data["n_repeat_acquisitions"],
        )

    preset = data["preset"]
    current_to_model_scale = float(data["current_to_model_scale"])
    n_repeat_acquisitions = int(data.get("n_repeat_acquisitions", 1))

    if preset == "npl_default":
        return PCMNoiseSimulation.npl_default(
            current_to_model_scale=current_to_model_scale,
            j_order=j_order,
            n_repeat_acquisitions=n_repeat_acquisitions,
        )
    if preset == "npl_extrapolate":
        return PCMNoiseSimulation.npl_extrapolate(
            current_to_model_scale=current_to_model_scale,
            j_order=j_order,
            n_repeat_acquisitions=n_repeat_acquisitions,
        )
    raise ValueError(f"Unsupported PCM noise preset: {preset!r}")


@dataclass(kw_only=True)
class PCMAcquisitionVariation:
    """Sample one ratio and NPL sigma parameters without changing the simulator."""

    noise_sigma_ranges: dict[str, PCMRandomRange]
    current_to_model_scale: float
    sampling_ratio: PCMRandomRange | None = None  # Legacy config: migrate to sampling.

    def __post_init__(self):
        if isinstance(self.sampling_ratio, dict):
            self.sampling_ratio = make_dataclass(PCMRandomRange, self.sampling_ratio)
        self.noise_sigma_ranges = {
            name: make_dataclass(PCMRandomRange, spec)
            if isinstance(spec, dict)
            else spec
            for name, spec in self.noise_sigma_ranges.items()
        }
        expected = {"dark_sigma_raw", "beam_sigma", "bias_sigma", "bias_gradient_sigma"}
        if set(self.noise_sigma_ranges) != expected:
            raise ValueError(f"Expected noise sigma ranges for {sorted(expected)}")
        if (
            not np.isfinite(self.current_to_model_scale)
            or self.current_to_model_scale <= 0
        ):
            raise ValueError("Invalid PCM current-to-model scale")

    def draw_noise(
        self,
        rng: np.random.Generator,
        *,
        noise: PCMNoiseSimulation,
        j_order: int,
    ) -> PCMNoiseSimulation:
        """Draw only the sigma parameters, keeping the supplied NPL baseline."""
        sigmas = {
            name: spec.draw(rng) for name, spec in self.noise_sigma_ranges.items()
        }
        sigmas["dark_sigma"] = sigmas.pop("dark_sigma_raw") * (
            self.current_to_model_scale / (1 << j_order)
        )
        return replace(noise, noise_params=replace(noise.noise_params, **sigmas))

    def draw(
        self,
        rng: np.random.Generator,
        *,
        noise: PCMNoiseSimulation,
        j_order: int,
    ) -> tuple[float, PCMNoiseSimulation]:
        """Backward-compatible legacy acquisition variation."""
        if self.sampling_ratio is None:
            raise ValueError("Draw the ratio from sampling, then call draw_noise.")
        ratio = self.sampling_ratio.draw(rng)
        return ratio, self.draw_noise(rng, noise=noise, j_order=j_order)
