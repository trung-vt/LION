from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from LION.classical_algorithms.spgl1_torch import SPGL1Parameter
from LION.config.config_utils import load_yaml, make_dataclass
from LION.config.ExperimentConfig import (
    ExperimentConfig,
    ExperimentConfigLoader,
    PretrainedConfig,
    ReproduceConfig,
    RuntimeDirs,
    TrainConfig,
)
from LION.config.LogConfig import make_log_config
from LION.config.sapm_config import (
    SAPMModelConfig,
    SAPMOptimConfig,
    make_sapm_model_config,
)
from LION.data_loaders.pcm.data_primitives import (
    PCMDataCategory,
    PCMDataParameter,
    PCMSamplingParameter,
    SamplesParameter,
)
from LION.pcm.noise import (
    PCMAcquisitionVariation,
    PCMNoiseSimulation,
    load_pcm_noise_simulation,
)
from LION.pcm_tools.pcm_sampling import (
    PCMAcquisitionSamplingCase,
    make_acquisition_sampling_cases,
)
from LION.train.self_supervised_train import SelfSupervisedCriterion
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class SupervisedPCMDataConfig(PCMDataParameter):
    """PCM data configuration for supervised training."""

    train_perm_range: list[int]
    val_perm_range: list[int]
    test_perm_range: list[int]


@dataclass(kw_only=True)
class ZeroShotPCMDataConfig(PCMDataParameter):
    """PCM data configuration for single-sample zero-shot training."""

    p_target_train: float
    p_target_val: float | None
    base_train_partition_seed: int
    base_val_partition_seed: int

    this_sample_id: int = -1
    this_category: PCMDataCategory | None = None


@dataclass(kw_only=True)
class PCMSSDUConfig(LIONParameter):
    """SSDU measurement-splitting configuration."""

    p_target_train: float
    p_target_val: float


@dataclass(kw_only=True)
class TrainedPCMMethodParameter(LIONParameter):
    """Configuration shared by trained PCM reconstruction methods."""

    run_dir: Path
    checkpoint: str
    scale_factor_case: tuple[float, int] | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)


@dataclass(kw_only=True)
class SupervisedPCMMethodParameter(TrainedPCMMethodParameter):
    """Configuration for a trained supervised PCM reconstruction method."""


@dataclass(kw_only=True)
class SelfSupervisedPCMMethodParameter(TrainedPCMMethodParameter):
    """Configuration for a trained dataset self-supervised PCM reconstruction method."""


@dataclass(kw_only=True)
class ZeroShotPCMMethodParameter(LIONParameter):
    """Configuration for a zero-shot PCM reconstruction method."""

    config_path: Path
    criterion: SelfSupervisedCriterion = "best_val_ssdu_loss"

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)


@dataclass(kw_only=True)
class SAPMSupervisedPCMConfig(ExperimentConfig):
    """Full configuration for supervised PCM SAPM training."""

    acquisition_variation: PCMAcquisitionVariation | None = None

    data: SupervisedPCMDataConfig  # type: ignore[assignment]
    model: SAPMModelConfig  # type: ignore[assignment]
    optim: SAPMOptimConfig  # type: ignore[assignment]
    sampling: PCMSamplingParameter = field(default_factory=PCMSamplingParameter)
    validation_sampling: PCMSamplingParameter | None = None
    test_sampling: PCMSamplingParameter | None = None
    spgl1: SPGL1Parameter | None = None
    noise: PCMNoiseSimulation | None = None
    uses_normalized_op: bool = True


@dataclass(kw_only=True)
class SAPMSelfSupervisedPCMConfig(SAPMSupervisedPCMConfig):
    """Full configuration for dataset self-supervised PCM SAPM training."""

    ssdu: PCMSSDUConfig


@dataclass(kw_only=True)
class SAPMZeroShotPCMConfig(ExperimentConfig):
    """Full configuration for zero-shot PCM SAPM training."""

    data: ZeroShotPCMDataConfig  # type: ignore[assignment]
    model: SAPMModelConfig  # type: ignore[assignment]
    optim: SAPMOptimConfig  # type: ignore[assignment]
    sampling: PCMSamplingParameter = field(default_factory=PCMSamplingParameter)
    spgl1: SPGL1Parameter | None = None
    noise: PCMNoiseSimulation | None = None
    uses_normalized_op: bool = True


SAPMPCMConfig = (
    SAPMSupervisedPCMConfig | SAPMSelfSupervisedPCMConfig | SAPMZeroShotPCMConfig
)


def make_supervised_pcm_training_cases(
    config: SAPMSupervisedPCMConfig,
) -> list[PCMAcquisitionSamplingCase]:
    """Resolve actual train acquisitions, including legacy interval configs."""
    j_order = config.data.j_order
    assert j_order is not None
    cases = make_acquisition_sampling_cases(j_order=j_order, sampling=config.sampling)
    variation = config.acquisition_variation
    if variation is not None and variation.sampling_ratio is not None:
        # Old experiments stored the sampling range next to noise, with
        # a nominal 20% case. Preserve their physics but correct their labels.
        if config.sampling.effective_ratio_range is not None or len(cases) != 1:
            raise ValueError("Migrate legacy ratio to sampling.sampling_ratio_range.")
        cases = [
            replace(
                case, sampling_ratio=None, sampling_ratio_range=variation.sampling_ratio
            )
            for case in cases
        ]
    if not cases:
        raise ValueError("PCM training requires at least one sampling case.")
    return cases


def make_supervised_pcm_validation_cases(
    config: SAPMSupervisedPCMConfig,
) -> list[PCMAcquisitionSamplingCase]:
    """Explicit validation sampling, or the actual training distribution."""
    if config.validation_sampling is None:
        return make_supervised_pcm_training_cases(config)
    j_order = config.data.j_order
    assert j_order is not None
    cases = make_acquisition_sampling_cases(
        j_order=j_order, sampling=config.validation_sampling
    )
    if not cases:
        raise ValueError("PCM validation requires at least one sampling case.")
    return cases


def supervised_pcm_scale_keys(config: SAPMSupervisedPCMConfig) -> tuple[int, ...]:
    """Range/shared settings have one learned scale, fixed legacy cases keep keys."""
    cases = make_supervised_pcm_training_cases(config)
    if config.sampling.share_scale_factor_across_ratios or any(
        case.sampling_ratio_range is not None for case in cases
    ):
        return (0,)
    return tuple(range(len(cases)))


def _load_pcm_sapm_common(path: Path) -> tuple[dict, dict, SamplesParameter, dict]:
    """Load fields shared by supervised, self-supervised, and zero-shot PCM SAPM configs."""
    data = load_yaml(path)

    data_dict = dict(data.get("data", {}))
    samples = make_dataclass(SamplesParameter, data_dict.pop("samples", {}))
    j_order = data_dict.get("j_order")
    assert j_order is not None

    common = dict(
        reproduce=make_dataclass(ReproduceConfig, data.get("reproduce", {})),
        model=make_sapm_model_config(data.get("model", {})),
        optim=make_dataclass(SAPMOptimConfig, data.get("optim", {})),
        train=make_dataclass(TrainConfig, data.get("train", {})),
        log=make_log_config(data.get("log", {})),
        pretrained=make_dataclass(PretrainedConfig, data.get("pretrained", {})),
        dirs=make_dataclass(RuntimeDirs, data.get("dirs", {})),
        sampling=make_dataclass(PCMSamplingParameter, data.get("sampling", {})),
        spgl1=make_dataclass(SPGL1Parameter, data.get("spgl1")),
        noise=load_pcm_noise_simulation(data.get("noise"), j_order=j_order),
        uses_normalized_op=bool(data.get("uses_normalized_op", True)),
        render_only=bool(data.get("render_only", False)),
        render_saved_results=bool(data.get("render_saved_results", True)),
        gen_lambda_maps=bool(data.get("gen_lambda_maps", False)),
        lambda_map_checkpoint_stems=list(data.get("lambda_map_checkpoint_stems") or []),
    )
    return data, data_dict, samples, common


def load_pcm_sapm_supervised_config_from_path(path: Path) -> SAPMSupervisedPCMConfig:
    """Load a supervised PCM TV/TGV SAPM config from YAML."""
    data, data_dict, samples, common = _load_pcm_sapm_common(path)
    return SAPMSupervisedPCMConfig(
        data=SupervisedPCMDataConfig(**data_dict, samples=samples),
        acquisition_variation=make_dataclass(
            PCMAcquisitionVariation, data.get("acquisition_variation")
        ),
        validation_sampling=make_dataclass(
            PCMSamplingParameter, data.get("validation_sampling")
        ),
        test_sampling=make_dataclass(PCMSamplingParameter, data.get("test_sampling")),
        **common,
    )


class SAPMSupervisedPCMConfigLoader(ExperimentConfigLoader[SAPMSupervisedPCMConfig]):
    def __call__(self, path: Path) -> SAPMSupervisedPCMConfig:
        return cast(
            SAPMSupervisedPCMConfig, load_pcm_sapm_supervised_config_from_path(path)
        )


load_pcm_sapm_supervised_config = SAPMSupervisedPCMConfigLoader()


def load_pcm_sapm_self_supervised_config_from_path(
    path: Path,
) -> SAPMSelfSupervisedPCMConfig:
    """Load a dataset self-supervised PCM TV/TGV SAPM config from YAML."""
    data, data_dict, samples, common = _load_pcm_sapm_common(path)
    ssdu = make_dataclass(PCMSSDUConfig, data.get("ssdu", {}))
    return SAPMSelfSupervisedPCMConfig(
        data=SupervisedPCMDataConfig(**data_dict, samples=samples),
        ssdu=ssdu,
        acquisition_variation=make_dataclass(
            PCMAcquisitionVariation, data.get("acquisition_variation")
        ),
        validation_sampling=make_dataclass(
            PCMSamplingParameter, data.get("validation_sampling")
        ),
        test_sampling=make_dataclass(PCMSamplingParameter, data.get("test_sampling")),
        **common,
    )


class SAPMSelfSupervisedPCMConfigLoader(
    ExperimentConfigLoader[SAPMSelfSupervisedPCMConfig]
):
    def __call__(self, path: Path) -> SAPMSelfSupervisedPCMConfig:
        return cast(
            SAPMSelfSupervisedPCMConfig,
            load_pcm_sapm_self_supervised_config_from_path(path),
        )


load_pcm_sapm_self_supervised_config = SAPMSelfSupervisedPCMConfigLoader()


def load_pcm_sapm_zero_shot_config_from_path(path: Path) -> SAPMZeroShotPCMConfig:
    """Load a zero-shot PCM TV/TGV SAPM config from YAML."""
    _, data_dict, samples, common = _load_pcm_sapm_common(path)
    return SAPMZeroShotPCMConfig(
        data=ZeroShotPCMDataConfig(**data_dict, samples=samples), **common
    )


class SAPMZeroShotPCMConfigLoader(ExperimentConfigLoader[SAPMZeroShotPCMConfig]):
    def __call__(self, path: Path) -> SAPMZeroShotPCMConfig:
        return cast(
            SAPMZeroShotPCMConfig, load_pcm_sapm_zero_shot_config_from_path(path)
        )


load_pcm_sapm_zero_shot_config = SAPMZeroShotPCMConfigLoader()
