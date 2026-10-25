"""Acquisition sampling and validation labels for supervised PCM."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from LION.data_loaders.pcm.data_primitives import (
    PCMRandomRange,
    PCMSamplingParameter,
)
from LION.pcm.pcm_config import (
    load_pcm_sapm_supervised_config_from_path,
    make_supervised_pcm_training_cases,
    make_supervised_pcm_validation_cases,
    supervised_pcm_scale_keys,
)
from LION.pcm_tools.pcm_sampling import (
    make_acquisition_sampling_cases,
    make_sampling_cases,
)

BASE = Path(__file__).resolve().parents[2] / "config/pcm/pcm_utv_zero_inclusive.yaml"


def test_default_fixed_and_discrete_priority():
    assert PCMSamplingParameter().sampling_ratios == [0.2]
    ratio_range = PCMRandomRange(low=0.05, high=0.5, distribution="uniform")
    config = PCMSamplingParameter(
        sampling_ratios=[0.1, 0.3],
        sampling_ratio_range=ratio_range,
        coarse_j_values=[6],
    )
    assert config.effective_ratio_range is None
    assert make_sampling_cases(j_order=8, sampling=config) == [(0.1, 6), (0.3, 6)]
    cases = make_acquisition_sampling_cases(j_order=8, sampling=config)
    assert [case.label for case in cases] == ["s0.10_cJ6", "s0.30_cJ6"]


def test_range_labels_and_draws_and_no_silent_fixed_case():
    config = PCMSamplingParameter(
        sampling_ratios=[],
        sampling_ratio_range={"low": 0.05, "high": 0.5, "distribution": "uniform"},
        coarse_j_values=[6],
    )
    cases = make_acquisition_sampling_cases(j_order=8, sampling=config)
    assert len(cases) == 1
    assert cases[0].label == "s0.05-0.50_cJ6"
    rng = np.random.default_rng(193)
    assert all(0.05 <= cases[0].draw_ratio(rng) <= 0.5 for _ in range(100))
    with pytest.raises(ValueError, match="range"):
        make_sampling_cases(j_order=8, sampling=config)


def test_sampling_bounds_reject_invalid_range():
    with pytest.raises(ValueError, match="sampling_ratio_range"):
        PCMSamplingParameter(
            sampling_ratios=[],
            sampling_ratio_range={"low": 0.0, "high": 0.5, "distribution": "uniform"},
        )


def test_new_train_validate_test_independently():
    config = load_pcm_sapm_supervised_config_from_path(BASE)
    train = make_supervised_pcm_training_cases(config)
    val = make_supervised_pcm_validation_cases(config)
    assert [case.label for case in train] == ["s0.05-0.50_cJ6"]
    assert [case.label for case in val] == ["s0.05-0.50_cJ6"]
    assert supervised_pcm_scale_keys(config) == (0,)
    assert config.acquisition_variation.sampling_ratio is None
    assert config.test_sampling.sampling_ratios == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert config.acquisition_variation.noise_sigma_ranges["dark_sigma_raw"].low == 0
    independent_val = replace(
        config,
        validation_sampling=PCMSamplingParameter(
            sampling_ratios=[0.2], coarse_j_values=[6]
        ),
    )
    assert make_supervised_pcm_validation_cases(independent_val)[0].label == "s0.20_cJ6"
    assert (
        make_supervised_pcm_training_cases(independent_val)[0].label == "s0.05-0.50_cJ6"
    )


def test_legacy_ratio_and_fixed_multi_scale_keys(tmp_path):
    data = yaml.safe_load(BASE.read_text())
    data["sampling"] = {
        "randomising_scheme": "multilevel",
        "sampling_ratios": [0.2],
        "coarse_j_values": [6],
    }
    data.pop("validation_sampling")
    data.pop("test_sampling")
    data["acquisition_variation"]["sampling_ratio"] = {
        "low": 0.05,
        "high": 0.5,
        "distribution": "uniform",
    }
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(data))
    legacy = load_pcm_sapm_supervised_config_from_path(path)
    assert make_supervised_pcm_training_cases(legacy)[0].label == "s0.05-0.50_cJ6"
    assert supervised_pcm_scale_keys(legacy) == (0,)

    data.pop("acquisition_variation")
    data["sampling"]["sampling_ratios"] = [0.1, 0.2]
    path.write_text(yaml.safe_dump(data))
    old_fixed = load_pcm_sapm_supervised_config_from_path(path)
    assert [c.label for c in make_supervised_pcm_training_cases(old_fixed)] == [
        "s0.10_cJ6",
        "s0.20_cJ6",
    ]
    assert supervised_pcm_scale_keys(old_fixed) == (0, 1)


def test_tester_uses_explicit_test_sampling_not_validation(tmp_path):
    from LION.pcm.SupervisedPCMTester import (
        SupervisedPCMTester,
        SupervisedPCMTestConfig,
    )
    import torch

    experiment = tmp_path / "one_run"
    experiment.mkdir()
    (experiment / "config.yaml").write_text(BASE.read_text())
    config = SupervisedPCMTestConfig(
        root=tmp_path,
        experiment="one_run",
        epochs=["unused"],
        categories=["CIGS"],
        sample_ids=[611],
    )
    tester = SupervisedPCMTester(config, torch.device("cpu"))
    assert len(tester.selected_cases) == 6
    assert [ratio for key, ratio, j in tester.selected_cases] == [
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert {key for key, ratio, j in tester.selected_cases} == {0}
    assert len(tester.training_cases) == 1
    assert tester.training_cases[0].label == "s0.05-0.50_cJ6"
    custom = replace(config, sampling_ratios=[0.07, 0.14])
    tester2 = SupervisedPCMTester(custom, torch.device("cpu"))
    assert [round(r, 2) for _, r, _ in tester2.selected_cases] == [0.07, 0.14]


def test_training_and_fixed_validation_acquisition_reproducible(tmp_path):
    import torch

    from LION.config.RuntimeDirs import get_run_dirs
    from LION.pcm.SupervisedPCMTrainer import SupervisedPCMTrainer
    from LION.pcm.pcm_sapm import load_sapm_pcm_model, forward_func_sapm_pcm
    from LION.train.sapm import make_param_groups_sapm_common
    from LION.utils.reproduce import seed_from_parts

    config = load_pcm_sapm_supervised_config_from_path(BASE)
    # Same physics/resolution but tiny UNet and two PDHG iterations for CPU smoke.
    model = replace(
        config.model,
        cnn=replace(config.model.cnn, init_filters=8, n_blocks=2),
        n_iters=2,
        use_checkpoint=False,
    )
    config = replace(
        config,
        model=model,
        train=replace(config.train, device="cpu"),
        dirs=get_run_dirs(tmp_path, "run", mkdir=True),
    )
    trainer = SupervisedPCMTrainer(
        config,
        make_param_groups_sapm_common,
        load_sapm_pcm_model,
        forward_func_sapm_pcm,
        torch.nn.MSELoss(),
    )
    trainer.setup()
    assert trainer.logger.case_name(0) == "s0.05-0.50_cJ6"
    assert trainer.scale_keys == (0,)
    assert trainer.validation_scale_keys == (0,)
    torch.manual_seed(73)
    target = torch.rand(1, 256, 256)
    data = trainer.make_train_forward_data(
        target, key=0, rng=np.random.default_rng(3), device="cpu"
    )
    assert 0.05 <= data.sampling_ratio <= 0.5
    assert data.coarse_j == 6 and data.key == 0
    val_seed = seed_from_parts(
        "pcm_fixed_validation", str(config.reproduce.base_seed), "0"
    )
    first = trainer.make_val_forward_data(
        target, key=0, rng=np.random.default_rng(val_seed), device="cpu"
    )
    second = trainer.make_val_forward_data(
        target, key=0, rng=np.random.default_rng(val_seed), device="cpu"
    )
    assert first.sampling_ratio == second.sampling_ratio
    assert torch.equal(first.measurement, second.measurement)
    assert torch.equal(
        first.forward_op.op.sp.indices,
        second.forward_op.op.sp.indices,
    )
    fixed = replace(
        config,
        validation_sampling=PCMSamplingParameter(
            sampling_ratios=[0.2],
            coarse_j_values=[6],
            randomising_scheme="multilevel",
        ),
    )
    trainer.config = fixed
    trainer.validation_cases = make_supervised_pcm_validation_cases(fixed)
    assert trainer.validation_cases[0].label == "s0.20_cJ6"
    info = trainer.make_val_forward_data(
        target, key=0, rng=np.random.default_rng(val_seed), device="cpu"
    )
    assert info.sampling_ratio == 0.2 and info.key == 0
