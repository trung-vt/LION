"""Regression tests for the opt-in float64 oracle WLS acquisition path."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import sample_indices
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    create_float64_acquisition,
    load_png_float64,
    save_master_npz,
    solver_measurement,
)


def _fixture(tmp_path):
    root = tmp_path / "images"
    image_path = root / "CIGS" / "image0001.png"
    image_path.parent.mkdir(parents=True)
    pixels = np.full((256, 256), 128, dtype=np.uint8)
    pixels[33, 71] = 129
    Image.fromarray(pixels).save(image_path)
    row = {
        "split": "validation",
        "category": "CIGS",
        "permutation_position_1based": "901",
        "sample_id": "1",
        "source_relpath": "CIGS/image0001.png",
        "source_sha256": _sha256(image_path),
        "mask_seed_uint64": "1234567",
        "noise_seed_uint64": "7654321",
    }
    protocol = {
        "protocol_id": "pcm-oracle-wls-ols-v2-float64",
        "data": {"root": str(root)},
        "sampling": {
            "j_order": 8,
            "sampling_ratio_requested": 0.2,
            "coarse_j": 6,
            "sampling_scheme": "multilevel",
        },
        "noise": {
            "preset": "PCMNoiseSimulation.npl_default",
            "current_to_model_scale": 1.0 / 3.5e-8,
            "repeat_acquisitions": 1,
            "signal_to_bias_ratio": 0.3,
            "beam_ar": [-0.9221, -0.1709, -0.0769, -0.0354],
            "beam_ma": [0.0377, -0.3669],
            "beam_sigma": 1.2e-4,
            "bias_sigma": 2.9e-5,
            "bias_gradient_sigma": 1.6e-7,
            "dark_sigma": 3.61e-7 * (1.0 / 3.5e-8) / 256,
        },
        "precision": {"acquisition_compute_dtype": "float64"},
    }
    return image_path, row, protocol


def test_load_uint8_before_float64_normalisation(tmp_path):
    path, _, _ = _fixture(tmp_path)
    image = load_png_float64(path)
    assert image.dtype == torch.float64
    assert image.shape == (1, 1, 256, 256)
    assert image[0, 0, 33, 71] == torch.tensor(129 / 255, dtype=torch.float64)
    assert image[0, 0, 0, 0] == torch.tensor(128 / 255, dtype=torch.float64)


def test_full_noisy_simulation_is_float64_and_reproducible(tmp_path):
    _, row, protocol = _fixture(tmp_path)
    a, metadata = create_float64_acquisition(row=row, protocol=protocol)
    b, _ = create_float64_acquisition(row=row, protocol=protocol)
    assert a.measurement.dtype == torch.float64
    assert a.i0.dtype == torch.float64
    assert a.measurement.numel() == 13107
    assert len(a.sampled_indices) == 13107
    torch.testing.assert_close(a.measurement, b.measurement, rtol=0, atol=0)
    assert np.array_equal(a.sampled_indices, b.sampled_indices)
    assert metadata["acquisition_dtype"] == "float64"
    assert metadata["noise_seed_uint64"] == int(row["noise_seed_uint64"])


def test_no_noise_bias_cancellation_and_late_solver_cast(tmp_path):
    path, row, protocol = _fixture(tmp_path)
    x64 = load_png_float64(path)
    indices = sample_indices(
        j_order=8,
        sampling_ratio=0.2,
        coarse_j=6,
        randomising_scheme="multilevel",
        rng=np.random.default_rng(int(row["mask_seed_uint64"])),
    )
    noise = PCMNoiseSimulation.npl_default(current_to_model_scale=1 / 3.5e-8, j_order=8)
    zero = replace(
        noise,
        noise_params=replace(
            noise.noise_params,
            dark_sigma=0.0,
            bias_sigma=0.0,
            bias_gradient_sigma=0.0,
            beam_sigma=0.0,
        ),
    )
    kwargs = dict(j_order=8, sampled_indices=indices, noise=zero, device="cpu")
    a64 = simulate_pcm_acquisition(x64, rng=np.random.default_rng(7), **kwargs)
    a32 = simulate_pcm_acquisition(x64.float(), rng=np.random.default_rng(7), **kwargs)
    ref = a64.op(x64)
    non_dc = ref.reshape(-1)[1:]
    relative_error64 = (a64.measurement.reshape(-1)[1:] - non_dc).abs() / non_dc.abs()
    relative_error32 = (
        a32.measurement.reshape(-1)[1:].double() - non_dc
    ).abs() / non_dc.abs()
    assert float(relative_error64.max()) < 1e-6
    assert float((relative_error32 > 0.5).double().mean()) == 1.0

    # Convert only after subtracting the large split-pattern currents.
    solver_y = solver_measurement(a64, dtype=torch.float32)
    assert solver_y.dtype == torch.float32
    assert a64.measurement.dtype == torch.float64
    assert torch.count_nonzero(solver_y.reshape(-1)[1:]) == len(indices) - 1


def test_saved_master_is_float64_and_source_is_immutable(tmp_path):
    path, row, protocol = _fixture(tmp_path)
    checksum = _sha256(path)
    acquisition, metadata = create_float64_acquisition(row=row, protocol=protocol)
    saved = tmp_path / "pilot.npz"
    save_master_npz(saved, acquisition, metadata)
    with np.load(saved, allow_pickle=False) as bundle:
        assert bundle["measurement"].dtype == np.float64
        assert bundle["sampled_indices"].dtype == np.uint16
        assert bundle["i0"].dtype == np.float64
        np.testing.assert_array_equal(
            bundle["measurement"], acquisition.measurement.cpu().numpy()
        )
        stored_metadata = json.loads(str(bundle["metadata_json"]))
    assert stored_metadata["source_sha256"] == checksum
    assert _sha256(path) == checksum
    with pytest.raises(FileExistsError):
        save_master_npz(saved, acquisition, metadata)


def test_reject_mismatched_source_hash(tmp_path):
    path, row, protocol = _fixture(tmp_path)
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="Source image hash mismatch"):
        create_float64_acquisition(row=row, protocol=protocol)
