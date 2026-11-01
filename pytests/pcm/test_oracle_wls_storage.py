"""Step 2 tests. Set PCM_TEST_DEVICE=cuda:0 to exercise the approved GPU."""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.pcm_tools.noise_generate import add_noise
from scripts.experiments.pcm.oracle_wls import float64_acquisition as storage
from scripts.experiments.pcm.oracle_wls.acquisition_archive import validation_rows
from pytests.pcm.test_oracle_wls_float64_acquisition import _fixture

DEVICE = os.environ.get("PCM_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_optional_capture_preserves_measurements_and_rng(dtype, repeats, batch):
    x = torch.linspace(0.01, 0.99, batch * 16 * 16, dtype=dtype).reshape(
        batch, 1, 16, 16
    )
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8,
        j_order=4,
        n_repeat_acquisitions=repeats,
    )
    rng0, rng1 = np.random.default_rng(77), np.random.default_rng(77)
    args = dict(j_order=4, sampled_indices=np.arange(51), noise=noise, device=DEVICE)
    ordinary = simulate_pcm_acquisition(x, rng=rng0, **args)
    recorded = simulate_pcm_acquisition(x, rng=rng1, record_swaps=True, **args)
    torch.testing.assert_close(
        ordinary.measurement, recorded.measurement, rtol=0, atol=0
    )
    torch.testing.assert_close(ordinary.i0, recorded.i0, rtol=0, atol=0)
    assert ordinary.swap_flags is None
    assert recorded.swap_flags.shape == (batch, 1, repeats, 51)
    assert recorded.swap_flags.dtype == torch.bool
    assert recorded.swap_flags.device.type == "cpu"
    assert rng0.bit_generator.state == rng1.bit_generator.state


def test_recorded_flags_have_the_correct_noise_time_assignment():
    m = 17
    expected_flags = np.random.default_rng(813).integers(0, 2, size=m)
    trace = torch.arange(2 * m, dtype=torch.float64, device=DEVICE)
    flags = []
    plus, minus, _ = add_noise(
        y_clean_plus=torch.zeros(m, dtype=torch.float64, device=DEVICE),
        y_clean_minus=torch.zeros(m, dtype=torch.float64, device=DEVICE),
        e_abs_trace=trace,
        e_rel=torch.zeros_like(trace),
        rng=np.random.default_rng(813),
        swap_flags_out=flags,
    )
    np.testing.assert_array_equal(flags[0].numpy(), expected_flags.astype(bool))
    np.testing.assert_array_equal(plus.cpu().numpy(), 2 * np.arange(m) + expected_flags)
    np.testing.assert_array_equal(
        minus.cpu().numpy(), 2 * np.arange(m) + 1 - expected_flags
    )


def test_clean_path_has_no_invented_swaps():
    rng = np.random.default_rng(81)
    state = rng.bit_generator.state
    acquired = simulate_pcm_acquisition(
        torch.ones(1, 1, 4, 4, dtype=torch.float64),
        j_order=2,
        sampled_indices=np.arange(16),
        noise=None,
        rng=rng,
        device=DEVICE,
        record_swaps=True,
    )
    assert acquired.swap_flags is None
    assert rng.bit_generator.state == state


def test_repeated_batched_flags_follow_the_same_acquisition_order():
    x = torch.arange(2 * 16 * 16, dtype=torch.float64).reshape(2, 1, 16, 16) / 512
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8, j_order=4, n_repeat_acquisitions=3
    )
    args = dict(
        j_order=4, sampled_indices=np.arange(51), device=DEVICE, record_swaps=True
    )
    batch = simulate_pcm_acquisition(
        x, noise=noise, rng=np.random.default_rng(22), **args
    )
    rng = np.random.default_rng(22)
    for i in range(2):
        repeats = [
            simulate_pcm_acquisition(
                x[i : i + 1],
                noise=replace(noise, n_repeat_acquisitions=1),
                rng=rng,
                **args,
            )
            for _ in range(3)
        ]
        torch.testing.assert_close(
            batch.measurement[i : i + 1],
            torch.stack([r.measurement for r in repeats]).mean(0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batch.swap_flags[i : i + 1],
            torch.cat([r.swap_flags for r in repeats], dim=-2),
            rtol=0,
            atol=0,
        )


def _saved(tmp_path):
    _, row, protocol = _fixture(tmp_path)
    acquisition, metadata = storage.create_float64_acquisition(
        row=row, protocol=protocol, device=DEVICE
    )
    path = tmp_path / "master.npz"
    storage.save_master_npz(path, acquisition, metadata)
    return path, acquisition, metadata


def test_exact_roundtrip_with_bit_padding_and_runtime_int64_indices(tmp_path):
    path, acquisition, metadata = _saved(tmp_path)
    loaded, actual = storage.load_master_npz(
        path, expected_sha256=storage._sha256(path), device=DEVICE
    )
    assert actual == metadata
    assert loaded.sampled_indices.dtype == np.int64
    assert loaded.measurement.dtype == torch.float64
    torch.testing.assert_close(
        loaded.measurement, acquisition.measurement, rtol=0, atol=0
    )
    torch.testing.assert_close(
        loaded.swap_flags, acquisition.swap_flags, rtol=0, atol=0
    )
    with np.load(path, allow_pickle=False) as z:
        assert z["swap_flags_packed"].shape == (1, (13107 + 7) // 8)
        assert z["swap_flags_packed"].dtype == np.uint8
        assert int(z["swap_flags_packed"][0, -1]) & 0b11111000 == 0
    assert path.stat().st_mode & 0o222 == 0


def test_loader_never_resimulates_and_solver_copies_do_not_mutate_master(
    tmp_path, monkeypatch
):
    path, _, _ = _saved(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading must not simulate a new acquisition")

    monkeypatch.setattr(storage, "simulate_pcm_acquisition", forbidden)
    a, _ = storage.load_master_npz(path)
    b, _ = storage.load_master_npz(path)
    torch.testing.assert_close(a.measurement, b.measurement, rtol=0, atol=0)
    np.testing.assert_array_equal(a.sampled_indices, b.sampled_indices)
    reference = a.measurement.clone()
    for dtype in (torch.float32, torch.float64):
        solver = storage.solver_measurement(a, dtype=dtype)
        solver.zero_()
        torch.testing.assert_close(a.measurement, reference, rtol=0, atol=0)


def test_checksum_rejects_changed_file(tmp_path):
    path, _, _ = _saved(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        storage.load_master_npz(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "corruption", ["float32", "nan", "duplicate", "padding", "schema"]
)
def test_invalid_archive_is_rejected_without_a_checksum(tmp_path, corruption):
    path, _, _ = _saved(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if corruption == "float32":
        arrays["measurement"] = arrays["measurement"].astype(np.float32)
    elif corruption == "nan":
        arrays["measurement"][0, 0, 0] = np.nan
    elif corruption == "duplicate":
        arrays["sampled_indices"][1] = arrays["sampled_indices"][0]
    elif corruption == "padding":
        arrays["swap_flags_packed"][0, -1] |= np.uint8(128)
    else:
        md = json.loads(str(arrays["metadata_json"]))
        md["archive_schema"] = "wrong"
        arrays["metadata_json"] = json.dumps(md)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError):
        storage.load_master_npz(bad)


def test_partial_write_leaves_no_final_file_or_temporary(tmp_path, monkeypatch):
    _, row, protocol = _fixture(tmp_path)
    acquisition, metadata = storage.create_float64_acquisition(
        row=row, protocol=protocol
    )

    def fail(handle, **kwargs):
        handle.write(b"partial data")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(storage.np, "savez_compressed", fail)
    target = tmp_path / "interrupted.npz"
    with pytest.raises(OSError, match="interrupted"):
        storage.save_master_npz(target, acquisition, metadata)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_no_overwrite_even_through_an_existing_symlink(tmp_path):
    path, acquisition, metadata = _saved(tmp_path)
    digest = storage._sha256(path)
    link = tmp_path / "alias.npz"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        storage.save_master_npz(link, acquisition, metadata)
    assert storage._sha256(path) == digest


def test_locked_noise_parameter_mismatch_is_not_silently_ignored(tmp_path):
    _, row, protocol = _fixture(tmp_path)
    protocol["noise"]["beam_sigma"] = 0.123
    with pytest.raises(ValueError, match="locked parameter"):
        storage.create_float64_acquisition(row=row, protocol=protocol)


def _selection_csv(extra=False):
    stream = io.StringIO()
    fields = ["split", "category", "permutation_position_1based", "sample_id"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    categories = ["CIGS", "Organic", "Perovskite", "Reference", "Silicone"]
    for category in categories:
        for position in range(901, 911):
            writer.writerow(
                dict(
                    split="validation",
                    category=category,
                    permutation_position_1based=position,
                    sample_id=position,
                )
            )
    writer.writerow(
        dict(
            split="test",
            category="CIGS",
            permutation_position_1based=911,
            sample_id=911,
        )
    )
    if extra:
        writer.writerow(
            dict(
                split="validation",
                category="CIGS",
                permutation_position_1based=901,
                sample_id=901,
            )
        )
    return stream.getvalue().encode(), {"data": {"categories": categories}}


def test_archive_selection_never_includes_heldout_rows():
    data, protocol = _selection_csv()
    selected = validation_rows(data, protocol)
    assert len(selected) == 50
    assert all(r["split"] == "validation" for r in selected)


def test_archive_selection_rejects_duplicate_or_incomplete_validation():
    data, protocol = _selection_csv(extra=True)
    with pytest.raises(ValueError):
        validation_rows(data, protocol)
    data, protocol = _selection_csv()
    with pytest.raises(ValueError):
        validation_rows(
            data.replace(b"validation,CIGS,901,901", b"test,CIGS,901,901"), protocol
        )
