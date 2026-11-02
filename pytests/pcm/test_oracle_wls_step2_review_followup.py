"""Independent synthetic regression tests for the Step 2 reviewer findings.

The module creates 50 validation PNGs and 5 held-out PNGs in a temporary
directory. No original synthetic PCM images or frozen archives are modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from LION.pcm.noise import PCMNoiseSimulation
from scripts.experiments.pcm.oracle_wls import acquisition_archive as archive
from scripts.experiments.pcm.oracle_wls import float64_acquisition as storage
from pytests.pcm.test_oracle_wls_float64_acquisition import _fixture

CATEGORIES = ["CIGS", "Organic", "Perovskite", "Reference", "Silicone"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(split, category, sample_id, stream):
    key = (
        f"pcm-oracle-wls-ols-v1|master=20261102|split={split}|category={category}"
        f"|sample={sample_id:04d}|repeat=0|stream={stream}"
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def inputs(base):
    permutation = np.random.default_rng(5678).permutation(range(1, 1001))
    rows = []
    for c, category in enumerate(CATEGORIES):
        for pos in range(901, 912):
            split = "validation" if pos <= 910 else "test"
            sid = int(permutation[pos - 1])
            rel = f"{category}/image{sid:04d}.png"
            path = base / "data" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.random.default_rng(100 * c + pos).integers(
                40, 230, (256, 256), dtype=np.uint8
            )
            Image.fromarray(pixels).save(path)
            rows.append(
                dict(
                    split=split,
                    category=category,
                    permutation_position_1based=pos,
                    sample_id=sid,
                    source_relpath=rel,
                    source_sha256=sha(path),
                    mask_seed_uint64=seed(split, category, sid, "mask"),
                    noise_seed_uint64=seed(split, category, sid, "noise"),
                )
            )
    manifest = base / "source.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    noise = PCMNoiseSimulation.npl_default(current_to_model_scale=1 / 3.5e-8, j_order=8)
    protocol = {
        "protocol_id": "pcm-oracle-wls-ols-v2-float64",
        "data": {
            "root": str(base / "data"),
            "categories": CATEGORIES,
            "manifest_sha256": sha(manifest),
        },
        "sampling": {
            "j_order": 8,
            "sampling_scheme": "multilevel",
            "sampling_ratio_requested": 0.2,
            "coarse_j": 6,
            "multilevel_alpha": 1.0,
        },
        "noise": {
            "preset": "PCMNoiseSimulation.npl_default",
            "current_to_model_scale": 1 / 3.5e-8,
            "repeat_acquisitions": 1,
            **asdict(noise.noise_params),
            "signal_to_bias_ratio": noise.signal_to_bias_ratio,
        },
        "precision": {"acquisition_compute_dtype": "float64"},
    }
    parent = base / "parent.json"
    parent.write_text(json.dumps(protocol, indent=2) + "\n")
    return parent, manifest


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    base = tmp_path_factory.mktemp("step2_review_guard")
    parent, source = inputs(base)
    archive.build_archive(parent, source, base / "archive", device="cpu")
    return base


def writable_copy(complete, tmp_path):
    dest = tmp_path / "archive"
    shutil.copytree(complete / "archive", dest)
    for path in dest.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return dest


def rewrite_sha(root, relative):
    path = root / "SHA256SUMS"
    lines = path.read_text().splitlines()
    path.write_text(
        "\n".join(
            (
                f"{sha(root / relative)}  {relative}"
                if line.endswith(f"  {relative}")
                else line
            )
            for line in lines
        )
        + "\n"
    )


def test_review_archive_complete_and_replay(complete):
    root = complete / "archive"
    files = list((root / "validation").rglob("*.npz"))
    assert len(files) == 50
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o444 for f in files)
    assert list((root / "test").iterdir()) == []
    assert (root / "SHA256SUMS").is_file()
    recorded = json.loads((root / "manifest.json").read_text())
    assert recorded["category_counts"] == {category: 10 for category in CATEGORIES}
    result = archive.verify_archive(
        root, replay=True, expected_manifest_sha256=sha(root / "manifest.json")
    )
    assert result["status"] == "PASS"
    assert result["bit_exact_replays"] == 50
    with pytest.raises(FileExistsError):
        archive.build_archive(complete / "parent.json", complete / "source.csv", root)


def test_manifest_and_npz_consistent_edit_detected(complete, tmp_path):
    root = writable_copy(complete, tmp_path)
    manifest_path = root / "manifest.json"
    meta = json.loads(manifest_path.read_text())
    record = meta["acquisitions"][0]
    path = root / record["file"]
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key].copy() for key in handle.files}
    arrays["measurement"][0, 0, 5000] += 1e-3
    np.savez_compressed(path, **arrays)
    record["sha256"] = sha(path)
    record["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        archive.verify_archive(root)


def test_noise_parameter_disagreement_detected_even_if_checksums_updated(
    complete, tmp_path
):
    root = writable_copy(complete, tmp_path)
    protocol_path = root / "protocol.json"
    manifest_path = root / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["noise"]["dark_sigma"] *= 2
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_sha256"] = sha(protocol_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for relative in ("manifest.json", "protocol.json"):
        rewrite_sha(root, relative)
    with pytest.raises(ValueError, match="Noise parameters disagree"):
        archive.verify_archive(root)


def test_external_manifest_hash_defeats_consistent_sidecar_edit(complete, tmp_path):
    root = writable_copy(complete, tmp_path)
    original_manifest_sha = sha(root / "manifest.json")
    manifest_path = root / "manifest.json"
    doc = json.loads(manifest_path.read_text())
    doc["checked_by_reviewer"] = True
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n")
    rewrite_sha(root, "manifest.json")
    with pytest.raises(ValueError, match="External manifest SHA256"):
        archive.verify_archive(root, expected_manifest_sha256=original_manifest_sha)


def test_missing_test_folder_or_unexpected_non_npz_rejected(complete, tmp_path):
    root = writable_copy(complete, tmp_path)
    (root / "test" / "notes.txt").write_text("Unexpected")
    with pytest.raises(ValueError, match="test directory"):
        archive.verify_archive(root)
    (root / "test" / "notes.txt").unlink()
    (root / "test").rmdir()
    with pytest.raises(ValueError, match="test directory"):
        archive.verify_archive(root)


def test_missing_locked_noise_parameter_rejected_before_generation(tmp_path):
    _, row, protocol = _fixture(tmp_path)
    protocol["noise"].pop("dark_sigma")
    with pytest.raises(ValueError, match="Missing locked noise parameters"):
        storage.create_float64_acquisition(row=row, protocol=protocol)


def test_ordering_file_and_spyrit_dependency_recorded(complete):
    assert "LION/pcm_tools/ordering.py" in archive.CODE_FILES
    protocol = json.loads((complete / "archive" / "protocol.json").read_text())
    assert protocol["execution"]["spyrit"]
    assert protocol["executed_source_sha256"]["LION/pcm_tools/ordering.py"]
    case = next((complete / "archive" / "validation" / "CIGS").glob("*.npz"))
    with np.load(case, allow_pickle=False) as h:
        metadata = json.loads(str(h["metadata_json"]))
    assert len(metadata["sp_indices_sha256"]) == 64


def test_operator_permutation_change_is_caught(tmp_path, monkeypatch):
    _, row, protocol = _fixture(tmp_path)
    acquisition, metadata = storage.create_float64_acquisition(
        row=row, protocol=protocol
    )
    dest = tmp_path / "master.npz"
    storage.save_master_npz(dest, acquisition, metadata)
    monkeypatch.setattr(
        storage,
        "normal_to_dyadic_permutation",
        lambda j_order: np.arange(4**j_order),
    )
    with pytest.raises(ValueError, match="dyadic row order"):
        storage.load_master_npz(dest)


def test_one_case_cli_rejects_heldout_test(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "float64_acquisition.py",
            "--protocol",
            "unused.json",
            "--manifest",
            "unused.csv",
            "--split",
            "test",
            "--category",
            "CIGS",
            "--permutation-position",
            "911",
            "--output",
            "unused.npz",
        ],
    )
    with pytest.raises(SystemExit) as error:
        storage.main()
    assert error.value.code == 2


def test_trusted_case_loader_rejects_changed_npz(complete, tmp_path):
    root = writable_copy(complete, tmp_path)
    item = json.loads((root / "manifest.json").read_text())["acquisitions"][0]
    acq, md = archive.load_archived_case(root, item["category"], item["sample_id"])
    assert acq.measurement.dtype == torch.float64
    assert md["category"] == item["category"]
    p = root / item["file"]
    raw = bytearray(p.read_bytes())
    raw[-32] ^= 1
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256SUMS mismatch|checksum mismatch"):
        archive.load_archived_case(root, item["category"], item["sample_id"])


@pytest.mark.parametrize(
    "category, mask_seed, noise_seed, swapped",
    [
        ("CIGS", 10655734495037282845, 17327149251506351634, 6558),
        ("Organic", 16421984672317766477, 14292621417379134263, 6616),
        ("Perovskite", 9272213425636458974, 18408036670776216973, 6539),
        ("Reference", 9662329869901666358, 9354787048262327880, 6580),
        ("Silicone", 14335965865702207047, 9594105724508425934, 6464),
    ],
)
def test_reviewer_swap_seed_anchors(category, mask_seed, noise_seed, swapped):
    """Reproduce five independent reviewer-provided real-archive swap counts."""
    from LION.pcm_tools.noise_generate import (
        generate_noise,
        generate_pairwise_swap_indices,
    )

    assert seed("validation", category, 733, "mask") == mask_seed
    assert seed("validation", category, 733, "noise") == noise_seed
    noise = PCMNoiseSimulation.npl_default(current_to_model_scale=1 / 3.5e-8, j_order=8)
    rng = np.random.default_rng(noise_seed)
    generate_noise(
        num_measurements=2 * 13107,
        i_bias=1.0,
        noise_params=noise.noise_params,
        rng=rng,
    )
    indices = generate_pairwise_swap_indices(
        num_pairs=13107, rng=rng, device=torch.device("cpu")
    )
    assert int((indices[::2] % 2).sum()) == swapped


def test_review_pinned_dyadic_order_and_sample_mask():
    from LION.pcm_tools.ordering import normal_to_dyadic_permutation
    from LION.pcm_tools.pcm_sampling import sample_indices

    order = np.asarray(normal_to_dyadic_permutation(8), dtype="<i8")
    assert hashlib.sha256(order.tobytes()).hexdigest() == (
        "b78c3dfd0d35d70366082c27f00d071631e827328eb28e87c9c36a127fac105c"
    )
    mask = sample_indices(
        j_order=8,
        sampling_ratio=0.2,
        coarse_j=6,
        randomising_scheme="multilevel",
        rng=np.random.default_rng(10655734495037282845),
    )
    assert hashlib.sha256(np.asarray(mask, dtype="<i8").tobytes()).hexdigest() == (
        "403db641f053601337e1284525b490e0cbde519874a81ee72da22459412ca28c"
    )


def test_reviewer_flags_align_with_ramp_for_multiple_images_repeats(monkeypatch):
    from LION.pcm import simulation

    def ramp(*, num_measurements, i_bias, noise_params, rng):
        return (
            np.arange(num_measurements, dtype=np.float64),
            np.zeros(num_measurements),
            np.zeros(num_measurements),
        )

    monkeypatch.setattr(simulation, "generate_noise", ramp)
    source = torch.rand(
        2,
        1,
        16,
        16,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(3),
    )
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1 / 3.5e-8,
        j_order=4,
        n_repeat_acquisitions=2,
    )
    settings = {
        "j_order": 4,
        "sampled_indices": np.arange(3, 54),
        "device": "cpu",
    }
    clean = simulation.simulate_pcm_acquisition(
        source, noise=None, rng=np.random.default_rng(0), **settings
    )
    noisy = simulation.simulate_pcm_acquisition(
        source,
        noise=noise,
        rng=np.random.default_rng(9),
        record_swaps=True,
        **settings,
    )
    expected = (2 * noisy.swap_flags.to(torch.float64) - 1).mean(dim=-2)
    torch.testing.assert_close(
        noisy.measurement - clean.measurement,
        expected,
        rtol=0,
        atol=1e-12,
    )
