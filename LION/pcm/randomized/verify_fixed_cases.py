"""Validate immutable PCM validation/test/unseen NPZ acquisitions and manifests.

No GPU is needed: this reads and hashes only existing test artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from LION.pcm.randomized.data import sha256
from LION.pcm.randomized.fixed_protocol import NOISE_SCENARIOS, SAMPLING_RATIOS


def verify(root: Path) -> dict:
    counts = {}
    split_sources = {}
    for split, n in (("validation", 96), ("test", 96), ("unseen", 24)):
        directory = root / split
        m = directory / "manifest.json"
        manifest = json.loads(m.read_text())
        assert manifest["status"] == "completed", split
        assert manifest["split"] == split, split
        assert manifest["number_of_acquisitions"] == n, split
        assert len(manifest["acquisitions"]) == n, split
        assert tuple(manifest["protocol"]["sampling_ratios"]) == SAMPLING_RATIOS
        assert set(manifest["protocol"]["noise_scenarios"]) == set(NOISE_SCENARIOS)
        known = set()
        masks_by_case = {}
        sources = set()
        for entry in manifest["acquisitions"]:
            path = directory / entry["case_file"]
            assert path.is_file(), path
            assert sha256(path) == entry["case_file_sha256"], path
            key = (entry["case"], entry["sampling_ratio"], entry["scenario"])
            assert key not in known, key
            known.add(key)
            sources.add(entry["source_sha256"])
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {"reference", "measurement", "sampled_indices"}
                reference = z["reference"]
                measurements = z["measurement"]
                indices = z["sampled_indices"]
            side = entry["side"]
            assert reference.shape == (1, 1, side, side), path
            count = int(np.ceil(entry["sampling_ratio"] * side**2))
            assert len(indices) == count
            assert len(np.unique(indices)) == count
            assert indices[0] == 0 and np.array_equal(indices, np.sort(indices))
            assert measurements.size == count
            assert np.isfinite(reference).all() and np.isfinite(measurements).all()
            assert np.issubdtype(indices.dtype, np.integer)
            mask_key = (entry["case"], entry["sampling_ratio"])
            if mask_key in masks_by_case:
                assert np.array_equal(indices, masks_by_case[mask_key]), mask_key
            else:
                masks_by_case[mask_key] = indices.copy()
            assert entry["coarse_j"] == side.bit_length() - 3, entry
        assert len(known) == n
        assert len(sources) == manifest["number_of_images"]
        counts[split] = n
        split_sources[split] = sources
    assert split_sources["validation"].isdisjoint(split_sources["test"])
    return {
        "status": "verified",
        "acquisitions": counts,
        "total_acquisitions": sum(counts.values()),
        "validation_test_source_hashes_disjoint": True,
        "mask_same_across_noise_conditions": True,
        "unique_rows_dc_exact_count": True,
        "file_sha256_verified": True,
        "reference_and_measurement_finite": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.root), indent=2))


if __name__ == "__main__":
    main()
