"""Freeze PCM validation/test acquisition tensors as checksummed NPZ files.

This exports the *actual* targets, sampled Hadamard rows, simulated noisy
coefficients, seeds and source hashes. It does not train, tune or reconstruct.
All generated files use the fixed_protocol.py noise/sampling contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from LION.pcm.randomized.data import MixedImageCorpus, sha256, write_json
from LION.pcm.randomized.fixed_protocol import (
    NOISE_SCENARIOS,
    SAMPLING_RATIOS,
    fixed_acquisition,
    protocol_summary,
)


def _save_npz_exact(path: Path, *, image, measurements, indices):
    """Write atomically. An existing acquisition is never silently replaced."""
    expected = {
        "reference": np.asarray(image, dtype=np.float32),
        "measurement": np.asarray(measurements, dtype=np.float32),
        "sampled_indices": np.asarray(indices, dtype=np.int64),
    }
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            for key, values in expected.items():
                if not np.array_equal(old[key], values):
                    raise RuntimeError(
                        f"Existing immutable acquisition differs: {path} ({key})"
                    )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **expected)
    temp.replace(path)


def generate(
    source: Path,
    destination: Path,
    *,
    split: str,
    device: str = "cuda:1",
    per_domain: int = 4,
    historical_metrics: Path | None = None,
) -> dict:
    """Save a reproducible dataset; fail on mismatch with an existing file."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This experiment explicitly requires a free CUDA GPU")
    torch.set_num_threads(2)
    torch_device = torch.device(device)
    if split == "unseen":
        source_doc = json.loads(source.read_text())
        chosen = [
            x
            for x in source_doc["cases"]
            if (x["cell"], x["label"])
            in (("cigs", "active_full"), ("perovskite", "center_256"))
        ]
        if len(chosen) != 2:
            raise RuntimeError("Expected exactly two fixed unseen image crops")
        cases = []
        for record in chosen:
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("Unseen source hash mismatch")
            img = torch.from_numpy(np.load(record["path"], allow_pickle=False))[
                None, None
            ]
            cases.append(
                (
                    img,
                    {
                        "domain": record["cell"],
                        "sha256": record["sha256"],
                        "label": record["cell"] + "_" + record["label"],
                        "source_file": record["path"],
                    },
                )
            )
    elif split in ("validation", "test"):
        corpus = MixedImageCorpus(source, side=64)
        cases = [
            (img, dict(record, label=f"{split}_{i:02d}_{record['domain']}"))
            for i, (img, record) in enumerate(corpus.fixed_cases(split, per_domain))
        ]
        if len(cases) != per_domain * 2:
            raise RuntimeError("Expected equal fixed natural and synthetic panels")
    else:
        raise ValueError("Invalid frozen split")

    expected_count = len(cases) * len(SAMPLING_RATIOS) * len(NOISE_SCENARIOS)
    if expected_count > 256:
        raise ValueError("Acquisition count exceeds safe 256-case maximum")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    original = pd.read_csv(historical_metrics) if historical_metrics else None
    for i, (target_cpu, source_record) in enumerate(cases):
        target = target_cpu.to(device=torch_device, dtype=torch.float32)
        side = target.shape[-1]
        for ratio in SAMPLING_RATIOS:
            for scenario in NOISE_SCENARIOS:
                acquisition, info = fixed_acquisition(
                    target,
                    split=split,
                    case_index=i,
                    ratio=ratio,
                    scenario=scenario,
                )
                name = f"case_{i:02d}_r{int(round(ratio*100)):03d}_{scenario}"
                file = destination / (name + ".npz")
                _save_npz_exact(
                    file,
                    image=target.detach().cpu().numpy(),
                    measurements=acquisition.measurement.detach().cpu().numpy(),
                    indices=acquisition.sampled_indices,
                )
                data = {
                    "case": i,
                    "source": source_record["label"],
                    "domain": source_record["domain"],
                    "source_sha256": source_record["sha256"],
                    "case_file": file.name,
                    "case_file_sha256": sha256(file),
                    "scenario": scenario,
                    "sampling_ratio": ratio,
                    "side": side,
                    "noise_seed": info["seed_noise"],
                    "mask_seed": info["seed_mask"],
                    "mask_sha256": info["mask_sha256"],
                    "coarse_j": info["sampling"]["coarse_j"],
                    "measurement_count": info["sampling"]["measurements"],
                    "noise_parameters": info["noise_params"],
                    "realized_noise_rms": info["realized_measurement_residual_rms"],
                }
                if original is not None:
                    expected = original[
                        (original["case_index"] == i)
                        & np.isclose(original["sampling_ratio"], ratio)
                        & (original["scenario"] == scenario)
                    ]
                    if len(expected) != len(original["method"].unique()):
                        raise RuntimeError("Missing original comparison rows")
                    if (expected["mask_sha256"] != info["mask_sha256"]).any():
                        raise AssertionError("Masks differ from the original benchmark")
                    historical_rms = expected["measurement_noise_rms"].to_numpy(
                        dtype=float
                    )
                    delta = np.max(np.abs(historical_rms - data["realized_noise_rms"]))
                    if delta > 2e-5:
                        raise AssertionError(
                            f"Noise differs from original test acquisition: {name} delta={delta}"
                        )
                    data["max_noise_rms_difference_vs_original"] = float(delta)
                records.append(data)
        print(
            f"{split}: persisted {i + 1}/{len(cases)} source images ({side}x{side})",
            flush=True,
        )

    image_meta = {
        i: {"name": record["label"], "sha256": record["sha256"]}
        for i, (_, record) in enumerate(cases)
    }
    manifest = {
        "split": split,
        "status": "completed",
        "frozen": True,
        "archive_format": "NPZ containing reference float32, measurement float32, sampled_indices int64",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256(source),
        "number_of_images": len(cases),
        "number_of_acquisitions": len(records),
        "images": image_meta,
        "protocol": protocol_summary(cases[0][0].shape[-1].bit_length() - 1),
        "test_images_never_train_or_tune": True,
        "dtype": "float32",
        "comparison_to_earlier_metrics": "same masks and realized noise RMS within 2e-5"
        if original is not None
        else "not checked against historical benchmark",
        "acquisitions": records,
    }
    path = destination / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise RuntimeError(f"An existing frozen dataset manifest differs: {path}")
    else:
        write_json(path, manifest)
    return {
        "destination": str(destination),
        "split": split,
        "cases": len(records),
        "source_images": len(cases),
        "manifest_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test", "unseen"), required=True
    )
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--compare-original-metrics", type=Path)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.source_manifest,
                args.output,
                split=args.split,
                device=args.device,
                per_domain=args.per_domain,
                historical_metrics=args.compare_original_metrics,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
