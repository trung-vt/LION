"""Inspect actual randomized draws and resolution-dependent image-noise components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from LION.pcm.randomized.data import MixedImageCorpus, write_json
from LION.pcm.randomized.noise import NoiseRange, parameter_dict, sample_parameters
from LION.pcm.randomized.runner import write_csv
from LION.pcm_tools.noise_generate import generate_noise


def audit_runs(root: Path) -> dict:
    manifest = json.loads((root / "data_manifest.json").read_text())
    ids = {
        split: {r["sha256"] for r in rows} for split, rows in manifest["splits"].items()
    }
    if (
        ids["train"] & (ids["validation"] | ids["test"])
        or ids["validation"] & ids["test"]
    ):
        raise RuntimeError("Source-file content leakage across splits")
    summaries = []
    for directory in sorted((root / "runs").iterdir()):
        if not (directory / "run.json").is_file():
            continue
        run = json.loads((directory / "run.json").read_text())
        if run["status"] != "completed":
            continue
        draws = [
            json.loads(line)
            for line in (directory / "draws.jsonl").read_text().splitlines()
        ]
        if any(d["source_hash"] not in ids["train"] for d in draws):
            raise RuntimeError(f"Training leakage in {directory}")
        row = {
            "run": directory.name,
            "updates": run["completed_updates"],
            "draws": len(draws),
            "distinct_sources": len({d["source_hash"] for d in draws}),
            "natural_draws": sum(d["domain"] == "natural" for d in draws),
            "pcm_draws": sum(d["domain"] == "pcm_synthetic" for d in draws),
            "seconds": run["elapsed_seconds"],
        }
        for key in (
            "sigma_g",
            "sigma_bias",
            "sigma_beam",
            "sigma_dark",
            "residual_rms",
        ):
            values = [d[key] for d in draws]
            row[key + "_min"] = min(values)
            row[key + "_max"] = max(values)
        for key in ("requested_ratio", "coarse_j", "measurements"):
            values = [d[key] for d in draws if key in d]
            row[key + "_min"] = min(values) if values else None
            row[key + "_max"] = max(values) if values else None
        summaries.append(row)
    write_csv(root / "training_coverage.csv", summaries)
    result = {
        "source_content_splits_disjoint": True,
        "runs_audited": len(summaries),
        "coverage": summaries,
    }
    write_json(root / "integrity_audit.json", result)
    return result


def component_scaling(root: Path) -> None:
    """Do not mistake the innovation SD for the marginal trace or image-noise SD."""
    rows = []
    for side in (64, 256, 1024):
        for draw in range(8):
            rng = np.random.default_rng(77_000 + draw)
            p = sample_parameters(rng, NoiseRange())
            # Constant mean 0.5 isolates how trace length changes noise strength.
            intensity, i_bias = 0.5, 0.5 / 0.3
            dark, bias, beam = generate_noise(side * side, i_bias, p, rng)
            row = {
                "side": side,
                "pixels": side * side,
                "draw": draw,
                "sigma_dark": p.dark_sigma,
                "sigma_beam": p.beam_sigma,
                "beam_image_rms": float(np.mean((intensity * beam) ** 2) ** 0.5),
                "bias_image_rms": float(np.mean((bias - i_bias) ** 2) ** 0.5),
                "dark_image_rms": float(np.mean(dark**2) ** 0.5),
                "combined_rms": float(
                    np.mean((dark + bias - i_bias + intensity * beam) ** 2) ** 0.5
                ),
                "beam_terminal_std_proxy": float(abs(beam[-1])),
            }
            rows.append(row)
    write_csv(root / "image_noise_scaling.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit_runs(args.root)
    component_scaling(args.root)
    print(
        f"Audited {result['runs_audited']} completed runs; no source-file split leakage."
    )
