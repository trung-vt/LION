"""Replay frozen, validation-selected PnP test cases with per-iteration diagnostics.

This is test-only instrumentation: the original training runs and selected
hyperparameters are untouched. Ground truth is read only for reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from LION.pcm.randomized.data import MixedImageCorpus, sha256, write_json
from LION.pcm.randomized.models import build_denoiser
from LION.pcm.randomized.monitoring import convergence_figures, image_grid
from LION.pcm.randomized.runner import metrics, write_csv
from LION.pcm.randomized.sweep import make_cases, solve


def replay(
    root: Path,
    *,
    device: str = "cuda:0",
    iterations: int | None = None,
    per_domain: int | None = None,
) -> dict:
    """Measure actual image quality/relative updates for each accepted PnP iterate."""
    sweep = root / "pnp_sweep"
    protocol = json.loads((sweep / "protocol.json").read_text())
    if protocol["status"] != "completed":
        raise RuntimeError("PnP tuning must finish before convergence replay")
    manifest = root / "data_manifest.json"
    if sha256(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Saved manifest differs from the one used for tuning")
    iterations = protocol["iterations"] if iterations is None else iterations
    per_domain = protocol["per_domain"] if per_domain is None else per_domain
    if iterations != protocol["iterations"] or per_domain != protocol["per_domain"]:
        raise ValueError("Replay must use original iteration budget and case count")
    folder = sweep / "convergence"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite convergence replay: {folder}")
    folder.mkdir()
    snapshots_dir = folder / "snapshots"
    snapshots_dir.mkdir()
    write_json(
        folder / "protocol.json",
        {
            "status": "running",
            "source_pnp_protocol": str((sweep / "protocol.json").resolve()),
            "purpose": "test-only numerical diagnostics; does not affect selection or weights",
            "initialization": {"pnp_admm": "zeros", "gs_proximal_gradient": "adjoint"},
            "metric_reference": "held-out clean test crop, never used for selection",
            "snapshot_iterations": [0, 1, 5, 10, 20, "final"],
            "timing_note": "Per-iteration CPU metrics introduce overhead, not directly comparable to original inference timing",
        },
    )
    torch.set_num_threads(2)
    dev = torch.device(device)
    corpus = MixedImageCorpus(manifest, side=64)
    selected = json.loads((sweep / "selected.json").read_text())
    original_test = pd.read_csv(sweep / "test.csv")
    rows: list[dict] = []
    summary: list[dict] = []
    try:
        for model_name in protocol["models"]:
            chk = root / "runs" / f"{model_name}_mixture" / "best.pt"
            saved = torch.load(chk, map_location=dev, weights_only=False)
            model = build_denoiser(model_name, 64, dev).eval()
            model.load_state_dict(saved["model_state_dict"])
            model.requires_grad_(False)
            for condition in selected:
                if condition["model"] != model_name:
                    continue
                ratio, dark, parameters = (
                    condition["ratio"],
                    condition["dark_sigma"],
                    condition["config"],
                )
                cases = make_cases(corpus, "test", ratio, dark, per_domain, dev)
                for case, (clean, record, acquisition, info) in enumerate(cases):
                    snapshots: dict[str, np.ndarray] = {}
                    prediction, trace = solve(
                        acquisition.op,
                        acquisition.measurement,
                        model,
                        parameters,
                        iterations,
                        reference=clean,
                        snapshots=snapshots,
                    )
                    final = metrics(prediction, clean)
                    existing = original_test[
                        (original_test.model == model_name)
                        & np.isclose(original_test.ratio, ratio)
                        & np.isclose(original_test.dark_sigma, dark)
                        & (original_test.case == case)
                    ]
                    if len(existing) != 1:
                        raise RuntimeError("Missing or ambiguous original test case")
                    old = existing.iloc[0]
                    if old["source_hash"] != record["sha256"]:
                        raise AssertionError(
                            "Held-out source hash changed during replay"
                        )
                    replay_difference = abs(float(old["mse"]) - final["mse"])
                    # GS finite precision and the explicit line-search stopping
                    # rule may yield small differences. Record all mismatches,
                    # never silently claim exact replay.
                    replay_match = replay_difference <= 2e-4
                    key = f"{model_name}_r{ratio:.2f}_dark{dark:.2f}_case{case:02d}"
                    snapshots["final"] = prediction.detach().cpu().numpy()
                    np.savez_compressed(
                        snapshots_dir / f"{key}.npz",
                        clean=clean.detach().cpu().numpy(),
                        zero_filled=acquisition.op.adjoint(acquisition.measurement)
                        .detach()
                        .cpu()
                        .numpy(),
                        **{
                            "iteration_" + step: value
                            for step, value in snapshots.items()
                        },
                    )
                    for item in trace:
                        rows.append(
                            {
                                "model": model_name,
                                "ratio": ratio,
                                "dark_sigma": dark,
                                "case": case,
                                "domain": record["domain"],
                                "source_hash": record["sha256"],
                                "coarse_j": info["coarse_j"],
                                **item,
                            }
                        )
                    summary.append(
                        {
                            "model": model_name,
                            "ratio": ratio,
                            "dark_sigma": dark,
                            "case": case,
                            "domain": record["domain"],
                            "source_hash": record["sha256"],
                            "iterations_completed": trace[-1]["iteration"],
                            "stop_reason": trace[-1]["stop_reason"],
                            "initial_psnr": trace[0]["psnr"],
                            "final_psnr": final["psnr"],
                            "final_ssim": final["ssim"],
                            "original_final_mse": float(old["mse"]),
                            "replayed_final_mse": final["mse"],
                            "final_mse_abs_difference": replay_difference,
                            "matches_original_tolerance": replay_match,
                        }
                    )
                    if (
                        np.isclose(ratio, 0.2)
                        and np.isclose(dark, 0.2)
                        and case in (0, per_domain)
                    ):
                        grid = [
                            ("Ground truth", clean.detach().cpu().numpy()),
                            (
                                "Zero-fill",
                                acquisition.op.adjoint(acquisition.measurement)
                                .detach()
                                .cpu()
                                .numpy(),
                            ),
                        ]
                        for step in ("0", "1", "5", "10", "20", "final"):
                            if step in snapshots:
                                grid.append((f"Iter {step}", snapshots[step]))
                        image_grid(
                            folder / "figures" / f"{key}.png",
                            [(record["domain"], grid)],
                        )
                    # Publish progress for the dashboard as the replay proceeds.
                    write_csv(folder / "convergence_test.csv", rows)
                    write_csv(folder / "convergence_cases.csv", summary)
            del model, saved
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        generated = convergence_figures(root, folder)
        result = json.loads((folder / "protocol.json").read_text())
        result.update(
            status="completed",
            cases=len(summary),
            trace_rows=len(rows),
            replay_mismatch_cases=sum(
                not row["matches_original_tolerance"] for row in summary
            ),
            max_final_mse_difference=max(
                row["final_mse_abs_difference"] for row in summary
            ),
            static_figures=generated,
        )
        write_json(folder / "protocol.json", result)
        return result
    except Exception as exc:
        result = json.loads((folder / "protocol.json").read_text())
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder / "protocol.json", result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(replay(args.root, device=args.device), indent=2))


if __name__ == "__main__":
    main()
