"""Out-of-training-set PCM reconstruction on fixed NPL-style simulated acquisitions.

The supplied current maps are measured images, not actual compressed/noisy
acquisitions. All under-sampled measurements are deliberately simulated from
those maps. Hyperparameters are frozen BEFORE examining their test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from LION.classical_algorithms.pdhg_direct import tv_pdhg
from LION.classical_algorithms.spgl1_torch import SPGL1Parameter
from LION.pcm.randomized.data import sha256, write_json
from LION.pcm.randomized.fixed_protocol import (
    NOISE_SCENARIOS,
    SAMPLING_RATIOS,
    fixed_acquisition,
    protocol_summary,
)
from LION.pcm.randomized.models import (
    build_denoiser,
    build_regularizer,
    regularized_reconstruction,
)
from LION.pcm.randomized.monitoring import image_grid
from LION.pcm.randomized.runner import metrics, synchronize
from LION.pcm.randomized.sweep import solve
from LION.pcm.reconstruct import spgl1_reconstruction

METHODS = (
    "zero_fill",
    "tv_fixed_002",
    "spgl1",
    "utv_supervised",
    "utv_ssdu",
    "utgv_supervised",
    "utgv_ssdu",
    "pnp_admm_dwtdnet",
    "gs_pnp_drunet",
)
MODELS = {
    "utv_supervised": "utv_supervised_physical_pcm",
    "utv_ssdu": "utv_ssdu_physical_pcm",
    "utgv_supervised": "utgv_supervised_physical_pcm",
    "utgv_ssdu": "utgv_ssdu_physical_pcm",
    "pnp_admm_dwtdnet": "dwtdnet_mixture",
    "gs_pnp_drunet": "gs_drunet_mixture",
}
FROZEN_PNP_CONFIGS = {
    "pnp_admm_dwtdnet": {"algorithm": "pnp_admm", "rho": 1.0, "sigma": 0.2},
    "gs_pnp_drunet": {
        "algorithm": "gs_proximal_gradient",
        "lambda": 1.0,
        "tau": 0.5,
        "sigma": 0.2,
    },
}


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(set(r) for r in rows)))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _load_model(name: str, side: int, device: torch.device, saved_runs: Path):
    ref = MODELS[name]
    checkpoint = saved_runs / ref / "best.pt"
    if name.startswith("pnp") or name.startswith("gs_pnp"):
        architecture = "dwtdnet" if name == "pnp_admm_dwtdnet" else "gs_drunet"
        model = build_denoiser(architecture, side, device)
    else:
        model = build_regularizer(name, side, iterations=64, device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("completed_updates", 0) < 1:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint}")
    trained_state = dict(state["model_state_dict"])
    # Window attention buffers encode the input resolution, but are deterministic
    # non-trainable buffers. Preserve freshly instantiated 256x256 attention masks.
    destination_state = model.state_dict()
    adapted_masks = []
    for key in tuple(trained_state):
        if (
            key.endswith(".attn_mask")
            and key in destination_state
            and trained_state[key].shape != destination_state[key].shape
        ):
            if key in dict(model.named_parameters()):
                raise RuntimeError(f"Refusing to drop trainable tensor {key}")
            del trained_state[key]
            adapted_masks.append(key)
    loaded = model.load_state_dict(trained_state, strict=False)
    if sorted(loaded.missing_keys) != sorted(adapted_masks) or loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {loaded}")
    model.eval().requires_grad_(False)
    return model, {
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "trained_image_side": 64,
        "completed_updates": state["completed_updates"],
        "regenerated_resolution_dependent_masks": adapted_masks,
    }


def _run_method(method: str, op, y, model):
    if method == "zero_fill":
        return op.adjoint(y), []
    if method == "tv_fixed_002":
        x = tv_pdhg(
            measurement=y,
            initial_image=op.adjoint(y),
            forward_op=op,
            lambda_reg=0.02,
            max_iterations=256,
            primal_stepsize=0.2,
            dual_stepsize=0.2,
            use_tqdm=False,
        )
        return x, []
    if method == "spgl1":
        param = SPGL1Parameter(
            factor=1e5,
            iters=100,
            debias_iters=10,
            debias_support_tol=1e-5,
            debias_tol=1e-7,
            wavelet_name="db4",
        )
        x = spgl1_reconstruction(op, y, param, device=y.device, shows_prog_bar=False)
        return x, []
    if method in ("pnp_admm_dwtdnet", "gs_pnp_drunet"):
        return solve(op, y, model, FROZEN_PNP_CONFIGS[method], 20)
    return regularized_reconstruction(model, op, y), []


def evaluate_unseen(
    manifest: Path,
    saved_runs: Path,
    output: Path,
    *,
    case_labels: tuple[str, ...] = ("cigs_active_full", "perovskite_center_256"),
    scenarios: tuple[str, ...] = NOISE_SCENARIOS,
    ratios: tuple[float, ...] = SAMPLING_RATIOS,
    methods: tuple[str, ...] = METHODS,
    device: str = "cuda:0",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experimental results: {output}")
    output.mkdir(parents=True)
    source = json.loads(manifest.read_text())
    case_records = [
        c for c in source["cases"] if c["cell"] + "_" + c["label"] in case_labels
    ]
    if len(case_records) != len(case_labels):
        raise ValueError(f"Requested cases not found: {case_labels}")
    for c in case_records:
        if c["shape"] != [256, 256] or sha256(Path(c["path"])) != c["sha256"]:
            raise RuntimeError("Wrong crop shape or SHA256 mismatch")
    if any(s not in NOISE_SCENARIOS for s in scenarios) or any(
        r not in SAMPLING_RATIOS for r in ratios
    ):
        raise ValueError("Unknown fixed condition")
    if any(m not in METHODS for m in methods):
        raise ValueError("Unknown method")
    torch.set_num_threads(2)
    dev = torch.device(device)
    models = {}
    weights = {}
    for method in methods:
        if method in MODELS:
            models[method], weights[method] = _load_model(method, 256, dev, saved_runs)
    protocol = {
        "status": "running",
        "purpose": "held-out real-map images reconstructed from SYNTHETIC undersampled PCM measurements",
        "not_real_acquisition": True,
        "not_trained_on_unseen": True,
        "test_set": [c["cell"] + "_" + c["label"] for c in case_records],
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "fixed_protocol": protocol_summary(8),
        "scenarios": scenarios,
        "ratios": ratios,
        "methods": methods,
        "checkpoint_provenance": weights,
        "frozen_params": {
            "tv_fixed_002": {
                "lambda": 0.02,
                "iterations": 256,
                "tau": 0.2,
                "sigma": 0.2,
            },
            "spgl1": {
                "factor": 1e5,
                "iters": 100,
                "debias_iters": 10,
                "wavelet": "db4",
            },
            **FROZEN_PNP_CONFIGS,
        },
        "scientific_limits": [
            "Only two physical specimens; perovskite center crop has unknown agreement with supplied screenshot crop",
            "Per-cell contrast scaling uses a 99.5 percentile; not an absolute calibration",
            "The learned networks were trained briefly on 64x64 crops; no adaptation to the unseen data",
            "Same image serves as simulation input and reference; no truly independently measured compressed acquisitions",
            "No tuning or checkpoint selection on this unseen test set",
        ],
    }
    write_json(output / "protocol.json", protocol)
    rows: list[dict] = []
    failures: list[dict] = []
    snapshot = output / "reconstructions"
    snapshot.mkdir()
    try:
        for case_index, c in enumerate(case_records):
            clean = torch.from_numpy(np.load(c["path"])).to(dev)[None, None]
            for ratio in ratios:
                for scenario in scenarios:
                    acquisition, info = fixed_acquisition(
                        clean,
                        split="unseen",
                        case_index=case_index,
                        ratio=ratio,
                        scenario=scenario,
                    )
                    op, y = acquisition.op, acquisition.measurement
                    images = [
                        ("Reference map", clean.detach().cpu().numpy()),
                        ("Zero-filled", op.adjoint(y).detach().cpu().numpy()),
                    ]
                    for method in methods:
                        try:
                            synchronize(dev)
                            start = time.perf_counter()
                            with torch.no_grad():
                                pred, trace = _run_method(
                                    method, op, y, models.get(method)
                                )
                            synchronize(dev)
                            duration = time.perf_counter() - start
                            error = float((op(pred) - y).square().mean().sqrt())
                            m = metrics(pred, clean)
                            if not all(math.isfinite(v) for v in m.values()):
                                raise FloatingPointError("Invalid image quality")
                            record = {
                                "cell": c["cell"],
                                "crop": c["label"],
                                "case_index": case_index,
                                "method": method,
                                "sampling_ratio": ratio,
                                "scenario": scenario,
                                "mask_sha256": info["mask_sha256"],
                                "seed_mask": info["seed_mask"],
                                "seed_noise": info["seed_noise"],
                                "coarse_j": info["sampling"]["coarse_j"],
                                "num_measurements": info["sampling"]["measurements"],
                                "measurement_noise_rms": info[
                                    "realized_measurement_residual_rms"
                                ],
                                "measurement_residual_rms": error,
                                "seconds": duration,
                                "iterations": trace[-1]["iteration"] if trace else None,
                                "stop_reason": trace[-1]["stop_reason"]
                                if trace
                                else "",
                                **m,
                            }
                            rows.append(record)
                            write_rows(output / "reconstruction_metrics.csv", rows)
                            if ratio == 0.2 and scenario in (
                                "noiseless",
                                "pcm_default_10ms",
                                "pcm_short_100us",
                            ):
                                images.append(
                                    (
                                        f"{method} ({m['psnr']:.1f} dB)",
                                        pred.detach().cpu().numpy(),
                                    )
                                )
                            print(
                                f"CASE {case_index} {ratio:.0%} {scenario} {method}: "
                                f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} time={duration:.2f}s",
                                flush=True,
                            )
                        except (
                            RuntimeError,
                            ValueError,
                            FloatingPointError,
                            AssertionError,
                            torch.cuda.OutOfMemoryError,
                        ) as exc:
                            failures.append(
                                {
                                    "cell": c["cell"],
                                    "crop": c["label"],
                                    "ratio": ratio,
                                    "scenario": scenario,
                                    "method": method,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            write_rows(output / "failures.csv", failures)
                            print(
                                f"FAIL {case_index} {ratio} {scenario} {method}: {exc}",
                                flush=True,
                            )
                            if dev.type == "cuda":
                                torch.cuda.empty_cache()
                    if ratio == 0.2 and scenario in (
                        "noiseless",
                        "pcm_default_10ms",
                        "pcm_short_100us",
                    ):
                        fname = f"{c['cell']}_{c['label']}_{scenario}_r020"
                        image_grid(snapshot / f"{fname}.png", [(fname, images)])
        protocol.update(
            status="completed", reconstructions=len(rows), failures=len(failures)
        )
        write_json(output / "protocol.json", protocol)
        return protocol
    except Exception as exc:
        protocol.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "protocol.json", protocol)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", default=["cigs_active_full", "perovskite_center_256"]
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=NOISE_SCENARIOS, default=list(NOISE_SCENARIOS)
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=list(SAMPLING_RATIOS)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_unseen(
                args.manifest,
                args.runs,
                args.output,
                case_labels=tuple(args.cases),
                scenarios=tuple(args.scenarios),
                ratios=tuple(args.ratios),
                methods=tuple(args.methods),
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
