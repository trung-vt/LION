"""Archive all 216 frozen unseen-PCM reconstructions and contact sheets.

Uses previously saved immutable NPZ test acquisitions. No training or tuning.
Checks each result against the original fixed-condition benchmark CSV and
records any differences (including GPU precision/nondeterminism).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.randomized.data import sha256, write_json
from LION.pcm.randomized.figure_style import image_style, save_pcm_image
from LION.pcm.randomized.runner import metrics, synchronize
from LION.pcm.randomized.unseen_benchmark import (
    METHODS,
    _load_model,
    _run_method,
    write_rows,
)
from LION.pcm.randomized.unseen_analysis import METHOD_LABELS, SCENARIO_LABELS


def _png(path: Path, array: np.ndarray, *, error: bool = False) -> None:
    """Save an unchanged numeric image via LION's existing PCM render style."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a prediction preview: {path}")
    save_pcm_image(
        path,
        np.asarray(array),
        kind="absolute_error" if error else "reconstruction",
    )


def _montage(
    folder: Path,
    label: str,
    pictures: dict[str, np.ndarray],
    rows: list[dict],
    *,
    error: bool = False,
) -> None:
    """All nine methods plus the reference, with independent title/colorbar space."""
    order = ("reference",) + METHODS
    by_method = {row["method"]: row for row in rows}
    kind = "absolute_error" if error else "reconstruction"
    cmap, vmin, vmax = image_style(kind)
    fig, axes = plt.subplots(2, 5, figsize=(17.3, 8.3))
    last_image = None
    for axis, key in zip(axes.flat, order, strict=True):
        values = np.asarray(pictures[key]).squeeze()
        if error:
            values = np.abs(values - np.asarray(pictures["reference"]).squeeze())
        last_image = axis.imshow(
            values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        if key == "reference":
            title = "Reference (zero error)" if error else "Reference image"
        else:
            record = by_method[key]
            title = (
                f"{METHOD_LABELS[key]}\n"
                f"PSNR {record['psnr']:.2f} dB | SSIM {record['ssim']:.3f}"
            )
        axis.set_title(title, fontsize=11.5, pad=10)
        axis.axis("off")
    match = re.fullmatch(r"case_(\d{2})_r(\d{3})_(.+)", label)
    if match:
        case, rate, noise_id = match.groups()
        source = {"00": "CIGS", "01": "Perovskite centre"}.get(case, f"Case {case}")
        title = f"{source} | {int(rate)}% sampling | {SCENARIO_LABELS[noise_id]}"
    else:
        title = label.replace("_", " ")
    fig.suptitle(
        title + (" | absolute error" if error else " | reconstructions"),
        fontsize=13.5,
        y=0.97,
    )
    # Reserve enough height for the two-line headers of BOTH image rows,
    # the separate figure title and the common horizontal colorbar.
    fig.subplots_adjust(
        left=0.025, right=0.975, top=0.85, bottom=0.145, hspace=0.32, wspace=0.10
    )
    bar_axis = fig.add_axes((0.21, 0.065, 0.58, 0.022))
    colorbar = fig.colorbar(last_image, cax=bar_axis, orientation="horizontal")
    colorbar.set_label(
        "Absolute error, transformed units"
        if error
        else "Transformed current (relative units)",
        fontsize=11,
    )
    target = folder / (label + ("_abs_error" if error else "_recon") + ".png")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing montage: {target}")
    fig.savefig(target, dpi=135, facecolor="white")
    plt.close(fig)


def archive(
    frozen_manifest: Path,
    original_metrics: Path,
    saved_runs: Path,
    output: Path,
    *,
    device: str = "cuda:1",
) -> dict:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Run only on an available CUDA device")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay: {output}")
    manifest = json.loads(frozen_manifest.read_text())
    if manifest["split"] != "unseen" or manifest["number_of_acquisitions"] != 24:
        raise ValueError("Must supply exactly the frozen 24 unseen test conditions")
    initial = pd.read_csv(original_metrics)
    if len(initial) != 216 or set(initial.method.unique()) != set(METHODS):
        raise ValueError("Original comparison set is not the complete nine-method grid")
    output.mkdir(parents=True)
    raw_folder = output / "arrays"
    png_folder = output / "images"
    diff_folder = output / "abs_errors"
    grids = output / "montages"
    for folder in (raw_folder, png_folder, diff_folder, grids):
        folder.mkdir()
    torch.set_num_threads(2)
    dev = torch.device(device)
    models = {}
    ckpts = {}
    for method in METHODS:
        if method in ("zero_fill", "spgl1", "tv_fixed_002"):
            continue
        models[method], ckpts[method] = _load_model(method, 256, dev, saved_runs)
    report = {
        "status": "running",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(dev),
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": sha256(frozen_manifest),
        "original_metrics": str(original_metrics),
        "original_metrics_sha256": sha256(original_metrics),
        "checkpoint_sha256_by_method": {
            method: source["sha256"] for method, source in ckpts.items()
        },
        "no_training_no_test_tuning": True,
        "recon_display": "AFM hot [0,1]; numerical metrics use raw float32 arrays",
        "error_display": "Viridis; shared transformed-image error range [0,0.5]",
        "predictor_order": METHODS,
        "replay_mse_match_tolerance": 1e-3,
    }
    write_json(output / "protocol.json", report)
    rows = []
    cases = []
    try:
        for record in manifest["acquisitions"]:
            name = Path(record["case_file"]).stem
            acquisition_path = frozen_manifest.parent / record["case_file"]
            if sha256(acquisition_path) != record["case_file_sha256"]:
                raise RuntimeError("Saved acquisition has a different SHA256")
            with np.load(acquisition_path, allow_pickle=False) as data:
                reference = data["reference"].copy()
                measured = data["measurement"].copy()
                sampled_indices = data["sampled_indices"].copy()
            truth = torch.as_tensor(reference, device=dev)
            y = torch.as_tensor(measured, device=dev)
            side = truth.shape[-1]
            op = PhotocurrentMapOp(
                j_order=side.bit_length() - 1,
                s_indices=sampled_indices,
                normalized=True,
                device=dev,
            )
            if len(op.sp.indices) != len(sampled_indices):
                raise RuntimeError("Inconsistent measurement count")
            if record["scenario"] == "noiseless":
                mismatch = float((op(truth) - y).abs().max())
                if mismatch > 5e-5:
                    raise RuntimeError(f"Noiseless acquisition changed: {mismatch}")
            image_dict = {"reference": reference}
            one_case = []
            for method in METHODS:
                synchronize(dev)
                start_time = time.perf_counter()
                with torch.no_grad():
                    prediction_tensor, _trace = _run_method(
                        method, op, y, models.get(method)
                    )
                synchronize(dev)
                elapsed = time.perf_counter() - start_time
                predicted = prediction_tensor.detach().float().cpu().numpy()
                if not np.isfinite(predicted).all():
                    raise FloatingPointError(f"Invalid prediction {name} {method}")
                scores = metrics(prediction_tensor, truth)
                original = initial[
                    (initial.case_index == record["case"])
                    & np.isclose(initial.sampling_ratio, record["sampling_ratio"])
                    & (initial.scenario == record["scenario"])
                    & (initial.method == method)
                ]
                if len(original) != 1:
                    raise RuntimeError(f"Missing original metric: {name} {method}")
                old = original.iloc[0]
                if old["mask_sha256"] != record["mask_sha256"]:
                    raise AssertionError("Saved mask does not match original run")
                mse_difference = abs(scores["mse"] - float(old["mse"]))
                row = {
                    "case_file": record["case_file"],
                    "case_file_sha256": record["case_file_sha256"],
                    "case_index": record["case"],
                    "cell": record["source"],
                    "ratio": record["sampling_ratio"],
                    "scenario": record["scenario"],
                    "method": method,
                    "seconds": elapsed,
                    **scores,
                    "historical_mse": float(old["mse"]),
                    "historical_psnr": float(old["psnr"]),
                    "mse_diff_to_original": mse_difference,
                    "matches_original_within_1e3": mse_difference <= 1e-3,
                }
                folder = raw_folder / name
                folder.mkdir(exist_ok=True)
                array_path = folder / (method + ".npy")
                np.save(array_path, predicted.astype(np.float32), allow_pickle=False)
                row["array_sha256"] = sha256(array_path)
                rows.append(row)
                one_case.append(row)
                _png(png_folder / name / (method + ".png"), predicted)
                _png(
                    diff_folder / name / (method + ".png"),
                    np.abs(predicted - reference),
                    error=True,
                )
                image_dict[method] = predicted
            _png(png_folder / name / "reference.png", reference)
            _montage(grids, name, image_dict, one_case)
            _montage(grids, name, image_dict, one_case, error=True)
            cases.append(name)
            write_rows(output / "all_reconstructions.csv", rows)
            print(f"ARCHIVED {len(cases)}/24: {name}, 9 methods", flush=True)
        mismatched = [r for r in rows if not r["matches_original_within_1e3"]]
        report.update(
            status="completed",
            cases=len(cases),
            reconstructions=len(rows),
            saved_pngs=len(list(png_folder.rglob("*.png"))),
            saved_float_arrays=len(list(raw_folder.rglob("*.npy"))),
            montage_count=len(list(grids.glob("*_recon.png"))),
            error_montage_count=len(list(grids.glob("*_abs_error.png"))),
            mismatch_cases=len(mismatched),
            maximum_mse_replay_difference=max(x["mse_diff_to_original"] for x in rows),
        )
        write_json(output / "protocol.json", report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            successful_reconstructions=len(rows),
            complete_cases=len(cases),
        )
        write_json(output / "protocol.json", report)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--original-metrics", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    print(
        json.dumps(
            archive(
                args.frozen_manifest,
                args.original_metrics,
                args.runs,
                args.output,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
