"""Step 4: independent Monte Carlo validation of Step 3 marginal oracle variances.

The noise samples use the *existing* LION generate_noise/add_noise functions,
not the oracle formula. For each image A and x stay fixed, while fresh
independent noise and pair-swap draws are made.

The frozen Step 2 and Step 3 archives are only read. No held-out test images
or acquisition data are accessed. Runs use CPU/float64 and write a new
versioned validation-results folder with an external manifest digest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import _sample_split_measurements, simulate_pcm_acquisition
from LION.pcm_tools.noise_generate import add_noise, generate_noise
from scripts.experiments.pcm.oracle_wls.build_oracle_weights import verify_weights
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    _write_atomic_new,
    load_master_npz,
    load_png_float64,
    write_json_new,
)

SCHEMA = "pcm-oracle-wls-monte-carlo-v1"
COMPONENT_MODES = {
    "dark": "variance_dark",
    "bias_white": "variance_bias_white",
    "bias_drift": "variance_bias_drift",
    "beam": "variance_beam",
}
COMPONENT_STD = {
    "dark": "dark_sigma",
    "bias_white": "bias_sigma",
    "bias_drift": "bias_gradient_sigma",
    "beam": "beam_sigma",
}
# Declared before running experiments; conservative limits for 1024+ independent
# realizations of approximately Gaussian (but not necessarily Gaussian-mixture)
# per-measurement noise. Correlated measurement coordinates are not independent.
ACCEPTANCE = {
    "min_repeats_full": 1024,
    "min_repeats_control": 512,
    "full_median_abs_variance_rel_error_max": 0.09,
    "full_p95_abs_variance_rel_error_max": 0.18,
    "full_mean_variance_ratio_min": 0.93,
    "full_mean_variance_ratio_max": 1.07,
    "full_fraction_abs_mean_z_below_3point5_min": 0.98,
    "control_median_abs_variance_rel_error_max": 0.14,
    "control_p95_abs_variance_rel_error_max": 0.27,
    "control_mean_variance_ratio_min": 0.87,
    "control_mean_variance_ratio_max": 1.13,
}


def independent_mc_seed(category: str, sample_id: int, mode: str) -> int:
    """Use a new seeded stream, unrelated to Step 2's frozen noise seed."""
    payload = (
        f"{SCHEMA}|master=20261102|"
        f"category={category}|sample={sample_id:04d}|mode={mode}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def frozen_case(
    acquisition_root: Path,
    weights_root: Path,
    case_record: dict,
    weight_record: dict,
    acquisition_protocol: dict,
):
    """Read fixed clean x/A, saved weights, and validate source identities."""
    acquired, acquisition_metadata = load_master_npz(
        acquisition_root / case_record["file"],
        expected_sha256=case_record["sha256"],
    )
    if acquired.measurement.dtype != torch.float64:
        raise ValueError("Frozen noisy acquisitions must remain float64")
    image_path = (
        Path(acquisition_protocol["data"]["root"])
        / acquisition_metadata["source_relpath"]
    )
    if _sha256(image_path) != acquisition_metadata["source_sha256"]:
        raise ValueError("Original source image has changed")
    x = load_png_float64(image_path)
    p, q = _sample_split_measurements(x, acquired.op)
    p, q = p.reshape(-1).contiguous(), q.reshape(-1).contiguous()
    clean = p - q
    torch.testing.assert_close(clean, acquired.op(x).reshape(-1), rtol=0, atol=2e-12)
    with np.load(weights_root / weight_record["file"], allow_pickle=False) as f:
        note = json.loads(str(f["metadata_json"]))
        variance = {
            key: f[key].astype(np.float64)
            for key in (
                "variance",
                "variance_dark",
                "variance_bias_white",
                "variance_bias_drift",
                "variance_beam",
            )
        }
        indices = f["sampled_indices"].copy()
    if (
        note["source_acquisition_sha256"] != case_record["sha256"]
        or note["source_png_sha256"] != acquisition_metadata["source_sha256"]
        or not np.array_equal(indices, acquired.sampled_indices)
    ):
        raise ValueError("Analytical weights do not match the selected acquisition")
    if any(value.shape != (len(p),) for value in variance.values()):
        raise ValueError("Unexpected oracle-variance vector shape")
    return acquired, x, p, q, clean, variance


def noise_parameters_for_mode(default, mode: str):
    """Turn off the other components without changing the noise-generating code."""
    if mode == "full":
        return default
    if mode not in COMPONENT_MODES:
        raise ValueError(f"Unsupported noise component: {mode}")
    toggles = {
        key: (getattr(default, key) if key == COMPONENT_STD[mode] else 0.0)
        for key in COMPONENT_STD.values()
    }
    return replace(default, **toggles)


def one_realization(
    plus: torch.Tensor,
    minus: torch.Tensor,
    i0: float,
    params,
    rng: np.random.Generator,
) -> np.ndarray:
    """One full fresh simulated trace, preserving actual RNG and swap semantics."""
    m = len(plus)
    e_dark, e_bias, e_rel = generate_noise(
        num_measurements=2 * m,
        i_bias=i0,
        noise_params=params,
        rng=rng,
    )
    noisy_plus, noisy_minus, _ = add_noise(
        y_clean_plus=plus,
        y_clean_minus=minus,
        e_abs_trace=torch.as_tensor(e_dark + e_bias, dtype=torch.float64),
        e_rel=torch.as_tensor(e_rel, dtype=torch.float64),
        rng=rng,
    )
    # Subtract the clean differential value BEFORE accumulating moments.
    # Float64 avoids catastrophic cancellation of the large physical bias.
    return (noisy_plus - noisy_minus - (plus - minus)).detach().cpu().numpy().copy()


def streaming_moments(draws, *, n: int):
    """Float64 Welford mean and unbiased variance, without retaining raw draws."""
    if n < 2:
        raise ValueError("At least two independent repetitions are required")
    mu = None
    m2 = None
    t = 0
    for t, draw in enumerate(draws, start=1):
        e = np.asarray(draw, dtype=np.float64)
        if mu is None:
            mu = np.zeros_like(e)
            m2 = np.zeros_like(e)
        if e.shape != mu.shape or not np.isfinite(e).all():
            raise ValueError("Invalid or nonfinite simulated residual")
        diff = e - mu
        mu += diff / t
        m2 += diff * (e - mu)
    if t != n:
        raise ValueError("The generator returned the wrong realization count")
    return mu, m2 / (n - 1)


def numerical_diagnostics(
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    predicted_variance: np.ndarray,
    *,
    sample_count: int,
    mode: str,
):
    """Summarise independently sampled noise with fixed predeclared thresholds."""
    if not (
        empirical_variance.shape == predicted_variance.shape == empirical_mean.shape
        and np.isfinite(empirical_variance).all()
        and np.isfinite(predicted_variance).all()
        and np.all(predicted_variance > 0)
    ):
        raise ValueError("Invalid empirical or oracle arrays")
    ratio = empirical_variance / predicted_variance
    absolute_relative = np.abs(ratio - 1)
    # Pointwise zero-mean diagnostic. Correlated measurement coordinates must
    # not be treated as independent replicates or independent hypothesis tests.
    mean_z = empirical_mean * np.sqrt(sample_count / predicted_variance)
    report = {
        "sample_count": int(sample_count),
        "mode": mode,
        "variance_ratio_mean": float(np.mean(ratio)),
        "variance_ratio_median": float(np.median(ratio)),
        "median_abs_variance_relative_error": float(np.median(absolute_relative)),
        "p95_abs_variance_relative_error": float(np.quantile(absolute_relative, 0.95)),
        "max_abs_variance_relative_error": float(np.max(absolute_relative)),
        "mean_z_median_abs": float(np.median(np.abs(mean_z))),
        "mean_z_p95_abs": float(np.quantile(np.abs(mean_z), 0.95)),
        "fraction_abs_mean_z_below_3point5": float(np.mean(np.abs(mean_z) < 3.5)),
        "residual_mean_rms": float(np.sqrt(np.mean(empirical_mean**2))),
        "predicted_noise_std_rms": float(np.sqrt(np.mean(predicted_variance))),
    }
    control = mode != "full"
    prefix = "control_" if control else "full_"
    low = ACCEPTANCE[prefix + "mean_variance_ratio_min"]
    high = ACCEPTANCE[prefix + "mean_variance_ratio_max"]
    conditions = [
        report["median_abs_variance_relative_error"]
        <= ACCEPTANCE[prefix + "median_abs_variance_rel_error_max"],
        report["p95_abs_variance_relative_error"]
        <= ACCEPTANCE[prefix + "p95_abs_variance_rel_error_max"],
        low <= report["variance_ratio_mean"] <= high,
        report["fraction_abs_mean_z_below_3point5"]
        >= ACCEPTANCE["full_fraction_abs_mean_z_below_3point5_min"],
    ]
    report["predeclared_checks_passed"] = bool(all(conditions))
    report["predeclared_checks"] = [bool(x) for x in conditions]
    return report


def _npz_case(output, arrays):
    """Create exactly one compressed per-case evidence file; never overwrite."""
    file = Path(output)
    file.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic_new(file, lambda f: np.savez_compressed(f, **arrays))
    file.chmod(0o444)


def _locked_noise(noise_settings: dict):
    """Reject a silently changed nominal NPL noise preset."""
    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=noise_settings["current_to_model_scale"],
        j_order=8,
    )
    for key, expected in asdict(noise.noise_params).items():
        if json.dumps(expected) != json.dumps(noise_settings[key]):
            raise ValueError(f"Noise preset changed: {key}")
    if noise.signal_to_bias_ratio != noise_settings["signal_to_bias_ratio"]:
        raise ValueError("Bias ratio differs from locked Step 2 parameters")
    return noise


def run_validation(
    acquisition_root,
    weight_root,
    output,
    *,
    expected_acquisition_manifest_sha256,
    expected_weight_manifest_sha256,
    repetitions=1024,
    control_repetitions=768,
    cases_per_category=2,
    enable_controls=True,
):
    """Run the predeclared, zero-tuning, fixed-sensing Monte Carlo experiment."""
    source, weights, destination = map(Path, (acquisition_root, weight_root, output))
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Step 4 directory {destination}"
        )
    if repetitions < ACCEPTANCE["min_repeats_full"]:
        raise ValueError("Production validation requires at least 1024 repetitions")
    if enable_controls and control_repetitions < ACCEPTANCE["min_repeats_control"]:
        raise ValueError("Component controls require at least 512 repetitions")
    if cases_per_category < 1 or cases_per_category > 10:
        raise ValueError("Choose between 1 and 10 validation images per category")
    verify_weights(
        source,
        weights,
        expected_acquisition_manifest_sha256=expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=expected_weight_manifest_sha256,
    )
    source_manifest = json.loads((source / "manifest.json").read_text())
    weight_manifest = json.loads((weights / "manifest.json").read_text())
    source_protocol = json.loads((source / "protocol.json").read_text())
    by_weight = {
        (r["category"], int(r["sample_id"])): r for r in weight_manifest["files"]
    }
    category_names = list(source_protocol["data"]["categories"])
    selected = []
    for category in category_names:
        candidates = sorted(
            (r for r in source_manifest["acquisitions"] if r["category"] == category),
            key=lambda x: int(x["permutation_position_1based"]),
        )
        if len(candidates) != 10:
            raise ValueError("Missing approved validation category images")
        selected.extend(candidates[:cases_per_category])
    params = _locked_noise(source_protocol["noise"]).noise_params

    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for index, case in enumerate(selected, start=1):
        key = (case["category"], int(case["sample_id"]))
        acq, x, p, q, clean, oracle_components = frozen_case(
            source,
            weights,
            case,
            by_weight[key],
            source_protocol,
        )
        i0 = float(acq.i0.reshape(-1)[0])
        # Separate deterministic streams per case and per noise component.
        modes = (
            ["full"] + list(COMPONENT_MODES)
            if enable_controls and (index - 1) % cases_per_category == 0
            else ["full"]
        )
        for mode in modes:
            n = repetitions if mode == "full" else control_repetitions
            seed = independent_mc_seed(case["category"], int(case["sample_id"]), mode)
            noise_params = noise_parameters_for_mode(params, mode)
            rng = np.random.default_rng(seed)
            draws = (one_realization(p, q, i0, noise_params, rng) for _ in range(n))
            empirical_mean, empirical_var = streaming_moments(draws, n=n)
            target_key = "variance" if mode == "full" else COMPONENT_MODES[mode]
            predicted_var = oracle_components[target_key]
            report = numerical_diagnostics(
                empirical_mean,
                empirical_var,
                predicted_var,
                sample_count=n,
                mode=mode,
            )
            filename = (
                f"validation/{case['category']}/"
                f"sample_{case['sample_id']:04d}_{mode}_mc.npz"
            )
            arrays = {
                "predicted_variance": predicted_var,
                "empirical_variance": empirical_var,
                "empirical_mean": empirical_mean,
                "sampled_indices": np.asarray(acq.sampled_indices, dtype=np.uint16),
                "metadata_json": json.dumps(
                    {
                        "schema": SCHEMA,
                        "category": case["category"],
                        "sample_id": int(case["sample_id"]),
                        "mode": mode,
                        "repetitions": n,
                        "mc_seed_decimal": str(seed),
                        "step2_acquisition_sha256": case["sha256"],
                        "step3_weight_sha256": by_weight[key]["sha256"],
                    },
                    sort_keys=True,
                ),
            }
            _npz_case(destination / filename, arrays)
            record = {
                "file": filename,
                "sha256": _sha256(destination / filename),
                "bytes": (destination / filename).stat().st_size,
                "category": case["category"],
                "sample_id": int(case["sample_id"]),
                "mc_seed_decimal": str(seed),
                "step2_acquisition_sha256": case["sha256"],
                "step3_weight_sha256": by_weight[key]["sha256"],
                **report,
            }
            records.append(record)
            print(
                f"{index:02d}/{len(selected)} {key} {mode} n={n}: "
                f"ratio_mean={report['variance_ratio_mean']:.5f}, "
                f"p95_rel={report['p95_abs_variance_relative_error']:.4f}, "
                f"PASS={report['predeclared_checks_passed']}",
                flush=True,
            )
    protocol = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection": (
            f"first {cases_per_category} validation permutation positions "
            "901 onward, independently in each of five categories"
        ),
        "repetitions": repetitions,
        "control_repetitions": control_repetitions if enable_controls else 0,
        "tested_cases": len(selected),
        "control_cases": len(category_names) if enable_controls else 0,
        "steps_2_3_data_read_only": True,
        "test_acquisitions_read": 0,
        "sampling_mask_reused_not_regenerated": True,
        "step2_manifest_sha256": expected_acquisition_manifest_sha256,
        "step3_manifest_sha256": expected_weight_manifest_sha256,
        "step2_protocol_sha256": _sha256(source / "protocol.json"),
        "step3_protocol_sha256": _sha256(weights / "protocol.json"),
        "code_source_sha256": _sha256(Path(__file__)),
        "acceptance_thresholds": ACCEPTANCE,
        "statistical_limit": (
            "Each measurement's MC draws are independent, but measurements "
            "within one acquisition are correlated. These checks are descriptive "
            "conservative diagnostics, not a family-wise p-value or proof that "
            "the full covariance is diagonal."
        ),
        "monte_carlo_simulation": (
            "Existing LION.pcm_tools.noise_generate.generate_noise and add_noise; "
            "fixed clean split p/q derived from the original LION forward operator"
        ),
        "oracle_weights_unchanged": True,
        "solver_reconstructions_run": False,
    }
    write_json_new(destination / "protocol.json", protocol)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "record_count": len(records),
        "full_checks_pass": sum(
            r["predeclared_checks_passed"] for r in records if r["mode"] == "full"
        ),
        "full_checks_total": sum(r["mode"] == "full" for r in records),
        "control_checks_pass": sum(
            r["predeclared_checks_passed"] for r in records if r["mode"] != "full"
        ),
        "control_checks_total": sum(r["mode"] != "full" for r in records),
        "source_protocol_sha256": _sha256(destination / "protocol.json"),
        "predeclared_checks_all_passed": all(
            record["predeclared_checks_passed"] for record in records
        ),
        "records": records,
    }
    write_json_new(destination / "manifest.json", manifest)
    # Separate CSV for ordinary plotting; verify its digest too.
    columns = [
        "category",
        "sample_id",
        "mode",
        "sample_count",
        "variance_ratio_mean",
        "variance_ratio_median",
        "median_abs_variance_relative_error",
        "p95_abs_variance_relative_error",
        "mean_z_median_abs",
        "mean_z_p95_abs",
        "fraction_abs_mean_z_below_3point5",
        "predeclared_checks_passed",
        "file",
    ]
    path = destination / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    paths = ["protocol.json", "manifest.json", "summary.csv"] + [
        record["file"] for record in records
    ]
    checksums = "".join(f"{_sha256(destination / rel)}  {rel}\n" for rel in paths)
    _write_atomic_new(destination / "SHA256SUMS", lambda f: f.write(checksums.encode()))
    print("EXTERNAL_STEP4_MANIFEST_SHA256=" + _sha256(destination / "manifest.json"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-acquisition-manifest-sha256", required=True)
    parser.add_argument("--expected-weight-manifest-sha256", required=True)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--control-repetitions", type=int, default=768)
    parser.add_argument("--cases-per-category", type=int, default=2)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run_validation(
        args.acquisitions,
        args.weights,
        args.output,
        expected_acquisition_manifest_sha256=args.expected_acquisition_manifest_sha256,
        expected_weight_manifest_sha256=args.expected_weight_manifest_sha256,
        repetitions=args.repetitions,
        control_repetitions=args.control_repetitions,
        cases_per_category=args.cases_per_category,
        enable_controls=not args.no_controls,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
