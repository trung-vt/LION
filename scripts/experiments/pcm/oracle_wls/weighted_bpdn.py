"""Step 5 oracle WLS-BPDN using the unchanged LION db4 wavelet and SPGL1.

The experiment preserves the float64 frozen y and oracle inverse standard
deviations. The existing Wavelet2D and SPGL1 adapter work in float32, so
conversion occurs at their solver boundary, never in the stored archives.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from LION.operators.CompositeOp import CompositeOp
from LION.operators.Operator import Operator
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.spgl1_bpdn import spgl1_bpdn_torch
from scripts.experiments.pcm.oracle_wls.build_oracle_weights import verify_weights
from scripts.experiments.pcm.oracle_wls.float64_acquisition import (
    _sha256,
    load_master_npz,
)

ACQUISITION_MANIFEST_SHA256 = (
    "b30ff71a305bead9edda34b34200ca509a7654d7825d333d68ff9231f907ebac"
)
WEIGHT_MANIFEST_SHA256 = (
    "50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1"
)
CALIBRATION_SCHEMA = "pcm-step5-95pct-residual-calibration-v1"
WAVELET_NAME = "db4"
WAVELET_MODE = "periodization"
SOLVER_DTYPE = torch.float32


class DiagonallyWeightedOperator(Operator):
    """Linear wrapper B = D A Psi^-1 with exact corresponding adjoint.

    The stored float64 inverse standard deviations are checked, then cast
    only once into the existing float32 solver arithmetic.
    """

    def __init__(self, base: Operator, inverse_std):
        device = getattr(base, "device", None)
        if device is None:
            raise ValueError("Underlying operator must define a device")
        super().__init__(device=device)
        d = torch.as_tensor(inverse_std, dtype=torch.float64).detach().clone()
        if d.ndim != 1 or d.numel() != base.range_shape[-1]:
            raise ValueError("inverse_std length must match measurement count")
        if not torch.isfinite(d).all() or not (d > 0).all():
            raise ValueError("inverse_std must be strictly positive and finite")
        d32 = d.to(device=device, dtype=SOLVER_DTYPE)
        if not torch.isfinite(d32).all() or not (d32 > 0).all():
            raise ValueError("inverse_std cannot be safely represented in float32")
        self.base = base
        self.register_buffer("inverse_std", d32)

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.inverse_std

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return self.base.adjoint(r * self.inverse_std)

    @property
    def domain_shape(self):
        return self.base.domain_shape

    @property
    def range_shape(self):
        return self.base.range_shape


@dataclass(frozen=True)
class Step5Case:
    category: str
    sample_id: int
    acquisition: object
    inverse_std64: torch.Tensor
    variance64: torch.Tensor
    acquisition_sha256: str
    weight_sha256: str


def validate_frozen_archives(acquisitions_root, weights_root):
    """Validate all archived Step 3 source bindings against external hashes."""
    return verify_weights(
        acquisitions_root,
        weights_root,
        expected_acquisition_manifest_sha256=ACQUISITION_MANIFEST_SHA256,
        expected_weight_manifest_sha256=WEIGHT_MANIFEST_SHA256,
    )


def load_step5_case(
    acquisitions_root, weights_root, category, sample_id, *, verify_all=True
) -> Step5Case:
    """Load the same frozen A,y and matching oracle D; do not load ground truth."""
    source, weight_dir = Path(acquisitions_root), Path(weights_root)
    # Even when the caller already checked all 50 cases, always check the
    # independently pinned archive manifests. This is inexpensive and closes
    # the coherent NPZ+manifest+SHA256SUMS mutation route.
    if _sha256(source / "manifest.json") != ACQUISITION_MANIFEST_SHA256:
        raise ValueError("Externally anchored acquisition manifest differs")
    if _sha256(weight_dir / "manifest.json") != WEIGHT_MANIFEST_SHA256:
        raise ValueError("Externally anchored weight manifest differs")
    if verify_all:
        validate_frozen_archives(source, weight_dir)
    acquisitions = json.loads((source / "manifest.json").read_text())["acquisitions"]
    weights = json.loads((weight_dir / "manifest.json").read_text())["files"]
    match_a = [
        r
        for r in acquisitions
        if r["category"] == category and int(r["sample_id"]) == int(sample_id)
    ]
    match_w = [
        r
        for r in weights
        if r["category"] == category and int(r["sample_id"]) == int(sample_id)
    ]
    if len(match_a) != 1 or len(match_w) != 1:
        raise ValueError("Case not uniquely found in Step 2 and 3 validation archives")
    a, w = match_a[0], match_w[0]
    if not a["file"].startswith("validation/") or not w["file"].startswith(
        "validation/"
    ):
        raise ValueError("No held-out test acquisitions allowed in Step 5")
    acquired, info = load_master_npz(source / a["file"], expected_sha256=a["sha256"])
    if acquired.measurement.dtype != torch.float64:
        raise ValueError("Source measurement must remain float64")
    weight_file = weight_dir / w["file"]
    if _sha256(weight_file) != w["sha256"]:
        raise ValueError("Oracle weight checksum mismatch")
    with np.load(weight_file, allow_pickle=False) as f:
        note = json.loads(str(f["metadata_json"]))
        d, v = f["inverse_std"].copy(), f["variance"].copy()
        mask = f["sampled_indices"].copy()
    if (
        note["source_acquisition_sha256"] != a["sha256"]
        or note["source_acquisition_manifest_sha256"] != ACQUISITION_MANIFEST_SHA256
        or note["source_png_sha256"] != info["source_sha256"]
        or note["source_protocol_id"] != info["protocol_id"]
        or note["category"] != category
        or int(note["sample_id"]) != int(sample_id)
        or not np.array_equal(mask, acquired.sampled_indices)
    ):
        raise ValueError("Weights are not tied to the same frozen acquisition A,y")
    if v.dtype != np.float64 or d.dtype != np.float64 or v.shape != mask.shape:
        raise ValueError("Saved oracle variance must remain one float64 vector")
    if not np.isfinite(v).all() or not np.isfinite(d).all() or np.any(v <= 0):
        raise ValueError("Nonfinite or nonpositive oracle variance")
    if not np.array_equal(d, 1 / np.sqrt(v)):
        raise ValueError("Saved whitener differs from oracle variance")
    return Step5Case(
        str(category),
        int(sample_id),
        acquired,
        torch.from_numpy(d),
        torch.from_numpy(v),
        a["sha256"],
        w["sha256"],
    )


def load_calibration_record(root, category, sample_id, *, expected_manifest_sha256):
    """Reject unanchored or mismatched 95% epsilon data."""
    root = Path(root)
    if _sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("External calibration manifest SHA-256 mismatch")
    # Internal checksums also cover the independent calibration summary CSV.
    lines = (root / "SHA256SUMS").read_text().splitlines()
    covered = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        if rel in covered or rel not in ("manifest.json", "summary.csv"):
            raise ValueError("Unexpected calibration checksum sidecar entry")
        covered[rel] = digest
        if _sha256(root / rel) != digest:
            raise ValueError("Calibration SHA256SUMS checksum mismatch")
    if set(covered) != {"manifest.json", "summary.csv"}:
        raise ValueError("Incomplete calibration checksum sidecar")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != CALIBRATION_SCHEMA:
        raise ValueError("Unexpected residual-calibration schema")
    if manifest.get("status") != "complete":
        raise ValueError("Residual calibration is incomplete")
    records = manifest.get("cases", [])
    if not isinstance(records, list) or manifest.get("number_of_cases") != len(records):
        raise ValueError("Residual calibration case count differs from manifest")
    if manifest.get("all_per_case_coverage_in_predeclared_gate") is not True:
        raise ValueError("Residual calibration coverage gate did not pass")
    # The production archives have 50 validation cases. One-case disposable
    # test fixtures use independently generated, different source anchors.
    if (
        manifest.get("weight_manifest_sha256")
        == "50bf26f3dc136adb71513fd3d83e83b24c9bea95ff62229d2fb15ed5248994d1"
        and len(records) != 50
    ):
        raise ValueError(
            "Production residual calibration needs all 50 validation cases"
        )
    if (
        manifest["acquisition_manifest_sha256"] != ACQUISITION_MANIFEST_SHA256
        or manifest["weight_manifest_sha256"] != WEIGHT_MANIFEST_SHA256
    ):
        raise ValueError("Residual bounds belong to different source acquisitions")
    cases = [
        r
        for r in manifest["cases"]
        if r["category"] == category and int(r["sample_id"]) == int(sample_id)
    ]
    if len(cases) != 1:
        raise ValueError("Residual calibration not uniquely available")
    for key in ("epsilon_ols_95", "epsilon_wls_95"):
        if not math.isfinite(cases[0][key]) or cases[0][key] <= 0:
            raise ValueError("Invalid calibrated epsilon")
    return cases[0]


def solve_step5_case(
    case: Step5Case, *, method: str, epsilon: float, iters=120, opt_tol=1e-3, factor=1.0
):
    """Reuse SPGL1 BPDN unchanged, with exactly the same db4 wavelets.

    Returns the reconstruction and *both* OLS and WLS residual diagnostics
    for the same frozen y and A. No optional debiasing is used.
    """
    if method not in ("ols", "wls"):
        raise ValueError("method must be 'ols' or 'wls'")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if factor != 1.0:
        raise ValueError("Step 5 requires factor=1, as in the current PCM preset")
    acquired = case.acquisition
    if acquired.op.domain_shape[0] != acquired.op.domain_shape[1]:
        raise ValueError("PCM images must be square")
    wavelet = Wavelet2D(
        acquired.op.domain_shape,
        wavelet_name=WAVELET_NAME,
        mode=WAVELET_MODE,
        device=torch.device("cpu"),
    )
    base = CompositeOp(wavelet, acquired.op, device=torch.device("cpu"))
    op = (
        DiagonallyWeightedOperator(base, case.inverse_std64)
        if method == "wls"
        else base
    )
    y_master = acquired.measurement.reshape(-1)
    y32 = y_master.to(dtype=SOLVER_DTYPE).clone()
    # Weighted y is formed in float64 before its one-time solver conversion.
    rhs = (
        (y_master * case.inverse_std64).to(dtype=SOLVER_DTYPE)
        if method == "wls"
        else y32
    )
    coefficients, report = spgl1_bpdn_torch(
        op,
        rhs,
        epsilon=float(epsilon),
        factor=1.0,
        iters=iters,
        opt_tol=opt_tol,
        verbosity=0,
    )
    reconstructed = wavelet.inverse(coefficients)
    # Recompute physical and weighted residuals in float64, matching the
    # original frozen y and the float64 calibration definition.
    raw_residual = acquired.op(reconstructed.to(torch.float64)).reshape(-1) - y_master
    ols_norm = float(torch.linalg.vector_norm(raw_residual))
    wls_norm = float(torch.linalg.vector_norm(raw_residual * case.inverse_std64))
    selected = wls_norm if method == "wls" else ols_norm
    info = {
        **report.to_dict(),
        "method": method,
        "wavelet_name": WAVELET_NAME,
        "wavelet_mode": WAVELET_MODE,
        "source_acquisition_sha256": case.acquisition_sha256,
        "source_weight_sha256": case.weight_sha256,
        "master_measurement_dtype": "float64",
        "master_weight_dtype": "float64",
        "solver_dtype": "float32",
        "debias": False,
        "epsilon_used": float(epsilon),
        "ols_residual_norm": ols_norm,
        "wls_residual_norm": wls_norm,
        "selected_residual_norm": selected,
        "solver_residual_consistency_abs_error": abs(selected - report.residual_norm),
        "image_finite": bool(torch.isfinite(reconstructed).all()),
    }
    return reconstructed, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--expected-calibration-manifest-sha256", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--method", choices=["ols", "wls"], required=True)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--opt-tol", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    case = load_step5_case(
        args.acquisitions, args.weights, args.category, args.sample_id
    )
    rec = load_calibration_record(
        args.calibration,
        args.category,
        args.sample_id,
        expected_manifest_sha256=args.expected_calibration_manifest_sha256,
    )
    _, info = solve_step5_case(
        case,
        method=args.method,
        epsilon=rec[f"epsilon_{args.method}_95"],
        iters=args.iters,
        opt_tol=args.opt_tol,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
