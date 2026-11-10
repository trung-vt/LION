from __future__ import annotations
import csv, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np, torch
from LION.config.sapm_config import SAPMModelConfig, AdaptivePDHGConfig
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTGV3PCMNetwork import AdaptiveTGV3PCMNetwork
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_sapm import PhotocurrentMapMRProOp
from LION.utils.metrics import compute_metrics

ROOT = Path("/store/LION/ttv22/pcm/results/pcm")
OUT = ROOT / "perovskite0292_relaxation_three_methods_checkpoints_20261110_v1"
FROZEN = ROOT / "perovskite0292_c60_ternary_20261109_v1/frozen_case.npz"
METHODS = {
    "TV": ("tv", "TV", AdaptiveTVPCMNetwork),
    "TGV-2": ("tgv2", "TGV", AdaptiveTGVPCMNetwork),
    "TGV-3": ("tgv3", "TGV3", AdaptiveTGV3PCMNetwork),
}
VARIANTS = {
    "LION_existing": "pdhg_lion_legacy_local.py",
    "canonical_CP": "pdhg_canonical_cp_local.py",
}
STEPS = (256, 512, 1024)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def import_file(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def metrics(im, gt):
    a = compute_metrics(
        recon=torch.from_numpy(im)[None, None].float(),
        target=torch.from_numpy(gt)[None, None].float(),
        data_range=1.0,
    )
    return {k: float(a[k]) for k in ("psnr", "ssim", "haarpsi", "mse")}


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(FROZEN, allow_pickle=False) as z:
        gt = z["ground_truth"].copy()
        obs = z["measurement"].copy()
        ix = z["sampled_indices"].copy()
    assert obs.shape == (1, 1, 13107) and gt.shape == (256, 256)
    records = []
    images = {"ground_truth": gt, "measurement": obs, "sampled_indices": ix}
    for method, (short, reg, cls) in METHODS.items():
        src = ROOT / f"perovskite0292_{short}_cp_extrapolation_20261110_v1"
        meta = json.loads((src / "summary.json").read_text())
        assert digest(obs) == meta["frozen_measurement_sha256"]
        assert digest(ix) == meta["frozen_indices_sha256"]
        with (src / "metrics_by_iterations.csv").open() as fh:
            archive = {
                (r["variant"], int(r["iterations"])): r for r in csv.DictReader(fh)
            }
        if method == "TV":
            d = json.loads(
                (
                    ROOT / "perovskite0292_scalar_tv_oracle_20261109_v1/summary.json"
                ).read_text()
            )
            weights = float(d["best"]["lambda"])
            assert abs(float(meta["weights"]) - weights) < 1e-12
        elif method == "TGV-2":
            d = json.loads(
                (
                    ROOT / "perovskite0292_scalar_tgv_oracle_20261109_v1/selected.json"
                ).read_text()
            )
            weights = (float(d["selected"]["alpha1"]), float(d["selected"]["alpha0"]))
            assert all(abs(a - b) < 1e-12 for a, b in zip(weights, meta["weights"]))
        else:
            d = json.loads(
                (
                    ROOT / "perovskite0292_scalar_tgv3_oracle_20261110_v1/summary.json"
                ).read_text()
            )
            weights = tuple(d["weights"])
            assert tuple(meta["weights"]) == weights
        cfg = SAPMModelConfig(
            name="local-checkpoint-" + short,
            input_image="zeros",
            initial_image="zeros",
            n_iters=1024,
            backend="new",
            cnn=None,
            pdhg=AdaptivePDHGConfig(
                regularisation=reg,
                primal_stepsize=0.18,
                dual_stepsize=0.18,
                initial_scale_factor_raw=0.1,
                scale_factor_softplus_beta=1.0,
                lambda_softplus_beta=1.0,
                fd_pad_mode=["zeros", "zeros"],
                fd_dim=[-2, -1],
                theta=1.0,
            ),
        )
        model = cls(cfg, None).to(dev).eval()
        A = PhotocurrentMapMRProOp(
            PhotocurrentMapOp(j_order=8, s_indices=ix, normalized=True, device=dev)
        )
        K = model.make_stacked_operator(A)
        y = torch.from_numpy(obs).to(dev)
        z0 = torch.zeros((1, 1, 256, 256), dtype=y.dtype, device=dev)
        f = model.make_f(y, lambda_reg=weights)
        for variant, filename in VARIANTS.items():
            pdhg = import_file(f"{short}_{variant}", src / filename).pdhg
            for n in STEPS:
                torch.cuda.synchronize()
                tic = time.perf_counter()
                with torch.inference_mode():
                    z = pdhg(
                        f=f,
                        g=model.g,
                        operator=K,
                        initial_values=model.make_initial_values(z0),
                        max_iterations=n,
                        primal_stepsize=0.18,
                        dual_stepsize=0.18,
                        relaxation=1.0,
                        use_checkpoint=False,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tic
                im = z[0].squeeze().detach().cpu().numpy().astype(np.float32)
                assert im.shape == gt.shape and np.isfinite(im).all()
                observed = metrics(im, gt)
                old = archive[variant, n]
                for k, v in observed.items():
                    tolerance = 3e-5 if k != "psnr" else 2e-5
                    assert abs(v - float(old[k])) < tolerance, (
                        method,
                        variant,
                        n,
                        k,
                        v,
                        old[k],
                    )
                with torch.inference_mode():
                    objective = float(f(*K(*z))[0])
                    residual = (
                        (A(z[0])[0] - y)
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                assert residual.shape == (len(ix),)
                assert abs(objective - float(old["objective"])) < 0.0015, (
                    method,
                    variant,
                    n,
                    objective,
                    old["objective"],
                )
                measurement_rmse = float(
                    np.sqrt(np.mean(residual.astype("float64") ** 2))
                )
                row = {
                    "method": method,
                    "variant": variant,
                    "iterations": n,
                    **observed,
                    "objective": objective,
                    "data_fidelity": 0.5
                    * float(np.square(residual.astype("float64")).sum()),
                    "measurement_residual_rmse": measurement_rmse,
                    "seconds": elapsed,
                    "archived_seconds": float(old["seconds"]),
                    "ms_per_iteration": elapsed * 1000 / n,
                    "weights": str(weights),
                }
                records.append(row)
                key = f"{short}_{variant}_n{n}"
                images[key] = im
                images[key + "_signed_error"] = im - gt
                images[key + "_measurement_residual"] = residual
                print(
                    "VERIFIED",
                    method,
                    variant,
                    n,
                    f'PSNR={observed["psnr"]:.6f}',
                    f'SSIM={observed["ssim"]:.6f}',
                    f'HaarPSI={observed["haarpsi"]:.6f}',
                    f"objective={objective:.5f}",
                    f"time={elapsed:.3f}s",
                    flush=True,
                )
    assert len(records) == 18
    with (OUT / "checkpoint_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    np.savez_compressed(OUT / "checkpoint_reconstructions_and_residuals.npz", **images)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "case": "Perovskite_0292",
                "GPU": torch.cuda.get_device_name(0),
                "iterations": STEPS,
                "sampling": "Frozen 20% NPL default Hadamard acquisition",
                "measurement_sha256": digest(obs),
                "indices_sha256": digest(ix),
                "step_sizes": {"tau": 0.18, "sigma": 0.18},
                "same_weights_and_initialization": True,
                "LION_source_changes": False,
                "spatial_error_definition": "unclipped reconstruction - ground truth",
                "measurement_residual_definition": "A reconstruction - y, normalized PCM operator",
                "all_18_match_archived_metrics_and_objective": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("COMPLETE", OUT, len(records), flush=True)


if __name__ == "__main__":
    main()
