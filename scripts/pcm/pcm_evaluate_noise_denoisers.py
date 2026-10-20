"""Evaluate Gaussian- and PCM-trained denoisers on the same held-out corruption."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.pcm.pcm_train_noise_denoiser import (
    OUT,
    ORDER,
    PCMImages,
    ResidualDenoiser,
    corrupt,
)
from LION.experiments.pcm_presets.presets import PRESETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    torch.manual_seed(2233)
    torch.set_num_threads(2)
    rng = np.random.default_rng(2233)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    for variant in ("gaussian", "pcm"):
        checkpoint = torch.load(
            args.output / variant / "best.pt", map_location=device, weights_only=False
        )
        model = ResidualDenoiser(**checkpoint["architecture"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[variant] = model
    # Split indices are taken from positions 901-1000, unseen during training and tuning.
    base = Path(
        "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
    )
    noise = PRESETS["synthetic_png_dataset_box_plots"].noise
    from PIL import Image

    rows = []
    for category in ("CIGS", "Organic", "Perovskite", "Reference", "Silicone"):
        for sample in ORDER[900 : 900 + args.per_category]:
            name = f"{category}/image{int(sample):04d}.png"
            image = torch.as_tensor(
                np.array(Image.open(base / name).convert("L"), dtype=np.float32) / 255,
                device=device,
            )[None, None]
            for condition in ("gaussian", "pcm"):
                noisy = corrupt(image, condition, noise, rng, 0.05)
                candidates = {"noisy": noisy}
                with torch.no_grad():
                    for label, model in models.items():
                        candidates[label] = model(noisy, 0.05)
                for label, estimate in candidates.items():
                    mse = float(((estimate - image) ** 2).mean().item())
                    rows.append(
                        dict(
                            category=category,
                            sample_id=int(sample),
                            corrupt=condition,
                            denoiser=label,
                            psnr=float(-10 * np.log10(max(1e-15, mse))),
                            mse=mse,
                        )
                    )
    out = args.output / "heldout_denoising.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for condition in ("gaussian", "pcm"):
        for model in ("noisy", "gaussian", "pcm"):
            nums = [
                r["psnr"]
                for r in rows
                if r["corrupt"] == condition and r["denoiser"] == model
            ]
            print(
                condition,
                model,
                "mean_psnr",
                round(float(np.mean(nums)), 4),
                "min_psnr",
                round(float(min(nums)), 4),
                flush=True,
            )
    print(out)


if __name__ == "__main__":
    main()
