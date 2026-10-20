"""Train compact grayscale denoisers on synthetic PCM images.

Independent 5678 permutation: positions 1-800 train; 801-900 validation;
901-1000 held out for reconstruction tests. Models are an inexpensive pilot,
not substitutes for full DRUNet/GS-DRUNet training.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.simulation import simulate_pcm_acquisition

ROOT = Path(
    "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
)
OUT = Path(
    "/store/LION/ttv22/pcm/results/pcm_admm_noise_study_20261020/denoiser_training"
)
CATEGORIES = ("CIGS", "Organic", "Perovskite", "Reference", "Silicone")
ORDER = np.random.default_rng(5678).permutation(np.arange(1, 1001))


class ResidualDenoiser(nn.Module):
    def __init__(self, width: int = 32, depth: int = 5):
        super().__init__()
        layers = [nn.Conv2d(2, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend(
                (nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True))
            )
        layers.append(nn.Conv2d(width, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma: float = 0.05):
        s = torch.ones_like(x[:, :1]) * sigma
        return x + self.net(torch.cat((x, s), dim=1))


class PCMImages(Dataset):
    def __init__(self, split: str, count_per_category: int):
        lo, hi = (0, 800) if split == "train" else (800, 900)
        selected = ORDER[lo:hi][:count_per_category]
        self.paths = [
            ROOT / cat / f"image{int(sid):04d}.png"
            for cat in CATEGORIES
            for sid in selected
        ]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arr = (
            np.asarray(Image.open(self.paths[index]).convert("L"), dtype=np.float32)
            / 255
        )
        return torch.from_numpy(arr)[None]


@torch.no_grad()
def corrupt(clean, mode: str, noise_model, rng, sigma: float) -> torch.Tensor:
    if mode == "gaussian":
        return clean + torch.randn_like(clean) * sigma
    j = clean.shape[-1].bit_length() - 1
    indices = np.arange(clean.shape[-1] ** 2)
    out = []
    for x in clean:
        acq = simulate_pcm_acquisition(
            x[None],
            j_order=j,
            sampled_indices=indices,
            noise=noise_model,
            rng=rng,
            device=clean.device,
        )
        out.append(acq.op.pseudo_inv(acq.measurement).squeeze(0))
    return torch.stack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gaussian", "pcm"), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-per-category", type=int, default=24)
    parser.add_argument("--val-per-category", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = args.output / args.mode
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2))
    train = DataLoader(
        PCMImages("train", args.train_per_category),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    val = DataLoader(
        PCMImages("val", args.val_per_category), batch_size=1, num_workers=0
    )
    model = ResidualDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    noise = PRESETS["synthetic_png_dataset_box_plots"].noise
    rng = np.random.default_rng(421)
    best = -float("inf")
    with (out / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("epoch", "loss", "val_psnr", "val_noisy_psnr")
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for index, clean in enumerate(train):
                if args.max_batches and index >= args.max_batches:
                    break
                clean = clean.to(device)
                noisy = corrupt(clean, args.mode, noise, rng, args.sigma)
                denoised = model(noisy, args.sigma)
                loss = torch.mean((denoised - clean) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            scores, baselines = [], []
            with torch.no_grad():
                for clean in val:
                    clean = clean.to(device)
                    noisy = corrupt(clean, args.mode, noise, rng, args.sigma)
                    denoised = model(noisy, args.sigma)
                    mse = float(torch.mean((denoised - clean) ** 2).item())
                    base_mse = float(torch.mean((noisy - clean) ** 2).item())
                    scores.append(-10 * np.log10(max(mse, 1e-15)))
                    baselines.append(-10 * np.log10(max(base_mse, 1e-15)))
            score = float(np.mean(scores))
            row = dict(
                epoch=epoch,
                loss=float(np.mean(losses)),
                val_psnr=score,
                val_noisy_psnr=float(np.mean(baselines)),
            )
            writer.writerow(row)
            file.flush()
            print("EPOCH", args.mode, json.dumps(row), flush=True)
            torch.save(
                dict(
                    epoch=epoch,
                    model=model.state_dict(),
                    architecture=dict(width=32, depth=5),
                    sigma=args.sigma,
                    corruption=args.mode,
                ),
                out / "latest.pt",
            )
            if score > best:
                best = score
                torch.save(
                    dict(
                        epoch=epoch,
                        model=model.state_dict(),
                        architecture=dict(width=32, depth=5),
                        sigma=args.sigma,
                        corruption=args.mode,
                    ),
                    out / "best.pt",
                )
    print("DONE", out, "BEST_PSNR", best, flush=True)


if __name__ == "__main__":
    main()
