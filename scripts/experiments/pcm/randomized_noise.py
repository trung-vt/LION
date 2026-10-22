"""Run bounded randomized-noise PCM training experiments.

Run from the repository root with ``PYTHONPATH=. python`` or ``python -m``.
No network downloads, Git commits or automatic long-running jobs are performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from LION.pcm.randomized.models import DENOISERS, REGULARIZERS
from LION.pcm.randomized.runner import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=DENOISERS + REGULARIZERS, required=True
    )
    parser.add_argument(
        "--noise-modes",
        nargs="+",
        choices=("gaussian", "pcm_like", "mixture"),
        default=["mixture"],
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--unroll-iterations", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-every", type=int, default=32)
    parser.add_argument("--validation-per-domain", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Data manifest does not exist: {args.manifest}")
    for model in args.models:
        modes = args.noise_modes if model in DENOISERS else ["physical_pcm"]
        for mode in modes:
            config = TrainingConfig(
                model=model,
                steps=args.steps,
                side=args.side,
                batch_size=args.batch_size,
                unroll_iterations=args.unroll_iterations,
                learning_rate=args.learning_rate,
                validation_every=args.validation_every,
                validation_per_domain=args.validation_per_domain,
                seed=args.seed,
                noise_mode=mode,
                device=args.device,
            )
            output = args.output_root / f"{model}_{mode}"
            result = train(config, args.manifest, output)
            print(
                f"COMPLETED {output} updates={result['completed_updates']} best_step={result['best_step']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
