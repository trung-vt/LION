"""Index existing natural and synthetic PCM image datasets for randomized training."""

from __future__ import annotations

import argparse
from pathlib import Path

from LION.pcm.randomized.data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-train", nargs="+", type=Path, required=True)
    parser.add_argument("--natural-validation", nargs="+", type=Path, required=True)
    parser.add_argument("--natural-test", nargs="+", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    manifest = build_manifest(
        args.output,
        natural_train=args.natural_train,
        natural_validation=args.natural_validation,
        natural_test=args.natural_test,
        synthetic_root=args.synthetic_root,
        seed=args.seed,
    )
    print(manifest["counts"])
    print(f"Duplicate files excluded: {len(manifest['duplicates_removed'])}")


if __name__ == "__main__":
    main()
