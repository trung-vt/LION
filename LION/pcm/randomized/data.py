"""Source-image manifests and mixed natural/PCM crops for randomized experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_manifest(
    destination: Path,
    *,
    natural_train: list[Path],
    natural_validation: list[Path],
    natural_test: list[Path],
    synthetic_root: Path,
    seed: int = 20261014,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite data manifest: {destination}")
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[dict] = []
    synthetic_ids = sorted({p.stem for p in synthetic_root.rglob("*.png")})
    permutation = np.random.default_rng(seed).permutation(synthetic_ids)
    id_split = {
        str(key): (
            "train"
            if i < int(0.8 * len(permutation))
            else "validation"
            if i < int(0.9 * len(permutation))
            else "test"
        )
        for i, key in enumerate(permutation)
    }

    def add(path: Path, split: str, domain: str, group: str) -> None:
        digest = sha256(path)
        if digest in seen:
            duplicates.append({"path": str(path), "duplicate_of": seen[digest][1]})
            return
        seen[digest] = (split, str(path))
        with Image.open(path) as image:
            width, height = image.size
        splits[split].append(
            {
                "path": str(path.resolve()),
                "domain": domain,
                "group": group,
                "sha256": digest,
                "width": width,
                "height": height,
            }
        )

    # Give explicit test/validation roots precedence over duplicate training files.
    for split, roots in (
        ("test", natural_test),
        ("validation", natural_validation),
        ("train", natural_train),
    ):
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(root)
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    add(path, split, "natural", "natural:" + path.name)
    if not synthetic_root.is_dir():
        raise FileNotFoundError(synthetic_root)
    for path in sorted(synthetic_root.rglob("*.png")):
        add(path, id_split[path.stem], "pcm_synthetic", "pcm:" + path.stem)
    for split, records in splits.items():
        for domain in ("natural", "pcm_synthetic"):
            if not any(r["domain"] == domain for r in records):
                raise ValueError(f"No {domain} images in split {split}")
    groups = {s: {r["group"] for r in rows} for s, rows in splits.items()}
    # Natural names can collide across source datasets; content hashes are authoritative there.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {g for g in groups[a] & groups[b] if g.startswith("pcm:")}
        if overlap:
            raise ValueError(f"Synthetic group leakage: {overlap}")
    manifest = {
        "seed": seed,
        "splits": splits,
        "duplicates_removed": duplicates,
        "counts": {
            s: {
                d: sum(r["domain"] == d for r in rows)
                for d in ("natural", "pcm_synthetic")
            }
            for s, rows in splits.items()
        },
        "protocol": {
            "natural": "DIV2K training / DIV2K validation / CBSD68 test, source-image separation",
            "synthetic": "80/10/10 split grouped by image identifier across all material categories",
            "leakage_limit": "Underlying synthetic-template ancestry is not available; template-disjoint generalization is not claimed.",
            "deduplication": "SHA-256 of complete source files; no low-resolution DIV2K copies included",
        },
    }
    write_json(destination, manifest)
    return manifest


class MixedImageCorpus:
    """Sample crops with equal expected natural and synthetic domain frequency."""

    def __init__(self, manifest: Path, side: int = 64):
        if side < 16 or side & (side - 1):
            raise ValueError("side must be a power of two, at least 16")
        self.manifest_path = manifest
        self.manifest = json.loads(manifest.read_text())
        self.side = side
        self.groups = {
            (split, domain): [r for r in records if r["domain"] == domain]
            for split, records in self.manifest["splits"].items()
            for domain in ("natural", "pcm_synthetic")
        }

    def load(
        self, record: dict, rng: np.random.Generator, *, center: bool = False
    ) -> torch.Tensor:
        with Image.open(record["path"]) as original:
            image = ImageOps.exif_transpose(original).convert("L")
            if min(image.size) < self.side:
                scale = self.side / min(image.size)
                image = image.resize(
                    (
                        int(np.ceil(image.width * scale)),
                        int(np.ceil(image.height * scale)),
                    ),
                    Image.Resampling.BICUBIC,
                )
            x = (
                (image.width - self.side) // 2
                if center
                else int(rng.integers(image.width - self.side + 1))
            )
            y = (
                (image.height - self.side) // 2
                if center
                else int(rng.integers(image.height - self.side + 1))
            )
            array = (
                np.asarray(
                    image.crop((x, y, x + self.side, y + self.side)), dtype=np.float32
                )
                / 255.0
            )
        if not center:
            array = np.rot90(array, int(rng.integers(4)))
            if rng.random() < 0.5:
                array = array[:, ::-1]
        return torch.from_numpy(array.copy())[None, None]

    def draw(
        self, rng: np.random.Generator, *, split: str = "train"
    ) -> tuple[torch.Tensor, dict]:
        domain = "natural" if rng.random() < 0.5 else "pcm_synthetic"
        group = self.groups[split, domain]
        record = group[int(rng.integers(len(group)))]
        return self.load(record, rng), record

    def fixed_cases(
        self, split: str, per_domain: int = 4
    ) -> list[tuple[torch.Tensor, dict]]:
        if split not in ("validation", "test"):
            raise ValueError("Fixed evaluation cases must be validation or test")
        result = []
        for domain in ("natural", "pcm_synthetic"):
            # Content-hash ordering avoids selecting consecutive categories or IDs.
            records = sorted(self.groups[split, domain], key=lambda row: row["sha256"])
            for i, row in enumerate(records[:per_domain]):
                result.append(
                    (self.load(row, np.random.default_rng(i), center=True), row)
                )
        return result
