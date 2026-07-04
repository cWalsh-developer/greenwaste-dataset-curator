from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from .models import ImageRecord


def assign_grouped_splits(
    records: list[ImageRecord],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train, val, and test fractions must sum to 1.0")

    groups_by_category: dict[str, list[str]] = {}
    for record in records:
        groups_by_category.setdefault(record.category, [])
        if record.group_id not in groups_by_category[record.category]:
            groups_by_category[record.category].append(record.group_id)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for category, group_ids in sorted(groups_by_category.items()):
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)
        n_groups = len(shuffled)
        n_train = round(n_groups * train_fraction)
        n_val = round(n_groups * val_fraction)

        for index, group_id in enumerate(shuffled):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments[group_id] = split

    return assignments


def write_split_dataset(
    records: list[ImageRecord],
    assignments: dict[str, str],
    output_dir: Path,
    copy_images: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = output_dir / "split_manifest.csv"

    fieldnames = [
        "split",
        "category",
        "group_id",
        "source",
        "source_id",
        "source_page",
        "license_name",
        "artist",
        "original_local_path",
        "split_local_path",
        "sha256",
        "perceptual_hash",
    ]
    with split_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            split = assignments[record.group_id]
            original_path = Path(record.local_path)
            split_path = output_dir / split / record.category / original_path.name
            if copy_images:
                split_path.parent.mkdir(parents=True, exist_ok=True)
                if original_path.exists():
                    shutil.copy2(original_path, split_path)

            writer.writerow(
                {
                    "split": split,
                    "category": record.category,
                    "group_id": record.group_id,
                    "source": record.source,
                    "source_id": record.source_id,
                    "source_page": record.source_page,
                    "license_name": record.license_name,
                    "artist": record.artist,
                    "original_local_path": record.local_path,
                    "split_local_path": split_path,
                    "sha256": record.sha256,
                    "perceptual_hash": record.perceptual_hash,
                }
            )
    return split_manifest
