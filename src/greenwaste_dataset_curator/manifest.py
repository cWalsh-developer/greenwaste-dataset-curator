from __future__ import annotations

import csv
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, TypeVar

from .models import DetectorProposal, ImageRecord


T = TypeVar("T", ImageRecord, DetectorProposal)


def write_dataclass_csv(path: Path, rows: Iterable[T], row_type: type[T]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_image_manifest(path: Path) -> list[ImageRecord]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                ImageRecord(
                    category=row["category"],
                    query=row["query"],
                    source=row["source"],
                    source_id=row["source_id"],
                    source_page=row["source_page"],
                    image_url=row["image_url"],
                    local_path=row["local_path"],
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                    sha256=row["sha256"],
                    perceptual_hash=row["perceptual_hash"],
                    license_name=row["license_name"],
                    artist=row["artist"],
                    group_id=row.get("group_id") or f"{row['source']}:{row['source_id']}",
                )
            )
        return rows
