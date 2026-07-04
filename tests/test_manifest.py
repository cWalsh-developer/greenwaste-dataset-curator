from pathlib import Path

from greenwaste_dataset_curator.manifest import read_image_manifest, write_dataclass_csv
from greenwaste_dataset_curator.models import ImageRecord


def test_write_and_read_image_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    rows = [
        ImageRecord(
            category="storage",
            query="wardrobe",
            source="wikimedia_commons",
            source_id="123",
            source_page="https://commons.wikimedia.org/?curid=123",
            image_url="https://example.com/file.jpg",
            local_path="dataset/images/storage/storage_abc.jpg",
            width=800,
            height=600,
            sha256="abc",
            perceptual_hash="ffff",
            license_name="CC BY-SA",
            artist="Unknown",
            group_id="wikimedia_commons:123",
        )
    ]

    write_dataclass_csv(path, rows, ImageRecord)
    loaded = read_image_manifest(path)

    assert loaded == rows
