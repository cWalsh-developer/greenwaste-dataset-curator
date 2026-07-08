from pathlib import Path

from PIL import Image

from greenwaste_dataset_curator.models import ImageRecord
from greenwaste_dataset_curator.quality import quality_filter_records


def make_record(path: Path, source_id: str) -> ImageRecord:
    return ImageRecord(
        category="beds_mattresses",
        query="bed bedroom",
        source="test",
        source_id=source_id,
        source_page=f"https://example.com/{source_id}",
        image_url=f"https://example.com/{source_id}.jpg",
        local_path=str(path),
        width=100,
        height=100,
        sha256=source_id,
        perceptual_hash=source_id,
        license_name="test",
        artist="test",
        group_id=source_id,
    )


def test_quality_filter_rejects_near_duplicate_images(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jpg"
    duplicate_path = tmp_path / "duplicate.jpg"

    image = Image.new("RGB", (100, 100), "white")
    for x in range(25, 75):
        for y in range(25, 75):
            image.putpixel((x, y), (60, 90, 140))
    image.save(first_path)

    zoomed = image.crop((10, 10, 90, 90)).resize((100, 100))
    zoomed.save(duplicate_path)

    accepted, rejected, decisions = quality_filter_records(
        records=[
            make_record(first_path, "one"),
            make_record(duplicate_path, "two"),
        ],
        output_dir=tmp_path / "quality",
        duplicate_phash_threshold=16,
        crop_duplicate_check=True,
        copy_images=False,
    )

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert decisions[1].decision == "reject"
    assert "near_duplicate" in decisions[1].reasons


def test_quality_filter_rejects_missing_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jpg"

    accepted, rejected, decisions = quality_filter_records(
        records=[make_record(missing_path, "missing")],
        output_dir=tmp_path / "quality",
        copy_images=False,
    )

    assert accepted == []
    assert len(rejected) == 1
    assert decisions[0].reasons == "unreadable_or_missing_file"
