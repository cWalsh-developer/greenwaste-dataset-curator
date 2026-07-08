from pathlib import Path

from PIL import Image

from greenwaste_dataset_curator.models import ImageRecord
from greenwaste_dataset_curator.quality import (
    greenwaste_categories_from_classes,
    quality_filter_records,
)


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

    accepted, review, rejected, decisions = quality_filter_records(
        records=[
            make_record(first_path, "one"),
            make_record(duplicate_path, "two"),
        ],
        output_dir=tmp_path / "quality",
        duplicate_phash_threshold=16,
        crop_duplicate_check=True,
        near_duplicate_action="reject",
        copy_images=False,
    )

    assert len(accepted) == 1
    assert review == []
    assert len(rejected) == 1
    assert decisions[1].decision == "reject"
    assert "near_duplicate" in decisions[1].reasons


def test_quality_filter_sends_near_duplicates_to_review_by_default(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jpg"
    duplicate_path = tmp_path / "duplicate.jpg"

    image = Image.new("RGB", (100, 100), "white")
    for x in range(25, 75):
        for y in range(25, 75):
            image.putpixel((x, y), (60, 90, 140))
    image.save(first_path)
    image.save(duplicate_path)

    accepted, review, rejected, decisions = quality_filter_records(
        records=[
            make_record(first_path, "one"),
            make_record(duplicate_path, "two"),
        ],
        output_dir=tmp_path / "quality",
        duplicate_phash_threshold=16,
        copy_images=False,
    )

    assert len(accepted) == 1
    assert len(review) == 1
    assert rejected == []
    assert decisions[1].decision == "review"


def test_quality_filter_rejects_missing_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jpg"

    accepted, review, rejected, decisions = quality_filter_records(
        records=[make_record(missing_path, "missing")],
        output_dir=tmp_path / "quality",
        copy_images=False,
    )

    assert accepted == []
    assert review == []
    assert len(rejected) == 1
    assert decisions[0].reasons == "unreadable_or_missing_file"


def test_quality_filter_rejects_non_photo_keyword(tmp_path: Path) -> None:
    path = tmp_path / "bed_illustration.jpg"
    Image.new("RGB", (100, 100), "white").save(path)
    record = make_record(path, "illustration")

    accepted, review, rejected, decisions = quality_filter_records(
        records=[record],
        output_dir=tmp_path / "quality",
        reject_non_photo=True,
        non_photo_visual_check=False,
        copy_images=False,
    )

    assert accepted == []
    assert review == []
    assert len(rejected) == 1
    assert decisions[0].decision == "reject"
    assert "non_photo_keyword:illustration" in decisions[0].reasons


def test_greenwaste_categories_from_detector_classes() -> None:
    categories = greenwaste_categories_from_classes({"bed", "couch", "person"})

    assert categories == {"beds_mattresses", "sofa"}


def test_quality_filter_reclassifies_single_detected_category(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sofa_folder_but_bed.jpg"
    Image.new("RGB", (100, 100), "white").save(path)
    record = make_record(path, "misfiled")
    record = ImageRecord(
        **{
            **record.__dict__,
            "category": "sofa",
        }
    )

    monkeypatch.setattr(
        "greenwaste_dataset_curator.quality.load_yolo_models",
        lambda model_paths: [object()],
    )
    monkeypatch.setattr(
        "greenwaste_dataset_curator.quality.detect_classes",
        lambda model, image_path, confidence, image_size: {"bed"},
    )

    accepted, review, rejected, decisions = quality_filter_records(
        records=[record],
        output_dir=tmp_path / "quality",
        yolo_model_path=Path("fake.pt"),
        require_target_object=True,
        reclassify_mismatched_category=True,
        copy_images=False,
    )

    assert review == []
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].category == "beds_mattresses"
    assert decisions[0].decision == "reclassify"
    assert decisions[0].original_category == "sofa"
    assert decisions[0].output_category == "beds_mattresses"


def test_quality_filter_combines_multiple_model_detections(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "storage_folder_but_bed_model_finds_bed.jpg"
    Image.new("RGB", (100, 100), "white").save(path)
    record = make_record(path, "multi-model")
    record = ImageRecord(
        **{
            **record.__dict__,
            "category": "storage",
        }
    )
    model_one = object()
    model_two = object()

    monkeypatch.setattr(
        "greenwaste_dataset_curator.quality.load_yolo_models",
        lambda model_paths: [model_one, model_two],
    )

    def fake_detect_classes(model, image_path, confidence, image_size):
        if model is model_one:
            return {"bed"}
        return set()

    monkeypatch.setattr(
        "greenwaste_dataset_curator.quality.detect_classes",
        fake_detect_classes,
    )

    accepted, review, rejected, decisions = quality_filter_records(
        records=[record],
        output_dir=tmp_path / "quality",
        yolo_model_paths=[Path("bed_model.pt"), Path("storage_model.pt")],
        require_target_object=True,
        reclassify_mismatched_category=True,
        copy_images=False,
    )

    assert review == []
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].category == "beds_mattresses"
    assert decisions[0].detected_classes == "bed"
    assert decisions[0].detected_categories == "beds_mattresses"
