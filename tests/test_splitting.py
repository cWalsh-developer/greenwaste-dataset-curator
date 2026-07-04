from greenwaste_dataset_curator.models import ImageRecord
from greenwaste_dataset_curator.splitting import assign_grouped_splits


def make_record(category: str, group_id: str) -> ImageRecord:
    return ImageRecord(
        category=category,
        query="query",
        source="test",
        source_id=group_id,
        source_page=f"https://example.com/{group_id}",
        image_url=f"https://example.com/{group_id}.jpg",
        local_path=f"dataset/{category}/{group_id}.jpg",
        width=640,
        height=480,
        sha256=group_id,
        perceptual_hash=group_id,
        license_name="test",
        artist="test",
        group_id=group_id,
    )


def test_assign_grouped_splits_keeps_same_group_together() -> None:
    records = [
        make_record("chair_seating", "listing-1"),
        make_record("chair_seating", "listing-1"),
        make_record("chair_seating", "listing-2"),
        make_record("chair_seating", "listing-3"),
        make_record("storage", "listing-4"),
        make_record("storage", "listing-5"),
        make_record("storage", "listing-6"),
    ]

    assignments = assign_grouped_splits(
        records=records,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert assignments["listing-1"] in {"train", "val", "test"}
    assert set(assignments) == {
        "listing-1",
        "listing-2",
        "listing-3",
        "listing-4",
        "listing-5",
        "listing-6",
    }


def test_assign_grouped_splits_rejects_bad_fractions() -> None:
    records = [make_record("storage", "listing-1")]

    try:
        assign_grouped_splits(records, train_fraction=0.8, val_fraction=0.2, test_fraction=0.2, seed=1)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
