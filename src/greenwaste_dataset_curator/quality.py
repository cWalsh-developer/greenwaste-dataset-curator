from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from .models import ImageRecord, QualityDecision


DEFAULT_TARGET_CLASSES = {
    "beds_mattresses": {"bed"},
    "chair_seating": {"chair", "bench"},
    "sofa": {"couch", "sofa"},
    "tables_desks": {"dining table", "table"},
}


def load_image(path: Path) -> Image.Image | None:
    try:
        image = Image.open(path)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def crop_hash_matches(
    first: imagehash.ImageMultiHash,
    second: imagehash.ImageMultiHash,
    bit_error_rate: float,
) -> bool:
    try:
        return first.matches(second, bit_error_rate=bit_error_rate)
    except (IndexError, ZeroDivisionError, ValueError):
        return False


def copy_record_image(record: ImageRecord, output_dir: Path, split_name: str) -> str:
    source_path = Path(record.local_path)
    target_path = output_dir / split_name / record.category / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        shutil.copy2(source_path, target_path)
    return str(target_path)


def detect_classes(
    model,
    image_path: Path,
    confidence: float,
    image_size: int,
) -> set[str]:
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]
    classes: set[str] = set()
    for box in result.boxes:
        class_id = int(box.cls.item())
        classes.add(str(result.names[class_id]).lower())
    return classes


def load_yolo_model(model_path: Path | None):
    if model_path is None:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install YOLO support with: pip install -e .[yolo]") from exc
    return YOLO(str(model_path))


def quality_filter_records(
    records: list[ImageRecord],
    output_dir: Path,
    yolo_model_path: Path | None = None,
    confidence: float = 0.25,
    image_size: int = 960,
    reject_person: bool = False,
    require_target_object: bool = False,
    duplicate_phash_threshold: int = 10,
    crop_duplicate_check: bool = True,
    crop_bit_error_rate: float = 0.25,
    copy_images: bool = False,
) -> tuple[list[ImageRecord], list[ImageRecord], list[QualityDecision]]:
    yolo_model = load_yolo_model(yolo_model_path)
    accepted: list[ImageRecord] = []
    rejected: list[ImageRecord] = []
    decisions: list[QualityDecision] = []
    accepted_hashes: list[tuple[ImageRecord, imagehash.ImageHash, imagehash.ImageMultiHash | None]] = []

    for record in records:
        image_path = Path(record.local_path)
        reasons: list[str] = []
        detected_classes: set[str] = set()
        duplicate_of = ""

        image = load_image(image_path)
        if image is None:
            reasons.append("unreadable_or_missing_file")
        else:
            phash_value = imagehash.phash(image)
            crop_hash = (
                imagehash.crop_resistant_hash(image)
                if crop_duplicate_check
                else None
            )

            for accepted_record, accepted_phash, accepted_crop_hash in accepted_hashes:
                phash_distance = phash_value - accepted_phash
                crop_match = (
                    crop_hash is not None
                    and accepted_crop_hash is not None
                    and crop_hash_matches(
                        crop_hash,
                        accepted_crop_hash,
                        bit_error_rate=crop_bit_error_rate,
                    )
                )
                if phash_distance <= duplicate_phash_threshold or crop_match:
                    reasons.append("near_duplicate")
                    duplicate_of = accepted_record.local_path
                    break

        if not reasons and yolo_model is not None:
            detected_classes = detect_classes(
                model=yolo_model,
                image_path=image_path,
                confidence=confidence,
                image_size=image_size,
            )
            if reject_person and "person" in detected_classes:
                reasons.append("contains_person")

            if require_target_object:
                target_classes = DEFAULT_TARGET_CLASSES.get(record.category)
                if target_classes and not detected_classes.intersection(target_classes):
                    reasons.append("target_not_detected")

        decision = "reject" if reasons else "accept"
        output_record = record
        if copy_images:
            copied_path = copy_record_image(
                record=record,
                output_dir=output_dir,
                split_name="rejected" if reasons else "accepted",
            )
            output_record = replace(record, local_path=copied_path)

        if reasons:
            rejected.append(output_record)
        else:
            accepted.append(output_record)
            if image is not None:
                accepted_hashes.append(
                    (
                        record,
                        imagehash.phash(image),
                        imagehash.crop_resistant_hash(image) if crop_duplicate_check else None,
                    )
                )

        decisions.append(
            QualityDecision(
                local_path=record.local_path,
                category=record.category,
                decision=decision,
                reasons=";".join(reasons),
                detected_classes=";".join(sorted(detected_classes)),
                duplicate_of=duplicate_of,
            )
        )

    return accepted, rejected, decisions
