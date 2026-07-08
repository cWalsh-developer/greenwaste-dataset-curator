from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote

import imagehash
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from .models import ImageRecord, QualityDecision


DEFAULT_TARGET_CLASSES = {
    "beds_mattresses": {"bed"},
    "chair_seating": {"chair", "bench"},
    "sofa": {"couch", "sofa"},
    "tables_desks": {"dining table", "table"},
}

DETECTOR_CLASS_TO_CATEGORY = {
    "armchair": "chair_seating",
    "bed": "beds_mattresses",
    "bench": "chair_seating",
    "bookcase": "storage",
    "cabinet": "storage",
    "chair": "chair_seating",
    "couch": "sofa",
    "cupboard": "storage",
    "desk": "tables_desks",
    "dining table": "tables_desks",
    "drawer": "storage",
    "dresser": "storage",
    "filing cabinet": "storage",
    "mattress": "beds_mattresses",
    "sofa": "sofa",
    "storage": "storage",
    "table": "tables_desks",
    "tables_desks": "tables_desks",
    "beds_mattresses": "beds_mattresses",
    "chair_seating": "chair_seating",
    "wardrobe": "storage",
}

NON_PHOTO_KEYWORDS = {
    "animation",
    "anime",
    "cartoon",
    "cgi",
    "clipart",
    "clip_art",
    "comic",
    "diagram",
    "drawing",
    "floor_plan",
    "icon",
    "illustration",
    "logo",
    "painting",
    "render",
    "rendered",
    "sketch",
    "svg",
    "vector",
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


def greenwaste_categories_from_classes(detected_classes: set[str]) -> set[str]:
    return {
        DETECTOR_CLASS_TO_CATEGORY[detected_class]
        for detected_class in detected_classes
        if detected_class in DETECTOR_CLASS_TO_CATEGORY
    }


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


def load_yolo_models(model_paths: list[Path]) -> list:
    if not model_paths:
        return []
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install YOLO support with: pip install -e .[yolo]") from exc
    return [YOLO(str(model_path)) for model_path in model_paths]


def normalize_model_paths(
    yolo_model_paths: Path | list[Path] | None,
    yolo_model_path: Path | None,
) -> list[Path]:
    model_paths: list[Path] = []
    if yolo_model_paths is not None:
        if isinstance(yolo_model_paths, Path):
            model_paths.append(yolo_model_paths)
        else:
            model_paths.extend(yolo_model_paths)
    if yolo_model_path is not None:
        model_paths.append(yolo_model_path)
    return model_paths


def non_photo_keyword(record: ImageRecord, keywords: set[str] | None = None) -> str | None:
    keywords = keywords or NON_PHOTO_KEYWORDS
    source_text = " ".join(
        [
            record.query,
            record.source_page,
            record.image_url,
            record.local_path,
        ]
    )
    normalized = unquote(source_text).lower().replace("-", "_").replace(" ", "_")
    for keyword in sorted(keywords):
        if keyword in normalized:
            return keyword
    return None


def looks_like_flat_art(image: Image.Image, strict: bool = False) -> bool:
    sample = image.convert("RGB")
    sample.thumbnail((160, 160))

    colors = sample.getcolors(maxcolors=sample.width * sample.height)
    unique_ratio = (len(colors) if colors is not None else sample.width * sample.height) / float(
        sample.width * sample.height
    )

    grayscale = sample.convert("L")
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_mean = ImageStat.Stat(edges).mean[0] / 255.0
    channel_std = sum(ImageStat.Stat(sample).stddev) / 3.0

    if strict:
        return unique_ratio < 0.25 and edge_mean > 0.035 and channel_std < 90.0

    return unique_ratio < 0.08 and edge_mean > 0.06 and channel_std < 70.0


def quality_filter_records(
    records: list[ImageRecord],
    output_dir: Path,
    yolo_model_path: Path | None = None,
    yolo_model_paths: list[Path] | None = None,
    confidence: float = 0.25,
    image_size: int = 960,
    reject_person: bool = False,
    require_target_object: bool = False,
    duplicate_phash_threshold: int = 10,
    crop_duplicate_check: bool = True,
    crop_bit_error_rate: float = 0.25,
    reject_non_photo: bool = False,
    non_photo_visual_check: bool = True,
    strict_non_photo_check: bool = False,
    reclassify_mismatched_category: bool = False,
    near_duplicate_action: str = "review",
    copy_images: bool = False,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], list[QualityDecision]]:
    if near_duplicate_action not in {"reject", "review"}:
        raise ValueError("near_duplicate_action must be either 'reject' or 'review'")

    model_paths = normalize_model_paths(
        yolo_model_paths=yolo_model_paths,
        yolo_model_path=yolo_model_path,
    )
    yolo_models = load_yolo_models(model_paths)
    accepted: list[ImageRecord] = []
    review: list[ImageRecord] = []
    rejected: list[ImageRecord] = []
    decisions: list[QualityDecision] = []
    accepted_hashes: list[tuple[ImageRecord, imagehash.ImageHash, imagehash.ImageMultiHash | None]] = []

    for record in records:
        image_path = Path(record.local_path)
        reasons: list[str] = []
        detected_classes: set[str] = set()
        detected_categories: set[str] = set()
        duplicate_of = ""
        output_category = record.category

        image = load_image(image_path)
        if image is None:
            reasons.append("unreadable_or_missing_file")
        else:
            if reject_non_photo:
                keyword = non_photo_keyword(record)
                if keyword is not None:
                    reasons.append(f"non_photo_keyword:{keyword}")
                elif non_photo_visual_check and looks_like_flat_art(
                    image,
                    strict=strict_non_photo_check,
                ):
                    reasons.append("non_photo_visual_heuristic")

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

        if not reasons and yolo_models:
            for yolo_model in yolo_models:
                detected_classes.update(
                    detect_classes(
                        model=yolo_model,
                        image_path=image_path,
                        confidence=confidence,
                        image_size=image_size,
                    )
                )
            detected_categories = greenwaste_categories_from_classes(detected_classes)
            if reject_person and "person" in detected_classes:
                reasons.append("contains_person")

            if require_target_object:
                target_classes = DEFAULT_TARGET_CLASSES.get(record.category)
                target_detected = (
                    record.category in detected_categories
                    or bool(target_classes and detected_classes.intersection(target_classes))
                )
                if not target_detected:
                    if reclassify_mismatched_category and len(detected_categories) == 1:
                        output_category = next(iter(detected_categories))
                    elif reclassify_mismatched_category and len(detected_categories) > 1:
                        reasons.append("ambiguous_detected_category")
                    else:
                        reasons.append("target_not_detected")

        if reasons == ["near_duplicate"] and near_duplicate_action == "review":
            decision = "review"
        else:
            decision = "reject" if reasons else "accept"
        if decision == "accept" and output_category != record.category:
            decision = "reclassify"

        output_record = replace(record, category=output_category)
        if copy_images:
            copied_path = copy_record_image(
                record=output_record,
                output_dir=output_dir,
                split_name="review" if decision == "review" else (
                    "rejected" if decision == "reject" else "accepted"
                ),
            )
            output_record = replace(output_record, local_path=copied_path)

        if decision == "review":
            review.append(output_record)
        elif decision == "reject":
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
                original_category=record.category,
                output_category=output_category,
                decision=decision,
                reasons=";".join(reasons),
                detected_classes=";".join(sorted(detected_classes)),
                detected_categories=";".join(sorted(detected_categories)),
                duplicate_of=duplicate_of,
            )
        )

    return accepted, review, rejected, decisions
