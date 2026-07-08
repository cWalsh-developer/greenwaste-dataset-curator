from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commons import CommonsCollector
from .manifest import read_image_manifest, write_dataclass_csv
from .models import DetectorProposal, ImageRecord, QualityDecision
from .quality import quality_filter_records
from .splitting import assign_grouped_splits, write_split_dataset
from .yolo import generate_yolo_proposals


def load_query_config(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(category): [str(query) for query in queries] for category, queries in payload.items()}


def collect_commons(args: argparse.Namespace) -> None:
    collector = CommonsCollector(
        output_dir=args.output_dir,
        min_width=args.min_width,
        min_height=args.min_height,
        delay_seconds=args.delay_seconds,
        phash_threshold=args.phash_threshold,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )
    query_map = load_query_config(args.query_config)
    records = collector.collect(query_map=query_map, images_per_query=args.images_per_query)
    manifest_path = args.output_dir / "manifest.csv"
    write_dataclass_csv(manifest_path, records, ImageRecord)
    print(f"Collected {len(records)} image(s)")
    print(f"Manifest: {manifest_path}")


def split_dataset(args: argparse.Namespace) -> None:
    records = read_image_manifest(args.manifest)
    assignments = assign_grouped_splits(
        records=records,
        train_fraction=args.train,
        val_fraction=args.val,
        test_fraction=args.test,
        seed=args.seed,
    )
    split_manifest = write_split_dataset(
        records=records,
        assignments=assignments,
        output_dir=args.output_dir,
        copy_images=not args.no_copy,
    )
    print(f"Split manifest: {split_manifest}")


def yolo_proposals(args: argparse.Namespace) -> None:
    records = read_image_manifest(args.manifest)
    proposals = generate_yolo_proposals(
        records=records,
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
    )
    write_dataclass_csv(args.output_csv, proposals, DetectorProposal)
    print(f"Wrote {len(proposals)} proposal(s) to {args.output_csv}")


def quality_filter(args: argparse.Namespace) -> None:
    records = read_image_manifest(args.manifest)
    accepted, rejected, decisions = quality_filter_records(
        records=records,
        output_dir=args.output_dir,
        yolo_model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        reject_person=args.reject_person,
        require_target_object=args.require_target_object,
        duplicate_phash_threshold=args.duplicate_phash_threshold,
        crop_duplicate_check=not args.no_crop_duplicate_check,
        crop_bit_error_rate=args.crop_bit_error_rate,
        reject_non_photo=args.reject_non_photo,
        non_photo_visual_check=not args.no_non_photo_visual_check,
        reclassify_mismatched_category=args.reclassify_mismatched_category,
        copy_images=not args.no_copy,
    )
    write_dataclass_csv(args.output_dir / "accepted_manifest.csv", accepted, ImageRecord)
    write_dataclass_csv(args.output_dir / "rejected_manifest.csv", rejected, ImageRecord)
    write_dataclass_csv(args.output_dir / "quality_review.csv", decisions, QualityDecision)
    print(f"Accepted image(s): {len(accepted)}")
    print(f"Rejected image(s): {len(rejected)}")
    print(f"Quality review: {args.output_dir / 'quality_review.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="greenwaste-curator",
        description="Collect and curate real-world furniture images for GreenWaste datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-commons", help="Collect images from Wikimedia Commons")
    collect.add_argument("--query-config", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--images-per-query", type=int, default=25)
    collect.add_argument("--min-width", type=int, default=640)
    collect.add_argument("--min-height", type=int, default=480)
    collect.add_argument("--delay-seconds", type=float, default=2.0)
    collect.add_argument("--phash-threshold", type=int, default=4)
    collect.add_argument("--max-retries", type=int, default=5)
    collect.add_argument("--backoff-seconds", type=float, default=5.0)
    collect.set_defaults(func=collect_commons)

    split = subparsers.add_parser("split", help="Create grouped train/val/test split")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--no-copy", action="store_true")
    split.set_defaults(func=split_dataset)

    proposals = subparsers.add_parser("yolo-proposals", help="Generate YOLO proposals for review")
    proposals.add_argument("--manifest", type=Path, required=True)
    proposals.add_argument("--model", type=Path, required=True)
    proposals.add_argument("--output-csv", type=Path, required=True)
    proposals.add_argument("--confidence", type=float, default=0.25)
    proposals.add_argument("--image-size", type=int, default=960)
    proposals.set_defaults(func=yolo_proposals)

    quality = subparsers.add_parser(
        "quality-filter",
        help="Filter collected images for duplicates, people, and wrong target objects",
    )
    quality.add_argument("--manifest", type=Path, required=True)
    quality.add_argument("--output-dir", type=Path, required=True)
    quality.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional YOLO model for semantic checks, e.g. yolo11n.pt",
    )
    quality.add_argument("--confidence", type=float, default=0.25)
    quality.add_argument("--image-size", type=int, default=960)
    quality.add_argument("--reject-person", action="store_true")
    quality.add_argument("--reject-non-photo", action="store_true")
    quality.add_argument(
        "--no-non-photo-visual-check",
        action="store_true",
        help="Only use source/filename keywords for non-photo rejection.",
    )
    quality.add_argument("--require-target-object", action="store_true")
    quality.add_argument(
        "--reclassify-mismatched-category",
        action="store_true",
        help=(
            "If the expected category is not detected but exactly one other "
            "GreenWaste category is detected, move the image into that category."
        ),
    )
    quality.add_argument("--duplicate-phash-threshold", type=int, default=10)
    quality.add_argument("--no-crop-duplicate-check", action="store_true")
    quality.add_argument("--crop-bit-error-rate", type=float, default=0.25)
    quality.add_argument("--no-copy", action="store_true")
    quality.set_defaults(func=quality_filter)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
