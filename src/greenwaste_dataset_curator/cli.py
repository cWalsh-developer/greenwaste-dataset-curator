from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commons import CommonsCollector
from .manifest import read_image_manifest, write_dataclass_csv
from .models import DetectorProposal, ImageRecord
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
    collect.add_argument("--delay-seconds", type=float, default=0.5)
    collect.add_argument("--phash-threshold", type=int, default=4)
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
