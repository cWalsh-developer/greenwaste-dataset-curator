# GreenWaste Dataset Curator

Separate dataset acquisition and curation project for the GreenWaste furniture detection work.

The purpose of this repository is to collect legally traceable, real-world furniture images and prepare them for manual annotation or YOLO training experiments without relying on sources whose terms prohibit ML training usage.

The initial supported source is Wikimedia Commons because its API provides programmatic access and licence metadata. The project records source page, image URL, licence, artist/attribution field, image dimensions, SHA-256 hashes, perceptual hashes, and grouped split IDs.

## Why This Exists

The original GreenWaste detector was trained mainly on clean product-style images. That creates a domain gap:

- clean backgrounds vs natural clutter;
- centred product images vs unusual camera angles;
- good lighting vs RealSense colour shifts and shadows;
- isolated furniture vs partial occlusion and multiple objects.

This project collects and curates more diverse real-world images while maintaining auditability, deduplication, and split discipline.

## Install

From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Create the GitHub Remote

This folder is already a local git repository. To publish it, create an empty GitHub repository named `greenwaste-dataset-curator`, then run:

```powershell
git remote add origin https://github.com/<your-username>/greenwaste-dataset-curator.git
git push -u origin main
```

If GitHub CLI is installed and authenticated, this can be done in one step:

```powershell
gh repo create greenwaste-dataset-curator --private --source . --remote origin --push
```

For YOLO proposal generation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev,yolo]
```

## Collect Wikimedia Commons Images

```powershell
.\.venv\Scripts\greenwaste-curator.exe collect-commons `
  --query-config configs\greenwaste_queries.json `
  --output-dir dataset\commons_greenwaste `
  --images-per-query 25 `
  --min-width 640 `
  --min-height 480 `
  --delay-seconds 3 `
  --max-retries 8 `
  --backoff-seconds 10
```

Output:

```text
dataset/commons_greenwaste/
  images/
    beds_mattresses/
    chair_seating/
    sofa/
    storage/
    tables_desks/
  manifest.csv
```

The `manifest.csv` is the important audit trail. Keep it with the dataset.

If Wikimedia returns too many request errors, reduce `--images-per-query` and
increase `--delay-seconds`. The collector automatically waits and retries when
the server returns HTTP 429 rate-limit responses.

## Filter Bad Search Results

Keyword search is intentionally broad, so some collected images may be wrong:
for example beds with people lying on them, cropped/zoomed duplicates, benches
returned for bed queries, or unrelated buildings. Run a quality filter before
manual annotation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[yolo]

.\.venv\Scripts\greenwaste-curator.exe quality-filter `
  --manifest dataset\commons_greenwaste\manifest.csv `
  --output-dir dataset\commons_greenwaste_quality `
  --model yolo11n.pt `
  --model "D:\Green Waste\V1_GreenWaste\runs\detect\train_final_all_labelled_augmented_70ep_20260620\weights\last.pt" `
  --reject-person `
  --reject-non-photo `
  --strict-non-photo-check `
  --require-target-object `
  --reclassify-mismatched-category `
  --duplicate-phash-threshold 6 `
  --near-duplicate-action review `
  --confidence 0.20
```

This creates:

```text
dataset/commons_greenwaste_quality/
  accepted/
  review/
  rejected/
  accepted_manifest.csv
  review_manifest.csv
  rejected_manifest.csv
  quality_review.csv
```

The quality filter:

- rejects likely near-duplicates, including some cropped/zoomed variants;
- rejects images where YOLO detects a person when `--reject-person` is used;
- rejects likely cartoons, drawings, illustrations, sketches, diagrams, renders,
  icons, SVG/vector images, and other non-photo results when
  `--reject-non-photo` is used;
- rejects images where the expected object is not detected when
  `--require-target-object` is used;
- moves images into the detected GreenWaste category when
  `--reclassify-mismatched-category` is used and exactly one alternative
  category is detected;
- records the reason for every accept/reject decision in `quality_review.csv`.

For bed collection, `--require-target-object` means the image must contain a
YOLO-detected `bed`. A bench or shed image should therefore be rejected. Treat
this as a review aid rather than a perfect truth source; keep a quick manual
check before annotation.

If a search result lands in the wrong folder but clearly belongs to another
GreenWaste category, `--reclassify-mismatched-category` places it under the
detected category in `accepted/` and updates `accepted_manifest.csv`. For
example, a bed collected under `sofa` can be moved to `beds_mattresses`.
Ambiguous images with multiple detected GreenWaste categories are rejected for
manual review rather than automatically moved.

You can pass `--model` more than once. This is useful when one model is better
for beds/people and another model is better for GreenWaste-specific classes
such as storage. The filter combines detections from all supplied models.

Near-duplicates are sent to `review/` by default rather than `rejected/`. This
is intentional: visually similar sofas, chairs, or room scenes may still be
useful training examples. Only use `--near-duplicate-action reject` if you want
to discard near-duplicates automatically.

If the non-photo filter is too aggressive, rerun with:

```powershell
--reject-non-photo --no-non-photo-visual-check
```

That keeps filename/source keyword filtering but disables the visual heuristic.

## Create Grouped Train/Val/Test Splits

```powershell
.\.venv\Scripts\greenwaste-curator.exe split `
  --manifest dataset\commons_greenwaste\manifest.csv `
  --output-dir dataset\commons_greenwaste_splits `
  --train 0.70 `
  --val 0.15 `
  --test 0.15
```

Splitting is grouped by source object/page ID, so related images stay in the same split. This prevents leakage where near-identical source material appears in both train and validation/test sets.

## Generate YOLO Proposals for Manual Review

This does not create final ground truth. It creates detector proposals that can be corrected in an annotation tool.

```powershell
.\.venv\Scripts\greenwaste-curator.exe yolo-proposals `
  --manifest dataset\commons_greenwaste\manifest.csv `
  --model yolo26n.pt `
  --output-csv dataset\commons_greenwaste\yolo_proposals.csv `
  --confidence 0.25 `
  --image-size 960
```

The proposal CSV includes predicted class, confidence, and bounding box coordinates. Use it to prioritise manual annotation, not as an automatically trusted label source.

## Use With the Main GreenWaste Project

Recommended workflow:

1. Collect real-world candidate images with this repository.
2. Review `manifest.csv` for source/licence traceability.
3. Run `split` to produce grouped train/validation/test assignments.
4. Run `yolo-proposals` to create pre-annotation suggestions.
5. Import selected images into the annotation tool.
6. Export corrected YOLO labels.
7. Merge corrected labels into the main GreenWaste YOLO dataset.

## Academic Framing

Use this wording in the dissertation:

> A separate dataset curation pipeline was developed to address the domain gap between clean product imagery and natural capture conditions. The pipeline records source and licence metadata, filters low-quality images, removes exact and near duplicates, and performs grouped train/validation/test splitting to reduce leakage risk. Detector-generated proposals are used only to reduce manual annotation workload and are treated as weak annotations requiring human verification.

## Notes on Source Legality

Avoid scraping marketplace sites unless explicit permission and compatible terms are available. This tool is designed around permitted, auditable sources and stores licence fields so that future dataset decisions can be defended.
