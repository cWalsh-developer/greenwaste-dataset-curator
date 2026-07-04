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
  --min-height 480
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
