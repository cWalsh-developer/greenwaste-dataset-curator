from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image, UnidentifiedImageError

from .dedupe import DedupeIndex
from .models import ImageRecord


API_URL = "https://commons.wikimedia.org/w/api.php"


class CommonsCollector:
    def __init__(
        self,
        output_dir: Path,
        min_width: int = 640,
        min_height: int = 480,
        delay_seconds: float = 0.5,
        phash_threshold: int = 4,
        user_agent: str = "GreenWasteDatasetCurator/0.1 academic image dataset research",
    ) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.min_width = min_width
        self.min_height = min_height
        self.delay_seconds = delay_seconds
        self.dedupe = DedupeIndex(phash_threshold=phash_threshold)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def search(self, query: str, limit: int = 50) -> Iterator[dict]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
        }
        response = self.session.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        yield from pages.values()

    def download_image(self, url: str) -> tuple[bytes, Image.Image] | None:
        try:
            response = self.session.get(url, timeout=45)
            response.raise_for_status()
            raw = response.content
            image = Image.open(io.BytesIO(raw))
            image.load()
            return raw, image
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            print(f"Failed to download {url}: {exc}")
            return None

    @staticmethod
    def metadata_value(extmetadata: dict, key: str, default: str = "Unknown") -> str:
        value = extmetadata.get(key, {}).get("value", default)
        return str(value).replace("\n", " ").strip()

    def process_page(self, page: dict, category: str, query: str) -> ImageRecord | None:
        image_info_list = page.get("imageinfo", [])
        if not image_info_list:
            return None

        info = image_info_list[0]
        image_url = info.get("url")
        if not image_url:
            return None

        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        if width < self.min_width or height < self.min_height:
            return None

        downloaded = self.download_image(image_url)
        if downloaded is None:
            return None
        raw, image = downloaded

        sha256_value = self.dedupe.sha256(raw)
        if self.dedupe.is_exact_duplicate(sha256_value):
            return None

        rgb_image = image.convert("RGB")
        phash_value = self.dedupe.phash(rgb_image)
        if self.dedupe.is_near_duplicate(phash_value):
            return None

        extmetadata = info.get("extmetadata", {})
        license_name = self.metadata_value(extmetadata, "LicenseShortName")
        artist = self.metadata_value(extmetadata, "Artist")

        category_dir = self.image_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{category}_{sha256_value[:16]}.jpg"
        output_path = category_dir / filename
        rgb_image.save(output_path, format="JPEG", quality=95)

        self.dedupe.add(sha256_value, phash_value)
        page_id = str(page.get("pageid", ""))
        group_id = f"wikimedia_commons:{page_id}"

        return ImageRecord(
            category=category,
            query=query,
            source="wikimedia_commons",
            source_id=page_id,
            source_page=f"https://commons.wikimedia.org/?curid={page_id}",
            image_url=image_url,
            local_path=str(output_path),
            width=width,
            height=height,
            sha256=sha256_value,
            perceptual_hash=str(phash_value),
            license_name=license_name,
            artist=artist,
            group_id=group_id,
        )

    def collect(self, query_map: dict[str, list[str]], images_per_query: int) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for category, queries in query_map.items():
            for query in queries:
                print(f"Searching '{query}' for category '{category}'")
                for page in self.search(query=query, limit=images_per_query):
                    record = self.process_page(page=page, category=category, query=query)
                    if record is not None:
                        records.append(record)
                        print(f"Saved {record.local_path}")
                    time.sleep(self.delay_seconds)
        return records
