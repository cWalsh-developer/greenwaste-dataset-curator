from __future__ import annotations

import hashlib

import imagehash
from PIL import Image


class DedupeIndex:
    def __init__(self, phash_threshold: int = 4) -> None:
        self.phash_threshold = phash_threshold
        self.sha256_values: set[str] = set()
        self.phashes: list[imagehash.ImageHash] = []

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def phash(image: Image.Image) -> imagehash.ImageHash:
        return imagehash.phash(image.convert("RGB"))

    def is_exact_duplicate(self, sha256_value: str) -> bool:
        return sha256_value in self.sha256_values

    def is_near_duplicate(self, phash_value: imagehash.ImageHash) -> bool:
        return any((phash_value - existing) <= self.phash_threshold for existing in self.phashes)

    def add(self, sha256_value: str, phash_value: imagehash.ImageHash) -> None:
        self.sha256_values.add(sha256_value)
        self.phashes.append(phash_value)
