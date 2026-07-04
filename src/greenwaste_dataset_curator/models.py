from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRecord:
    category: str
    query: str
    source: str
    source_id: str
    source_page: str
    image_url: str
    local_path: str
    width: int
    height: int
    sha256: str
    perceptual_hash: str
    license_name: str
    artist: str
    group_id: str


@dataclass(frozen=True)
class DetectorProposal:
    local_path: str
    category_hint: str
    predicted_class: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: int
    image_height: int
