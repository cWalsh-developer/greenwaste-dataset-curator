from __future__ import annotations

from pathlib import Path

from PIL import Image

from .models import DetectorProposal, ImageRecord


YOLO_TO_GREENWASTE = {
    "bed": "beds_mattresses",
    "bench": "chair_seating",
    "chair": "chair_seating",
    "couch": "sofa",
    "dining table": "tables_desks",
}


def generate_yolo_proposals(
    records: list[ImageRecord],
    model_path: Path,
    confidence: float,
    image_size: int,
) -> list[DetectorProposal]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the yolo extra with: pip install -e .[yolo]") from exc

    model = YOLO(str(model_path))
    proposals: list[DetectorProposal] = []
    for record in records:
        image_path = Path(record.local_path)
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        result = model.predict(
            source=str(image_path),
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )[0]

        for box in result.boxes:
            class_id = int(box.cls.item())
            predicted_class = result.names[class_id]
            mapped_class = YOLO_TO_GREENWASTE.get(predicted_class, predicted_class)
            x_min, y_min, x_max, y_max = [float(value) for value in box.xyxy[0].tolist()]
            proposals.append(
                DetectorProposal(
                    local_path=record.local_path,
                    category_hint=record.category,
                    predicted_class=mapped_class,
                    confidence=float(box.conf.item()),
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    image_width=width,
                    image_height=height,
                )
            )
    return proposals
