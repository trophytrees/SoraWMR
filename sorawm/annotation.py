from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import requests
from loguru import logger

from sorawm.configs import (
    ANNOTATION_DATA_YAML,
    ANNOTATION_IMAGES_DIR,
    ANNOTATION_LABELS_DIR,
    ANNOTATIONS_DIR,
)
from sorawm.utils.video_utils import save_frame_image

LABEL_TO_ID = {
    "watermark": 0,
    "watermark_text": 1,
    "watermark_icon": 2,
}


def ensure_manual_annotation_dataset() -> None:
    """
    Make sure the manual annotation dataset folders and YAML manifest exist.
    """
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    def as_posix(path: Path) -> str:
        return str(path.resolve()).replace("\\", "/")

    expected_content = "\n".join(
        [
            f"path: {as_posix(ANNOTATIONS_DIR)}",
            f"train: {as_posix(ANNOTATION_IMAGES_DIR)}",
            f"val: {as_posix(ANNOTATION_IMAGES_DIR)}",
            "names:",
            "  0: watermark",
            "  1: watermark_text",
            "  2: watermark_icon",
            "",
        ]
    )

    # Always ensure the manifest reflects the current workspace layout so training
    # jobs resolve absolute directories correctly (Ultralytics respects the last write).
    current = ANNOTATION_DATA_YAML.read_text(encoding="utf-8") if ANNOTATION_DATA_YAML.exists() else ""
    if current != expected_content:
        ANNOTATION_DATA_YAML.write_text(expected_content, encoding="utf-8")


@dataclass
class AnnotationBox:
    x: float
    y: float
    width: float
    height: float
    label: str = "watermark"

    def clamp(self) -> "AnnotationBox":
        """
        Ensure values stay within 0..1 and width/height are positive.
        """
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        width = min(max(self.width, 0.0), 1.0 - x)
        height = min(max(self.height, 0.0), 1.0 - y)
        return AnnotationBox(x=x, y=y, width=width, height=height, label=self.label)


def save_annotation_sample(
    video_path: Path,
    timestamp: float,
    boxes: Sequence[AnnotationBox],
) -> dict:
    """
    Save a single annotated frame (image + YOLO label file) into the manual dataset.
    """
    ensure_manual_annotation_dataset()
    if not boxes:
        raise ValueError("No annotation boxes provided.")

    base_name = f"{video_path.stem}_{int(timestamp * 1000):08d}"
    image_path = ANNOTATION_IMAGES_DIR / f"{base_name}.jpg"
    label_path = ANNOTATION_LABELS_DIR / f"{base_name}.txt"

    width, height, actual_ts = save_frame_image(video_path, image_path, timestamp)
    logger.debug("Saved annotated frame at %s", image_path)

    lines: list[str] = []
    for box in boxes:
        clamped = box.clamp()
        label_id = LABEL_TO_ID.get(clamped.label, LABEL_TO_ID["watermark"])
        x_center = clamped.x + clamped.width / 2
        y_center = clamped.y + clamped.height / 2
        lines.append(
            f"{label_id} {x_center:.6f} {y_center:.6f} {clamped.width:.6f} {clamped.height:.6f}"
        )

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug("Stored label file %s", label_path)
    return {
        "image_path": image_path,
        "label_path": label_path,
        "timestamp": actual_ts,
        "width": width,
        "height": height,
        "dataset_yaml": ANNOTATION_DATA_YAML,
    }


def _count_label_boxes(label_path: Path) -> int:
    count = 0
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def list_annotation_samples(limit: int | None = None) -> list[dict]:
    ensure_manual_annotation_dataset()
    samples: list[dict] = []
    label_files = sorted(ANNOTATION_LABELS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if limit:
        label_files = label_files[:limit]
    for label_path in label_files:
        boxes = _count_label_boxes(label_path)
        stem = label_path.stem
        image_path = ANNOTATION_IMAGES_DIR / f"{stem}.jpg"
        samples.append(
            {
                "id": stem,
                "image_path": image_path,
                "label_path": label_path,
                "boxes": boxes,
                "modified": label_path.stat().st_mtime,
            }
        )
    return samples


def annotation_dataset_summary() -> dict:
    ensure_manual_annotation_dataset()
    labels = list(ANNOTATION_LABELS_DIR.glob("*.txt"))
    images = list(ANNOTATION_IMAGES_DIR.glob("*.jpg"))
    total_boxes = sum(_count_label_boxes(label) for label in labels)
    return {
        "samples": len(labels),
        "frames": len(labels),
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "dataset_yaml": ANNOTATION_DATA_YAML,
    }


def auto_label_with_roboflow(
    *,
    image_dir: Path,
    api_key: str,
    workspace: str,
    workflow: str,
    api_url: str = "https://detect.roboflow.com",
    confidence: float | None = None,
    overwrite: bool = True,
    map_class_id: int = 0,
    progress_callback=None,
) -> dict:
    """
    Submit images to a Roboflow workflow and store YOLO label files locally.
    """
    ensure_manual_annotation_dataset()
    image_dir = image_dir if image_dir.is_absolute() else (Path.cwd() / image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    summary = {"processed": 0, "skipped": 0, "boxes": 0}
    total = len(images)
    endpoint = f"{api_url.rstrip('/')}/{workspace}/{workflow}"

    for idx, image_path in enumerate(images, 1):
        label_path = image_path.with_suffix(".txt")
        if not overwrite and label_path.exists():
            summary["skipped"] += 1
            if progress_callback:
                progress_callback(idx, total)
            continue

        with image_path.open("rb") as fp:
            response = requests.post(
                endpoint,
                params={"api_key": api_key},
                files={"file": (image_path.name, fp, "application/octet-stream")},
            )
        response.raise_for_status()
        payload = response.json()
        predictions = payload.get("predictions") or payload.get("preds") or []

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to load %s for metadata, skipping.", image_path)
            continue
        height, width = image.shape[:2]

        boxes_written = []
        for pred in predictions:
            conf = pred.get("confidence") or pred.get("confidence_score") or 0.0
            if confidence is not None and conf < confidence:
                continue
            x = pred.get("x") or pred.get("center_x")
            y = pred.get("y") or pred.get("center_y")
            w = pred.get("width") or pred.get("w")
            h = pred.get("height") or pred.get("h")
            if None in (x, y, w, h):
                continue
            x_norm = float(x) / width
            y_norm = float(y) / height
            w_norm = float(w) / width
            h_norm = float(h) / height
            boxes_written.append(
                f"{map_class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        if boxes_written:
            label_path.write_text("\n".join(boxes_written) + "\n", encoding="utf-8")
            summary["processed"] += 1
            summary["boxes"] += len(boxes_written)
        else:
            # If no predictions meet the threshold, ensure label file is removed.
            if label_path.exists():
                label_path.unlink()

        if progress_callback:
            progress_callback(idx, total)

    return summary
