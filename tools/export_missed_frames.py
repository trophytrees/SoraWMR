"""
Utility script to harvest frames where the current detector misses the Sora watermark.

Usage examples:
    python tools/export_missed_frames.py video.mov
    python tools/export_missed_frames.py --threshold 0.45 --stride 2 path/to/videos/*.mp4
    python tools/export_missed_frames.py --output datasets/missed_frames videos/list.txt

The script does **not** modify the original datasets or weights. It simply writes
PNG crops to the requested output directory so you can label and add them to a
fine-tuning set without breaking the existing pipeline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from loguru import logger

from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.utils.video_utils import VideoLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frames where the Sora watermark detector misses or has low confidence."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Video files or text files containing newline-separated video paths. "
            "Shell globs are supported (e.g. resources/*.mp4)."
        ),
    )
    parser.add_argument(
        "--output",
        default="datasets/missed_frames",
        help="Directory to store extracted frames (default: datasets/missed_frames).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold below which detections are treated as misses (default: 0.40).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame to save time. Use 1 to evaluate all frames (default: 1).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV file to append metadata about each exported frame.",
    )
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: list[Path] = []
    for entry in patterns:
        path = Path(entry)
        if path.is_file() and path.suffix.lower() in {".txt", ".lst"}:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        paths.append(Path(line))
        else:
            matches = list(path.parent.glob(path.name)) if "*" in path.name or "?" in path.name else [path]
            paths.extend(matches)
    unique = []
    seen = set()
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def init_metadata_writer(csv_path: Path) -> csv.writer:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if not file_exists:
        writer.writerow(
            [
                "image_path",
                "source_video",
                "frame_index",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
            ]
        )
    return writer


def save_frame(
    frame: np.ndarray,
    destination: Path,
    video_name: str,
    frame_idx: int,
    confidence: float | None,
) -> Path:
    file_name = f"{Path(video_name).stem}_f{frame_idx:06d}_c{(confidence or 0):.2f}.png"
    out_path = destination / file_name
    cv2.imwrite(str(out_path), frame)
    return out_path


def export_for_video(
    video_path: Path,
    detector: SoraWaterMarkDetector,
    output_dir: Path,
    conf_threshold: float,
    stride: int,
    metadata_writer: csv.writer | None,
) -> int:
    loader = VideoLoader(video_path)
    exported = 0
    logger.info(f"Processing {video_path} ({loader.total_frames} frames @ {loader.fps:.2f} fps)")
    for idx, frame in enumerate(loader):
        if idx % stride:
            continue
        result = detector.detect(frame)
        conf = result["confidence"]
        should_export = not result["detected"] or (conf is not None and conf < conf_threshold)
        if should_export:
            saved_path = save_frame(frame, output_dir, video_path.name, idx, conf)
            exported += 1
            if metadata_writer is not None:
                bbox = result["bbox"] if result["bbox"] else (None, None, None, None)
                metadata_writer.writerow(
                    [
                        str(saved_path),
                        str(video_path),
                        idx,
                        conf if conf is not None else "",
                        bbox[0] if bbox else "",
                        bbox[1] if bbox else "",
                        bbox[2] if bbox else "",
                        bbox[3] if bbox else "",
                    ]
                )
    logger.info(f"Exported {exported} frames from {video_path.name}")
    return exported


def main() -> None:
    args = parse_args()
    videos = expand_inputs(args.inputs)
    if not videos:
        raise SystemExit("No valid video files were found for the supplied inputs.")

    output_dir = Path(args.output).resolve()
    ensure_output_dir(output_dir)

    metadata_writer = None
    if args.metadata:
        metadata_writer = init_metadata_writer(Path(args.metadata).resolve())

    detector = SoraWaterMarkDetector()
    total_exported = 0
    for video_path in videos:
        total_exported += export_for_video(
            video_path,
            detector,
            output_dir,
            conf_threshold=args.threshold,
            stride=max(1, args.stride),
            metadata_writer=metadata_writer,
        )

    logger.success(f"Done. Exported {total_exported} frames to {output_dir}")


if __name__ == "__main__":
    main()
