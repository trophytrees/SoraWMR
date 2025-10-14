from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from loguru import logger
from ultralytics import YOLO

from sorawm.configs import ROOT


def generate_detection_preview(
    weights_path: Path,
    video_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.9,
    device: str | int = 0,
    project_name: Optional[str] = None,
) -> str:
    '''Run the detector on ``video_path`` and save an annotated video into ``output_dir``.

    Returns the relative path (from project root) to the generated annotated video.
    '''

    weights_path = weights_path.resolve()
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(
        "Generating detection preview for %s using weights %s", video_path, weights_path
    )

    model = YOLO(str(weights_path))

    temp_dir = output_dir / ".preview_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        project=str(temp_dir),
        name="run",
        exist_ok=True,
    )

    if not results:
        raise RuntimeError("YOLO predict returned no results.")

    save_dir = Path(results[0].save_dir)
    candidates = list(save_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"No annotated video generated in {save_dir}")

    target_path = output_dir / f"{(project_name or video_path.stem)}_preview.mp4"
    if target_path.exists():
        target_path.unlink()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidates[0]), target_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    relative = target_path.relative_to(ROOT)
    logger.info("Preview saved to %s", target_path)
    return str(relative).replace('\\', '/')
