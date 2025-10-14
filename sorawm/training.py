from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger
from ultralytics import YOLO


class TrainingError(Exception):
    """Raised when training fails."""


def fine_tune_detector(
    weights_path: Path,
    data_yaml: Path,
    epochs: int = 10,
    lr0: float = 5e-4,
    lrf: float = 5e-4,
    batch: int = 16,
    device: str | int = 0,
    workers: int = 0,
    project: Path | None = None,
    name: str = "finetune",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Fine-tune the YOLO model located at `weights_path` on the dataset described by `data_yaml`.

    Returns the path to the best weights produced by training.
    """

    weights_path = weights_path.resolve()
    data_yaml = data_yaml.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    logger.info(
        "Starting fine-tune: weights=%s, data=%s, epochs=%s, lr0=%s, lrf=%s",
        weights_path,
        data_yaml,
        epochs,
        lr0,
        lrf,
    )

    model = YOLO(str(weights_path))

    def on_epoch_end(trainer):
        if progress_callback:
            progress_callback(trainer.epoch + 1, trainer.epochs)

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            lr0=lr0,
            lrf=lrf,
            batch=batch,
            device=device,
            workers=workers,
            project=str(project) if project else None,
            name=name,
            exist_ok=True,
        )
    except Exception as exc:
        raise TrainingError(f"Training failed: {exc}") from exc

    try:
        model.remove_callback("on_fit_epoch_end", on_epoch_end)
    except Exception:
        pass

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise TrainingError(f"Training finished but best weights not found at {best_path}")

    if progress_callback:
        progress_callback(epochs, epochs)

    logger.info("Fine-tune completed. Best weights at %s", best_path)
    return best_path
