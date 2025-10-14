"""
Create timestamped backups of the current datasets and detector weights without
touching the active files. This lets you experiment with fine-tuning while
keeping a pristine copy of the working setup.

Usage:
    python tools/backup_assets.py
    python tools/backup_assets.py --dest backups/custom_name
    python tools/backup_assets.py --skip-datasets
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

DEFAULT_DATASET_DIR = Path("datasets")
DEFAULT_WEIGHTS = Path("resources/best.pt")
DEFAULT_BACKUP_ROOT = Path("backups")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup datasets and detector weights.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory. If omitted, a timestamped folder is generated under ./backups",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory to back up (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Detector weights file to back up (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip copying the datasets directory.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip copying the weights file.",
    )
    return parser.parse_args()


def create_destination(dest: Path | None) -> Path:
    if dest is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = DEFAULT_BACKUP_ROOT / f"sorawm_backup_{timestamp}"
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backing up assets to {dest}")
    return dest


def copy_datasets(source: Path, destination: Path) -> None:
    dest_path = destination / source.name
    if dest_path.exists():
        logger.warning(f"{dest_path} already exists; skipping dataset backup.")
        return
    logger.info(f"Copying dataset directory {source} -> {dest_path}")
    shutil.copytree(source, dest_path)
    logger.success(f"Dataset backup saved at {dest_path}")


def copy_weights(source: Path, destination: Path) -> None:
    if not source.exists():
        logger.warning(f"Weight file {source} not found; skipping weights backup.")
        return
    dest_path = destination / source.name
    if dest_path.exists():
        logger.warning(f"{dest_path} already exists; skipping weights backup.")
        return
    shutil.copy2(source, dest_path)
    logger.success(f"Weights backup saved at {dest_path}")


def main() -> None:
    args = parse_args()
    dest_dir = create_destination(args.dest)

    if not args.skip_datasets:
        if args.datasets.exists():
            copy_datasets(args.datasets, dest_dir)
        else:
            logger.warning(f"Dataset directory {args.datasets} does not exist; skipping.")

    if not args.skip_weights:
        copy_weights(args.weights, dest_dir)

    logger.success("Backup complete.")


if __name__ == "__main__":
    main()
