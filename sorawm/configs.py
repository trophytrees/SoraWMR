from pathlib import Path

ROOT = Path(__file__).parent.parent


RESOURCES_DIR = ROOT / "resources"
WATER_MARK_TEMPLATE_IMAGE_PATH = RESOURCES_DIR / "watermark_template.png"

WATER_MARK_DETECT_YOLO_WEIGHTS = RESOURCES_DIR / "best.pt"

OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


DEFAULT_WATERMARK_REMOVE_MODEL = "lama"

WORKING_DIR = ROOT / "working_dir"
WORKING_DIR.mkdir(exist_ok=True, parents=True)

LOGS_PATH = ROOT / "logs"
LOGS_PATH.mkdir(exist_ok=True, parents=True)

DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True, parents=True)

SQLITE_PATH = DATA_PATH / "db.sqlite3"

THUMBNAILS_DIR = WORKING_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True, parents=True)

FRAMES_DIR = WORKING_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True, parents=True)

VIDEO_UPLOADS_DIR = WORKING_DIR / "uploads"
VIDEO_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

ANNOTATIONS_DIR = ROOT / "datasets" / "manual_annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True, parents=True)

ANNOTATION_IMAGES_DIR = ANNOTATIONS_DIR / "images"
ANNOTATION_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

ANNOTATION_LABELS_DIR = ANNOTATIONS_DIR / "labels"
ANNOTATION_LABELS_DIR.mkdir(exist_ok=True, parents=True)

ANNOTATION_DATA_YAML = ANNOTATIONS_DIR / "data.yaml"

TRAINING_VIDEO_DIR = ROOT / "datasets" / "training_sources"
TRAINING_VIDEO_DIR.mkdir(exist_ok=True, parents=True)

BACKUPS_DIR = ROOT / "backups"
BACKUPS_DIR.mkdir(exist_ok=True, parents=True)

BASE_MODEL_PATH = RESOURCES_DIR / "best_original.pt"
ACTIVE_MODEL_PATH = RESOURCES_DIR / "best.pt"

ACTIVE_VIDEO_FILE = RESOURCES_DIR / "active_video.txt"
