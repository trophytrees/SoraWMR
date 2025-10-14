from __future__ import annotations
from datetime import datetime

import re
import shutil
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from sorawm.annotation import (
    AnnotationBox as ManualAnnotationBox,
    annotation_dataset_summary,
    list_annotation_samples,
    auto_label_with_roboflow,
    ensure_manual_annotation_dataset,
    save_annotation_sample,
)
from sorawm.configs import (
    ACTIVE_MODEL_PATH,
    BACKUPS_DIR,
    BASE_MODEL_PATH,
    OUTPUT_DIR,
    TRAINING_VIDEO_DIR,
    RESOURCES_DIR,
    ROOT,
    THUMBNAILS_DIR,
    VIDEO_UPLOADS_DIR,
)
from sorawm.preview import generate_detection_preview
from sorawm.training import fine_tune_detector, TrainingError
from sorawm.utils.video_utils import (
    capture_frame,
    frame_to_base64,
    get_video_metadata,
)
from sorawm.webui.job_manager import job_manager
from sorawm.webui.state import (
    get_active_source,
    get_active_video,
    set_active_source,
    set_active_video,
)
from sorawm.server.db import get_session
from sorawm.server.models import Task
from sorawm.server.worker import worker

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def _resolve_path_within_root(candidate: Path) -> Path:
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must stay within project workspace.")
    return resolved


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
}

VIDEO_SOURCE_DIRECTORIES: list[tuple[Path, str, str | None]] = [
    (VIDEO_UPLOADS_DIR, "upload", "/uploads"),
    (TRAINING_VIDEO_DIR, "training", None),
    (ROOT / "outputs", "output", "/outputs"),
    (ROOT / "preview_outputs", "preview", "/preview_outputs"),
    (OUTPUT_DIR, "output", "/output"),
]

ALLOWED_VIDEO_ROOTS = [directory.resolve() for directory, _, _ in VIDEO_SOURCE_DIRECTORIES]


def _safe_filename(filename: str, default_stem: str = "clip") -> str:
    candidate = Path(filename or "")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", candidate.stem) or default_stem
    suffix = candidate.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix or 'unknown'}")
    return f"{stem}{suffix}"


def _safe_model_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", (name or "").strip())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Model name cannot be empty")
    if not cleaned.lower().endswith(".pt"):
        cleaned = f"{cleaned}.pt"
    return cleaned


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _allowed_video_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(_is_within(resolved, root) for root in ALLOWED_VIDEO_ROOTS)


def _disambiguate_path(directory: Path, desired_name: str) -> Path:
    base = Path(desired_name)
    candidate = directory / base.name
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base.stem}_{counter}{base.suffix}"
        counter += 1
    return candidate


def _make_video_entry(
    path: Path,
    category: str,
    mount_prefix: str | None,
    base_dir: Path,
    active_resolved: Path | None,
) -> dict | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    rel = path.relative_to(ROOT).as_posix()
    url = None
    if mount_prefix:
        try:
            rel_to_mount = path.relative_to(base_dir).as_posix()
            url = f"{mount_prefix}/{rel_to_mount}"
        except ValueError:
            url = None
    return {
        "path": rel,
        "name": path.name,
        "modified": stat.st_mtime,
        "size": stat.st_size,
        "category": category,
        "active": bool(active_resolved and path.resolve() == active_resolved),
        "url": url,
    }


def _collect_video_entries() -> dict[str, dict]:
    active_raw = get_active_video(relative=False)
    active_resolved = Path(active_raw).resolve() if active_raw else None
    entries: dict[str, dict] = {}
    for directory, category, mount_prefix in VIDEO_SOURCE_DIRECTORIES:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            entry = _make_video_entry(path, category, mount_prefix, directory, active_resolved)
            if entry:
                entries[entry["path"]] = entry
    return entries


class AnnotationBoxPayload(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)
    label: str = "watermark"


class SaveAnnotationPayload(BaseModel):
    video_path: Path
    timestamp: float = Field(..., ge=0.0)
    boxes: list[AnnotationBoxPayload]


@router.get("/", response_class=HTMLResponse)
async def webui_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "year": datetime.utcnow().year})


@router.get("/api/jobs")
def list_jobs():
    jobs = [
        {
            "id": job.id,
            "type": job.type,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        for job in job_manager.list_jobs()
    ]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.post("/api/annotate")
def trigger_auto_annotation(
    image_dir: Path = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
    workspace: str = Body(..., embed=True),
    workflow: str = Body(..., embed=True),
    confidence: float | None = Body(None, embed=True),
    overwrite: bool = Body(True, embed=True),
):
    image_dir = (ROOT / image_dir).resolve() if not image_dir.is_absolute() else image_dir
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {image_dir}")

    def task(progress_callback):
        return auto_label_with_roboflow(
            image_dir=image_dir,
            api_key=api_key,
            workspace=workspace,
            workflow=workflow,
            confidence=confidence,
            overwrite=overwrite,
            map_class_id=0,
            progress_callback=lambda idx, total: progress_callback(int(idx / total * 100)),
        )

    job_id = job_manager.submit(job_type="AUTO_ANNOTATE", func=task)
    job_manager._update_job(job_id, message=f"Annotating images in {image_dir}")
    return {"job_id": job_id}


@router.post("/api/train")
def trigger_training(
    data_yaml: Path = Body(..., embed=True),
    epochs: int = Body(10, embed=True),
    lr0: float = Body(5e-4, embed=True),
    lrf: float = Body(5e-4, embed=True),
    batch: int = Body(16, embed=True),
    device: str | int = Body(0, embed=True),
    weights_path: Path | None = Body(None, embed=True),
):
    data_yaml = (ROOT / data_yaml).resolve() if not data_yaml.is_absolute() else data_yaml
    if not data_yaml.exists():
        raise HTTPException(status_code=400, detail=f"Data yaml not found: {data_yaml}")

    if weights_path is not None:
        resolved_weights = _resolve_path_within_root(weights_path)
    else:
        resolved_weights = ACTIVE_MODEL_PATH.resolve()
    if not resolved_weights.exists():
        raise HTTPException(status_code=400, detail=f"Weights file not found: {resolved_weights}")
    if resolved_weights.suffix != ".pt":
        raise HTTPException(status_code=400, detail="Only .pt weight files are supported")
    start_weights_rel = str(resolved_weights.relative_to(ROOT)).replace('\\', '/')
    started_from_base = resolved_weights.resolve() == BASE_MODEL_PATH.resolve()

    def task(progress_callback):
        try:
            best_path = fine_tune_detector(
                weights_path=resolved_weights,
                data_yaml=data_yaml,
                epochs=epochs,
                lr0=lr0,
                lrf=lrf,
                batch=batch,
                device=device,
                workers=0,
                project=ROOT / "runs" / "detect",
                name="finetune_webui",
                progress_callback=lambda done, total: progress_callback(int(done / total * 100)),
            )
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            backup_dir = BACKUPS_DIR / "models"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{timestamp}_{best_path.name}"
            shutil.copyfile(best_path, backup_path)
            return {
                "best_weights": str(best_path.relative_to(ROOT)).replace('\\', '/'),
                "backup_path": str(backup_path.relative_to(ROOT)).replace('\\', '/'),
                "start_weights": start_weights_rel,
                "started_from_base": started_from_base,
                "base_model_path": str(BASE_MODEL_PATH.relative_to(ROOT)).replace('\\', '/'),
            }
        except TrainingError as exc:
            raise RuntimeError(str(exc)) from exc

    job_id = job_manager.submit(job_type="TRAINING", func=task)
    job_manager._update_job(
        job_id, message=f"Training with data {data_yaml} (start: {start_weights_rel})"
    )
    return {"job_id": job_id}


@router.post("/api/preview")
def trigger_preview(
    video_path: Path = Body(..., embed=True),
    conf: float = Body(0.25, embed=True),
    iou: float = Body(0.9, embed=True),
):
    video_path = (ROOT / video_path).resolve() if not video_path.is_absolute() else video_path
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")

    output_dir = ROOT / "preview_outputs"
    weights_path = RESOURCES_DIR / "best.pt"

    def task(progress_callback):
        return {"preview": generate_detection_preview(
            weights_path=weights_path,
            video_path=video_path,
            output_dir=output_dir,
            conf=conf,
            iou=iou,
        )}

    job_id = job_manager.submit(job_type="PREVIEW", func=task)
    job_manager._update_job(job_id, message=f"Generating preview for {video_path.name}")
    return {"job_id": job_id}


@router.get("/api/video-sources")
def list_video_sources():
    entries = _collect_video_entries()
    ordered = sorted(entries.values(), key=lambda item: item["modified"], reverse=True)
    return {"videos": ordered, "active": get_active_video()}


@router.post("/api/video-sources/active")
def activate_video_source(video_path: Path | None = Body(None, embed=True)):
    if video_path is None or str(video_path).strip() == "":
        set_active_video(None)
        return {"active": None}
    candidate = _resolve_path_within_root(video_path)
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only video files can be activated")
    if not _allowed_video_path(candidate):
        raise HTTPException(status_code=400, detail="Video outside managed directories")
    set_active_video(candidate)
    return {"active": get_active_video()}


@router.delete("/api/video-sources")
async def delete_video_sources(
    video_path: Path | None = Body(None, embed=True),
    video_paths: list[Path] | None = Body(None, embed=True),
):
    candidates: list[Path] = []
    if video_paths:
        candidates.extend(video_paths)
    if video_path:
        candidates.append(video_path)
    if not candidates:
        raise HTTPException(status_code=400, detail="Provide at least one video path to delete")

    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    active_absolute = get_active_video(relative=False)
    active_resolved = Path(active_absolute).resolve() if active_absolute else None

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            target = _resolve_path_within_root(candidate)
        except HTTPException as exc:
            errors.append({"path": str(candidate), "error": exc.detail})
            continue

        target_resolved = target.resolve()
        if target_resolved in seen:
            continue
        seen.add(target_resolved)

        if not target_resolved.exists():
            errors.append({"path": str(candidate), "error": "Video not found"})
            continue
        if not target_resolved.is_file():
            errors.append({"path": str(candidate), "error": "Target is not a file"})
            continue
        if not _allowed_video_path(target_resolved):
            errors.append({"path": str(candidate), "error": "Video outside managed directories"})
            continue

        try:
            target_resolved.unlink()
            deleted.append(str(target_resolved.relative_to(ROOT)).replace("\\", "/"))
        except Exception as exc:  # pragma: no cover - filesystem errors
            errors.append({"path": str(candidate), "error": str(exc)})

    if active_resolved and (not active_resolved.exists() or active_resolved in seen):
        set_active_video(None)

    if not deleted and errors:
        raise HTTPException(status_code=400, detail=errors)

    return {"deleted": deleted, "errors": errors}


@router.post("/api/training-videos")
async def upload_training_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    filename = _safe_filename(video.filename, default_stem="training")
    target = _disambiguate_path(TRAINING_VIDEO_DIR, filename)
    try:
        async with aiofiles.open(target, "wb") as outfile:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                await outfile.write(chunk)
    finally:
        await video.close()

    active_raw = get_active_video(relative=False)
    active_resolved = Path(active_raw).resolve() if active_raw else None
    entry = _make_video_entry(target, "training", None, TRAINING_VIDEO_DIR, active_resolved)
    if entry is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported video format")
    return entry


@router.get("/api/video-meta")
def api_video_meta(video_path: Path = Query(..., alias="video_path")):
    candidate = _resolve_path_within_root(video_path)
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        metadata = get_video_metadata(candidate)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    relative = str(candidate.relative_to(ROOT)).replace("\\", "/")
    metadata["path"] = relative
    return metadata


@router.get("/api/frame")
def api_video_frame(video_path: Path = Query(..., alias="video_path"), timestamp: float = Query(0.0)):
    candidate = _resolve_path_within_root(video_path)
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        frame, actual_timestamp = capture_frame(candidate, timestamp)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    image_url = frame_to_base64(frame)
    height, width = frame.shape[:2]
    return {
        "image": image_url,
        "width": width,
        "height": height,
        "timestamp": actual_timestamp,
        "video": str(candidate.relative_to(ROOT)).replace("\\", "/"),
    }


@router.post("/api/annotations/manual")
def api_save_annotations(payload: SaveAnnotationPayload):
    if not payload.boxes:
        raise HTTPException(status_code=400, detail="No annotation boxes received.")

    resolved_video = _resolve_path_within_root(payload.video_path)
    if not resolved_video.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    manual_boxes = [
        ManualAnnotationBox(**box.dict()) for box in payload.boxes
    ]
    try:
        result = save_annotation_sample(resolved_video, payload.timestamp, manual_boxes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = annotation_dataset_summary()
    dataset_yaml = summary.get("dataset_yaml")
    if isinstance(dataset_yaml, Path):
        summary["dataset_yaml"] = str(dataset_yaml.relative_to(ROOT)).replace("\\", "/")

    return {
        "saved": len(manual_boxes),
        "image_path": str(result["image_path"].relative_to(ROOT)).replace("\\", "/"),
        "label_path": str(result["label_path"].relative_to(ROOT)).replace("\\", "/"),
        "timestamp": result["timestamp"],
        "width": result["width"],
        "height": result["height"],
        "dataset_yaml": summary["dataset_yaml"],
        "frames": summary.get("frames", summary.get("samples", 0)),
        "boxes": summary.get("boxes", 0),
        "summary": summary,
    }


@router.get("/api/annotations/summary")
def api_annotation_summary():
    ensure_manual_annotation_dataset()
    summary = annotation_dataset_summary()
    dataset_yaml = summary.get("dataset_yaml")
    if isinstance(dataset_yaml, Path):
        summary["dataset_yaml"] = str(dataset_yaml.relative_to(ROOT)).replace("\\", "/")
    return summary


@router.get("/api/annotations/samples")
def api_annotation_samples():
    ensure_manual_annotation_dataset()
    samples = []
    for sample in list_annotation_samples():
        image_path = sample.get("image_path")
        if isinstance(image_path, Path) and image_path.exists():
            image_rel = "/" + str(image_path.relative_to(ROOT)).replace("\\", "/")
        else:
            image_rel = None
        label_rel = str(sample["label_path"].relative_to(ROOT)).replace("\\", "/")
        samples.append(
            {
                "id": sample["id"],
                "image_path": image_rel,
                "label_path": label_rel,
                "boxes": sample["boxes"],
                "modified": sample["modified"],
            }
        )
    total_boxes = sum(item["boxes"] for item in samples)
    return {"samples": samples, "total_boxes": total_boxes}


@router.get("/api/removal-tasks")
async def list_removal_tasks():
    async with get_session() as session:
        result = await session.execute(select(Task))
        tasks = result.scalars().all()
    payload = []
    for task in tasks:
        try:
            rel_video = str(Path(task.video_path).resolve().relative_to(ROOT))
        except Exception:
            rel_video = task.video_path
        thumbnail = None
        if task.output_path:
            thumb_candidate = THUMBNAILS_DIR / f"{Path(task.output_path).stem}.jpg"
            if thumb_candidate.exists():
                try:
                    thumbnail = "/" + str(thumb_candidate.relative_to(ROOT)).replace('\\', '/')
                except ValueError:
                    thumbnail = None
        payload.append(
            {
                "id": task.id,
                "status": task.status,
                "percentage": task.percentage,
                "download_url": task.download_url,
                "video_path": rel_video.replace('\\', '/'),
                "thumbnail": thumbnail,
                "created_at": task.created_at.timestamp() if task.created_at else None,
            }
        )
    payload.sort(key=lambda item: item["created_at"] or 0, reverse=True)
    return {"tasks": payload}


@router.get("/api/models")
def list_models():
    resources = list(RESOURCES_DIR.glob("*.pt"))
    backups = list(BACKUPS_DIR.glob("**/*.pt"))
    runs = list((ROOT / "runs").glob("**/best*.pt"))

    active_source_abs = get_active_source(relative=False)
    active_source_resolved = Path(active_source_abs).resolve() if active_source_abs else None
    canonical_target = ACTIVE_MODEL_PATH.resolve()
    base_weights = BASE_MODEL_PATH.resolve()

    def to_entry(path: Path):
        resolved = path.resolve()
        source = "candidate"
        is_default = resolved == base_weights
        if is_default:
            source = "base"
        elif resolved == canonical_target:
            source = "active-copy"
        is_active = (
            resolved == active_source_resolved
            if active_source_resolved
            else resolved == canonical_target
        )
        protected = resolved in {base_weights, canonical_target}
        deletable = (
            not protected
            and not (active_source_resolved and resolved == active_source_resolved)
            and resolved.exists()
        )
        return {
            "path": str(path.relative_to(ROOT)).replace('\\', '/'),
            "name": path.name,
            "modified": path.stat().st_mtime,
            "size": path.stat().st_size,
            "active": is_active,
            "source": source,
            "protected": protected,
            "deletable": deletable,
            "default": is_default,
        }

    entries = [to_entry(p) for p in resources + backups + runs]
    entries.sort(key=lambda e: e["modified"], reverse=True)
    active_source_relative = get_active_source()
    active_target = str(ACTIVE_MODEL_PATH.relative_to(ROOT)).replace('\\', '/')
    return {
        "models": entries,
        "active_source": active_source_relative,
        "active_target": active_target,
        "base_path": str(base_weights.relative_to(ROOT)).replace('\\', '/'),
    }


@router.post("/api/models/activate")
async def activate_model(model_path: Path = Body(..., embed=True)):
    candidate = _resolve_path_within_root(model_path)
    if not candidate.exists():
        raise HTTPException(status_code=400, detail="Model file not found")
    if candidate.suffix != ".pt":
        raise HTTPException(status_code=400, detail="Only .pt files are supported")

    target = ACTIVE_MODEL_PATH.resolve()

    shutil.copyfile(candidate, target)
    set_active_source(candidate)
    await worker.reload_models()
    logger.info("Activated model %s -> %s", candidate, target)
    return {"message": "Model activated", "path": str(candidate.relative_to(ROOT)).replace('\\', '/'), "active_source": get_active_source()}


@router.delete("/api/models")
async def delete_model(
    model_path: Path | None = Body(None, embed=True),
    model_paths: list[Path] | None = Body(None, embed=True),
):
    candidates: list[Path] = []
    if model_paths:
        candidates.extend(model_paths)
    if model_path:
        candidates.append(model_path)
    if not candidates:
        raise HTTPException(status_code=400, detail="Provide at least one model path to delete")

    active_source_abs = get_active_source(relative=False)
    active_source_resolved = Path(active_source_abs).resolve() if active_source_abs else None
    protected_paths = {BASE_MODEL_PATH.resolve(), ACTIVE_MODEL_PATH.resolve()}

    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    seen: set[Path] = set()

    for candidate in candidates:
        try:
            target = _resolve_path_within_root(candidate)
        except HTTPException as exc:
            errors.append({"path": str(candidate), "error": exc.detail})
            continue

        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        if not resolved.exists():
            errors.append({"path": str(candidate), "error": "Model file not found"})
            continue
        if resolved.is_dir():
            errors.append({"path": str(candidate), "error": "Cannot delete directories"})
            continue
        if resolved in protected_paths or (active_source_resolved and resolved == active_source_resolved):
            errors.append({"path": str(candidate), "error": "This model is protected or currently active."})
            continue

        try:
            resolved.unlink()
            rel = str(resolved.relative_to(ROOT)).replace("\\", "/")
            deleted.append(rel)
            logger.info("Deleted model weights at %s", resolved)
        except Exception as exc:  # pragma: no cover - filesystem errors
            errors.append({"path": str(candidate), "error": str(exc)})

    if not deleted and errors:
        raise HTTPException(status_code=400, detail=errors)

    return {"deleted": deleted, "errors": errors}


@router.post("/api/models/rename")
async def rename_model(
    model_path: Path = Body(..., embed=True),
    new_name: str = Body(..., embed=True),
):
    candidate = _resolve_path_within_root(model_path)
    resolved = candidate.resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    if resolved.is_dir():
        raise HTTPException(status_code=400, detail="Cannot rename directories")

    safe_name = _safe_model_name(new_name)
    target = resolved.with_name(safe_name)

    if target.exists():
        raise HTTPException(status_code=400, detail="A model with that name already exists")

    protected_paths = {BASE_MODEL_PATH.resolve(), ACTIVE_MODEL_PATH.resolve()}
    active_source_abs = get_active_source(relative=False)
    active_source_resolved = Path(active_source_abs).resolve() if active_source_abs else None

    if resolved in protected_paths or (active_source_resolved and resolved == active_source_resolved):
        raise HTTPException(status_code=400, detail="This model is protected or currently active.")

    try:
        resolved.rename(target)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Renamed model %s -> %s", resolved, target)
    return {
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "name": target.name,
    }


@router.get("/api/outputs")
def list_outputs():
    entries = _collect_video_entries()
    videos = []
    for entry in entries.values():
        video = {
            "path": entry["path"],
            "name": entry["name"],
            "modified": entry["modified"],
            "size": entry["size"],
            "category": entry["category"],
            "active": entry["active"],
        }
        if entry.get("url"):
            video["url"] = entry["url"]
        videos.append(video)
    videos.sort(key=lambda item: item["modified"], reverse=True)
    return {"videos": videos}
