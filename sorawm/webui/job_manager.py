from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Job:
    id: str
    type: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(100, progress))
            if message is not None:
                job.message = message
            if result is not None:
                job.result.update(result)
            if error is not None:
                job.error = error
            job.updated_at = datetime.utcnow()

    def submit(
        self,
        job_type: str,
        func: Callable[..., Any],
        *args,
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, type=job_type)
        with self._lock:
            self._jobs[job_id] = job

        def wrapped():
            try:
                self._update_job(job_id, status=JobStatus.RUNNING, progress=1)

                def cb(value: int, total: int = 100):
                    pct = int((value / total) * 100)
                    self._update_job(job_id, progress=pct)
                    if progress_callback:
                        progress_callback(pct)

                kwargs_with_cb = dict(kwargs)
                kwargs_with_cb.setdefault("progress_callback", cb)
                result = func(*args, **kwargs_with_cb)
                self._update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    progress=100,
                    result={"output": result},
                )
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                self._update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    error=str(exc),
                    message="Job failed",
                )

        self._executor.submit(wrapped)
        return job_id

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()

