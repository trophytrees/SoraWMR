import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from sorawm.server.db import init_db
from sorawm.server.worker import worker
from sorawm.webui.state import set_active_source
from sorawm.configs import BASE_MODEL_PATH, ACTIVE_MODEL_PATH
import shutil


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    base_weights = BASE_MODEL_PATH.resolve()
    best_weights = ACTIVE_MODEL_PATH.resolve()

    if not base_weights.exists() and best_weights.exists():
        shutil.copyfile(best_weights, base_weights)
        logger.info("Captured base weights snapshot at %s", base_weights)

    if not best_weights.exists() and base_weights.exists():
        shutil.copyfile(base_weights, best_weights)
        logger.info("Restored active weights from base snapshot.")

    if best_weights.exists():
        set_active_source(best_weights)
    elif base_weights.exists():
        set_active_source(base_weights)

    await init_db()
    logger.info("Database initialized")

    await worker.initialize()

    _ = asyncio.create_task(worker.run())

    logger.info("Application started successfully")

    yield

    logger.info("Shutting down...")
    logger.info("Application shutdown complete")
