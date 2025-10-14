from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sorawm.configs import OUTPUT_DIR, ROOT, THUMBNAILS_DIR, WORKING_DIR
from sorawm.server.lifespan import lifespan
from sorawm.server.router import router
from sorawm.webui.router import router as webui_router


def init_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.include_router(webui_router)

    static_dir = ROOT / "static"
    static_dir.mkdir(exist_ok=True, parents=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True, parents=True)
    app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

    preview_dir = ROOT / "preview_outputs"
    preview_dir.mkdir(exist_ok=True, parents=True)
    app.mount("/preview_outputs", StaticFiles(directory=preview_dir), name="preview_outputs")

    uploads_dir = WORKING_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

    return app
