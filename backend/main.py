"""
CV Dataset Manager - Main FastAPI Application
Thin entry point: app init, middleware, router registration, and lifespan events.
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core startup helpers (must import before routers so state is ready)
from core.startup import (
    _ensure_cuda_torch,
    _ensure_packages,
    _cleanup_temp,
    _restore_jobs,
)
from core.workspace import _restore_datasets

# Routers
from routers import datasets, annotations, training, models, formats, augmentation
from routers import health, snapshots, video, inference, evaluation
from routers import settings as settings_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_ensure_cuda_torch, daemon=True).start()
    threading.Thread(
        target=_ensure_packages,
        args=([
            ("huggingface_hub", "huggingface_hub"),
            ("ultralytics",     "ultralytics"),
            ("cv2",             "opencv-python"),
            ("PIL",             "Pillow"),
            ("psutil",          "psutil"),
            ("clip",            "git+https://github.com/ultralytics/CLIP.git"),
            ("ftfy",            "ftfy"),
        ],),
        daemon=True,
    ).start()
    _cleanup_temp()
    _restore_datasets()
    _restore_jobs()
    yield


app = FastAPI(
    title="CV Dataset Manager",
    description="Professional Computer Vision Dataset Management Suite",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(datasets.router)
app.include_router(annotations.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(formats.router)
app.include_router(augmentation.router)
app.include_router(health.router)
app.include_router(snapshots.router)
app.include_router(video.router)
app.include_router(settings_router.router)
app.include_router(inference.router)
app.include_router(evaluation.router)


@app.get("/")
async def root():
    return {
        "status": "running",
        "name": "CV Dataset Manager API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
