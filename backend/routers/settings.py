import os
import threading
from pathlib import Path
from fastapi import APIRouter

from config import settings
from schemas.common import SettingsConfig

router = APIRouter()


@router.get("/api/settings")
async def get_settings():
    return settings._app_settings


@router.post("/api/settings")
async def update_settings(config: SettingsConfig):
    if config.datasets_path:
        new_path = Path(config.datasets_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        settings.DATASETS_DIR = new_path
        settings._app_settings["datasets_path"] = str(new_path)

    if config.models_path:
        new_path = Path(config.models_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        settings.MODELS_DIR = new_path
        settings._app_settings["models_path"] = str(new_path)

    if config.output_path:
        new_path = Path(config.output_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        settings.EXPORTS_DIR = new_path
        settings._app_settings["output_path"] = str(new_path)

    settings._app_settings["use_gpu"] = config.use_gpu
    settings._app_settings["gpu_device"] = config.gpu_device

    return {"success": True, "settings": settings._app_settings}


@router.post("/api/restart")
async def restart_server():
    """Trigger a uvicorn --reload by touching main.py."""
    def _touch():
        import time, pathlib
        time.sleep(0.3)
        # Touch this package's parent (backend/main.py)
        p = pathlib.Path(__file__).parent.parent / "main.py"
        p.touch()
    threading.Thread(target=_touch, daemon=True).start()
    return {"success": True, "message": "Reload triggered — uvicorn will restart in ~1 s"}


@router.post("/api/shutdown")
async def shutdown_server():
    import signal as _signal
    def _kill():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), _signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"success": True, "message": "Server shutting down"}
