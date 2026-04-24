from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

from config import settings

router = APIRouter()


@router.get("/api/models")
async def list_models():
    return {"models": settings.model_manager.list_models()}


@router.post("/api/models/load")
async def load_model(
    model_file: UploadFile = File(None),
    model_type: str = "yolo",
    model_name: str = None,
    pretrained: str = None,
):
    try:
        if model_file:
            model_path = settings.MODELS_DIR / (model_name or model_file.filename)
            content = await model_file.read()
            with open(model_path, "wb") as f:
                f.write(content)
            model_id = settings.model_manager.load_model(str(model_path), model_type)
        else:
            model_id = settings.model_manager.load_pretrained(model_type, pretrained)
        return {"success": True, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/models/download")
async def start_model_download(
    background_tasks: BackgroundTasks,
    model_type: str = Form(...),
    pretrained: str = Form(...),
    hf_token: Optional[str] = Form(None),
):
    if settings._download_status.get(pretrained, {}).get("status") == "downloading":
        return {"success": True, "already_started": True}

    settings._download_status[pretrained] = {
        "status": "downloading", "progress": 5, "phase": "downloading",
        "message": f"Downloading {pretrained}…",
    }

    def _do_download(mtype: str, mname: str, token: Optional[str]):
        try:
            def _phase_hook(phase: str):
                if phase == "loading_memory":
                    settings._download_status[mname] = {
                        "status": "downloading",
                        "progress": 80,
                        "phase": "loading_memory",
                        "message": "Loading model into memory… this may take a moment",
                    }

            settings._download_status[mname]["progress"] = 15
            settings.model_manager.load_pretrained(mtype, mname, hf_token=token, _status_hook=_phase_hook)
            info = settings.model_manager.loaded_models.get(mname, {})
            if info.get("error") and not info.get("path"):
                settings._download_status[mname] = {"status": "error", "progress": 0, "error": info["error"]}
            else:
                settings._download_status[mname] = {"status": "done", "progress": 100}
        except Exception as e:
            settings._download_status[mname] = {"status": "error", "progress": 0, "error": str(e)}

    background_tasks.add_task(_do_download, model_type, pretrained, hf_token)
    return {"success": True, "started": True}


@router.get("/api/models/download-status/{model_id}")
async def get_download_status(model_id: str):
    return settings._download_status.get(model_id, {"status": "idle", "progress": 0})
