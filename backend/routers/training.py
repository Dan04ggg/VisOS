import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response

from config import settings
from schemas.common import TrainingConfig

router = APIRouter()


def _resolve_base_model(model_arch: str, model_type: str) -> str:
    """Return the correct model identifier for the chosen arch + task type."""
    arch = model_arch.strip()
    if model_type == "rfdetr":
        return arch
    if model_type == "rtdetrv2":
        return arch
    if model_type == "rtdetr":
        base = arch.replace(".pt", "")
        filename = f"{base}.pt"
    else:
        base = arch.replace("-seg", "").replace("-cls", "").replace(".pt", "")
        if model_type == "segmentation":
            filename = f"{base}-seg.pt"
        elif model_type == "classification":
            filename = f"{base}-cls.pt"
        else:
            filename = f"{base}.pt"
    local_path = settings.MODELS_DIR / filename
    if local_path.exists():
        return str(local_path)
    return filename


@router.post("/api/install-cuda-torch")
async def install_cuda_torch():
    import sys, subprocess
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--upgrade",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return {"success": True, "message": "CUDA PyTorch installed. Restart the backend server to use GPU."}
        else:
            return {"success": False, "message": result.stderr[-2000:] or result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Install timed out after 5 minutes."}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/api/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    if config.dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[config.dataset_id]
    dataset_path = settings.DATASETS_DIR / config.dataset_id

    ds_format    = (dataset.get("format") or "").lower()
    ds_task_type = (dataset.get("task_type") or "detection").lower()
    mt = config.model_type

    CLASSIFICATION_FORMATS = {"classification-folder", "classification_folder"}
    SEGMENTATION_FORMATS   = {"cityscapes", "ade20k", "yolo-seg"}

    is_cls_dataset = ds_format in CLASSIFICATION_FORMATS or ds_task_type == "classification"
    is_seg_dataset = ds_format in SEGMENTATION_FORMATS or ds_task_type == "segmentation"
    is_det_dataset = not is_cls_dataset and not is_seg_dataset

    detection_model = mt in ("yolo", "rtdetr", "rfdetr")
    seg_model       = mt == "segmentation"
    cls_model       = mt == "classification"

    if is_cls_dataset and not cls_model:
        raise HTTPException(
            status_code=422,
            detail="This dataset is a classification dataset. Please select a Classification model type.",
        )
    if is_seg_dataset and cls_model:
        raise HTTPException(
            status_code=422,
            detail="Segmentation datasets cannot be used to train a classification model. "
                   "Please select an Object Detection or Segmentation model type.",
        )
    if is_det_dataset and seg_model:
        raise HTTPException(
            status_code=422,
            detail="This dataset does not contain polygon segmentation masks. "
                   "Please select an Object Detection model type, or convert your dataset "
                   "to a segmentation format first.",
        )
    if is_det_dataset and cls_model:
        raise HTTPException(
            status_code=422,
            detail="Detection datasets cannot be used for classification training. "
                   "Please use a dataset in classification-folder format.",
        )

    training_id = settings.training_manager.start_training(
        dataset_path,
        dataset["format"],
        config.model_type,
        {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "pretrained": config.pretrained,
            "device": config.device,
            "save_period": config.save_period,
            "base_model": _resolve_base_model(config.model_arch, config.model_type),
            "models_dir": str(settings.MODELS_DIR),
            "lr0": config.lr0,
            "lrf": config.lrf,
            "optimizer": config.optimizer,
            "patience": config.patience,
            "cos_lr": config.cos_lr,
            "warmup_epochs": config.warmup_epochs,
            "weight_decay": config.weight_decay,
            "mosaic": config.mosaic,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "amp": config.amp,
            "dropout": config.dropout,
        },
        name=config.name,
    )

    return {"success": True, "training_id": training_id}


@router.get("/api/train/jobs")
async def list_training_jobs():
    return {"jobs": settings.training_manager.list_training_jobs()}


@router.get("/api/train/{training_id}/status")
async def get_training_status(training_id: str):
    status = settings.training_manager.get_status(training_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training not found")
    return status


@router.post("/api/train/{training_id}/stop")
async def stop_training(training_id: str):
    success = settings.training_manager.stop_training(training_id)
    return {"success": success}


@router.post("/api/train/{training_id}/pause")
async def pause_training(training_id: str):
    success = settings.training_manager.pause_training(training_id)
    if not success:
        raise HTTPException(status_code=400, detail="Training not running or not found")
    return {"success": True, "message": "Pause requested — will stop after current epoch."}


@router.post("/api/train/{training_id}/resume")
async def resume_training(training_id: str):
    result = settings.training_manager.resume_training(training_id)
    if not result:
        raise HTTPException(status_code=400, detail="Cannot resume — not paused or no checkpoint found")
    return {"success": True, "training_id": result}


@router.post("/api/train/{training_id}/export-format")
async def export_model_format(training_id: str, format: str = "onnx"):
    export_path = settings.training_manager.export_model_format(training_id, format)
    if not export_path:
        raise HTTPException(status_code=404, detail="Model not found or export failed")
    if not os.path.exists(export_path):
        raise HTTPException(status_code=500, detail="Export produced no output file")
    ext = os.path.splitext(export_path)[1] or f".{format}"
    return FileResponse(export_path, filename=f"model{ext}")


@router.get("/api/train/{training_id}/export")
async def export_trained_model(training_id: str):
    model_path = settings.training_manager.get_model_path(training_id)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=os.path.basename(model_path))


@router.delete("/api/train/{training_id}")
async def delete_training_job(training_id: str):
    success = settings.training_manager.delete_job(training_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete — job is running or not found")
    return {"success": True}


@router.get("/api/train/{training_id}/checkpoints")
async def list_checkpoints(training_id: str):
    checkpoints = settings.training_manager.list_checkpoints(training_id)
    return {"checkpoints": checkpoints}


@router.get("/api/train/{training_id}/checkpoint/{filename}")
async def download_checkpoint(training_id: str, filename: str):
    checkpoints = settings.training_manager.list_checkpoints(training_id)
    for cp in checkpoints:
        if cp["name"] == filename:
            if not os.path.exists(cp["path"]):
                raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")
            return FileResponse(cp["path"], filename=filename)
    raise HTTPException(status_code=404, detail="Checkpoint not found")


@router.get("/api/train/{training_id}/report")
async def download_training_report(training_id: str):
    import json as _json
    status = settings.training_manager.get_status(training_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training not found")

    job = settings.training_manager.training_jobs.get(training_id, {})
    report = {
        "training_id":   training_id,
        "status":        status["status"],
        "model_type":    job.get("model_type", ""),
        "config":        {k: v for k, v in job.get("config", {}).items() if k not in ("device",)},
        "started_at":    status["started_at"],
        "device_info":   status["device_info"],
        "total_epochs":  status["total_epochs"],
        "current_epoch": status["current_epoch"],
        "model_path":    status["model_path"],
        "final_metrics": status["metrics"],
        "epoch_history": status["epoch_history"],
    }
    content = _json.dumps(report, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_report_{training_id}.json"'},
    )
