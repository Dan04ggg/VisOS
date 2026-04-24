import random
import shutil
import threading
import uuid
from fastapi import APIRouter, HTTPException

from config import settings
from core.workspace import _make_dataset_folder, _save_dataset_metadata
from schemas.common import SimplePreviewRequest, SimpleAugmentRequest

router = APIRouter()


def _convert_frontend_augconfig(config: dict) -> dict:
    """Convert frontend AugmentationConfig (camelCase) to augmenter format."""
    augs = {}
    if config.get("horizontalFlip"):
        augs["flip_horizontal"] = {"enabled": True, "params": {}}
    if config.get("verticalFlip"):
        augs["flip_vertical"] = {"enabled": True, "params": {}}
    if config.get("rotate90"):
        augs["rotate"] = {"enabled": True, "params": {"angle_range": [90, 90]}}
    elif config.get("randomRotate"):
        limit = float(config.get("rotateLimit", 15))
        augs["rotate"] = {"enabled": True, "params": {"angle_range": [-limit, limit]}}
    if config.get("randomCrop"):
        crop_scale = config.get("cropScale", [0.8, 1.0])
        augs["crop"] = {"enabled": True, "params": {"crop_range": crop_scale}}
    if config.get("brightness"):
        limit = float(config.get("brightnessLimit", 0.2))
        augs["brightness"] = {"enabled": True, "params": {"factor_range": [1 - limit, 1 + limit]}}
    if config.get("contrast"):
        limit = float(config.get("contrastLimit", 0.2))
        augs["contrast"] = {"enabled": True, "params": {"factor_range": [1 - limit, 1 + limit]}}
    if config.get("saturation"):
        limit = float(config.get("saturationLimit", 0.2))
        augs["saturation"] = {"enabled": True, "params": {"factor_range": [1 - limit, 1 + limit]}}
    if config.get("hue"):
        limit = float(config.get("hueLimit", 0.1))
        augs["hue"] = {"enabled": True, "params": {"shift_range": [-limit * 360, limit * 360]}}
    if config.get("blur"):
        limit = float(config.get("blurLimit", 3))
        augs["blur"] = {"enabled": True, "params": {"radius_range": [0.5, limit]}}
    if config.get("noise"):
        var = float(config.get("noiseVar", 0.1))
        augs["noise"] = {"enabled": True, "params": {"variance": var}}
    if config.get("cutout"):
        size = int(config.get("cutoutSize", 32))
        augs["cutout"] = {"enabled": True, "params": {"num_holes": 2, "size_range": [0.05, max(0.05, size / 640)]}}
    return augs


@router.get("/api/augmentations/list")
async def list_augmentation_options():
    return {"augmentations": settings.augmenter.get_available_augmentations()}


@router.post("/api/augment/preview")
async def augment_preview(request: SimplePreviewRequest):
    import base64, io

    dataset_id = request.dataset_id
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = settings.DATASETS_DIR / dataset_id
    images = settings.augmenter._find_images(dataset_path)
    if not images:
        raise HTTPException(status_code=400, detail="No images found in dataset")

    augs = _convert_frontend_augconfig(request.config)
    if not augs:
        raise HTTPException(status_code=400, detail="No augmentations enabled")

    aug_list = [(name, cfg.get("params", {})) for name, cfg in augs.items()]
    previews = []
    num_previews = min(request.num_previews, 6)

    from PIL import Image as PILImage
    for i in range(num_previews):
        img_info = random.choice(images)
        src_path = dataset_path / img_info["path"]
        try:
            img = PILImage.open(src_path).convert("RGB")
            n = random.randint(1, min(3, len(aug_list)))
            selected = random.sample(aug_list, n)
            for aug_name, params in selected:
                img, _ = settings.augmenter._apply_single_augmentation(img, aug_name, params)
            img.thumbnail((320, 320), PILImage.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            labels = [a[0] for a in selected]
            previews.append({"data_url": f"data:image/jpeg;base64,{b64}", "augmentations": labels})
        except Exception as e:
            previews.append({"data_url": None, "augmentations": [], "error": str(e)})

    return {"previews": previews}


@router.post("/api/augment")
async def simple_augment(request: SimpleAugmentRequest):
    dataset_id = request.dataset_id
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    num_images = dataset.get("num_images", 0)

    if request.target_size and request.target_size > 0:
        target_size = request.target_size
    elif request.augment_factor and request.augment_factor > 0:
        target_size = int(num_images * request.augment_factor)
    else:
        target_size = num_images * 2

    augs = _convert_frontend_augconfig(request.config)
    if not augs:
        raise HTTPException(status_code=400, detail="No augmentations enabled")

    output_name = request.output_name or f"{dataset['name']}_augmented"
    new_dataset_id, output_path = _make_dataset_folder(output_name)

    job_id = str(uuid.uuid4())[:8]
    settings._augmentation_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "generated": 0,
        "new_dataset_id": None,
        "new_dataset": None,
        "error": None,
    }

    def _run_augmentation():
        def _progress(pct: int, count: int):
            settings._augmentation_jobs[job_id]["progress"] = pct
            settings._augmentation_jobs[job_id]["generated"] = count

        result = settings.augmenter.augment_dataset(
            dataset_path, output_path, dataset["format"], target_size, augs,
            preserve_originals=True, progress_callback=_progress,
        )

        if not result.get("success"):
            shutil.rmtree(output_path, ignore_errors=True)
            settings._augmentation_jobs[job_id]["status"] = "error"
            settings._augmentation_jobs[job_id]["error"] = result.get("error", "Augmentation failed")
            return

        new_info = settings.dataset_parser.parse_dataset(output_path, dataset["format"], output_name)
        new_info["id"] = new_dataset_id
        settings.active_datasets[new_dataset_id] = new_info
        _save_dataset_metadata(new_dataset_id, new_info)

        settings._augmentation_jobs[job_id]["status"] = "done"
        settings._augmentation_jobs[job_id]["progress"] = 100
        settings._augmentation_jobs[job_id]["generated"] = result["augmented_images"]
        settings._augmentation_jobs[job_id]["new_dataset_id"] = new_dataset_id
        settings._augmentation_jobs[job_id]["new_dataset"] = new_info
        settings._augmentation_jobs[job_id]["total_images"] = result["total_images"]
        settings._augmentation_jobs[job_id]["original_images"] = result["original_images"]
        settings._augmentation_jobs[job_id]["augmented_images"] = result["augmented_images"]

    threading.Thread(target=_run_augmentation, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/augment/{job_id}/status")
async def get_augmentation_status(job_id: str):
    if job_id not in settings._augmentation_jobs:
        raise HTTPException(status_code=404, detail="Augmentation job not found")
    return settings._augmentation_jobs[job_id]
