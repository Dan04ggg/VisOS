import os
import json
import shutil
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

from config import settings
from core.workspace import _save_dataset_metadata, _make_dataset_folder
from schemas.common import (
    SplitRequest, ClassExtractRequest, ClassDeleteRequest, ClassMergeRequest,
    ClassRenameRequest, ClassAddRequest, AnnotationUpdate, AugmentationConfig,
    EnhancedSplitRequest, EnhancedAugmentationRequest, LocalFolderRequest,
    LocalPathRequest, SortingAction, YamlWizardConfig, BatchDeleteRequest,
    BatchSplitRequest, DuplicateDetectionRequest, ClipRegroupRequest,
    RemoveDuplicatesRequest, MergeRequest, DatasetRenameRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Dataset loading ───────────────────────────────────────────────────────────

@router.post("/api/datasets/load")
async def load_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: str = None,
    format_hint: str = None
):
    resolved_name = dataset_name or (files[0].filename.rsplit('.', 1)[0] if files else "dataset")
    dataset_id, dataset_path = _make_dataset_folder(resolved_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        for file in files:
            file_path = dataset_path / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            if file.filename.endswith(".zip"):
                shutil.unpack_archive(file_path, dataset_path)
                os.remove(file_path)

        dataset_info = settings.dataset_parser.parse_dataset(
            dataset_path, format_hint=format_hint, name=dataset_id
        )
        dataset_info["id"] = dataset_id
        settings.active_datasets[dataset_id] = dataset_info
        _save_dataset_metadata(dataset_id, dataset_info)
        return {"success": True, "dataset": dataset_info}
    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/datasets")
async def list_datasets():
    for dataset_id, dataset in list(settings.active_datasets.items()):
        if dataset.get("num_images", 0) == 0:
            try:
                dataset_path = settings.DATASETS_DIR / dataset_id
                fresh = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
                fresh["id"] = dataset_id
                settings.active_datasets[dataset_id] = fresh
                _save_dataset_metadata(dataset_id, fresh)
            except Exception as exc:
                logger.warning("Could not refresh stale dataset %s: %s", dataset_id, exc)
    return {"datasets": list(settings.active_datasets.values())}


@router.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    details = settings.dataset_parser.get_dataset_details(dataset_path, dataset["format"])
    return {"dataset": dataset, "details": details}


@router.get("/api/datasets/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: str, force_refresh: bool = False):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    cached = dataset.get("_cached_stats")
    if cached and not force_refresh:
        return {
            "dataset_id": dataset_id,
            "name": dataset["name"],
            "format": dataset["format"],
            "task_type": dataset.get("task_type", "detection"),
            "total_images": dataset.get("num_images", 0),
            "total_annotations": dataset.get("num_annotations", 0),
            "class_distribution": cached.get("class_distribution", {}),
            "splits": cached.get("splits", {}),
            "image_sizes": {},
            "avg_annotations_per_image": cached.get("avg_annotations_per_image", 0),
            "created_at": dataset["created_at"],
            "from_cache": True,
        }

    stats = settings.dataset_parser.get_dataset_details(dataset_path, dataset["format"])

    if dataset.get("num_images", 0) == 0 and stats.get("total_images", 0) > 0:
        fresh = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
        fresh["id"] = dataset_id
        settings.active_datasets[dataset_id] = fresh
        dataset = fresh

    cached_stats = {
        "class_distribution": stats.get("class_distribution", {}),
        "splits": stats.get("splits", {}),
        "avg_annotations_per_image": stats.get("avg_annotations_per_image", 0),
    }
    dataset["_cached_stats"] = cached_stats
    dataset["num_images"] = stats.get("total_images", dataset.get("num_images", 0))
    dataset["num_annotations"] = stats.get("total_annotations", dataset.get("num_annotations", 0))
    if stats.get("class_distribution"):
        dataset["classes"] = list(stats["class_distribution"].keys())
    settings.active_datasets[dataset_id] = dataset
    _save_dataset_metadata(dataset_id, dataset)

    return {
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "format": dataset["format"],
        "task_type": dataset.get("task_type", stats.get("task_type", "detection")),
        "total_images": stats.get("total_images", dataset["num_images"]),
        "total_annotations": stats.get("total_annotations", dataset["num_annotations"]),
        "class_distribution": stats.get("class_distribution", {}),
        "splits": stats.get("splits", {}),
        "image_sizes": {},
        "avg_annotations_per_image": stats.get("avg_annotations_per_image", 0),
        "created_at": dataset["created_at"],
        "from_cache": False,
    }


@router.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset_path = settings.DATASETS_DIR / dataset_id
    shutil.rmtree(dataset_path, ignore_errors=True)
    del settings.active_datasets[dataset_id]
    return {"success": True}


@router.put("/api/datasets/{dataset_id}/rename")
async def rename_dataset(dataset_id: str, request: DatasetRenameRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    new_name = request.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    dataset = settings.active_datasets[dataset_id]
    dataset["name"] = new_name
    settings.active_datasets[dataset_id] = dataset
    _save_dataset_metadata(dataset_id, dataset)
    return {"success": True, "dataset": dataset}


# ── Dataset splitting ─────────────────────────────────────────────────────────

@router.post("/api/datasets/{dataset_id}/split")
async def split_dataset(dataset_id: str, request: SplitRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    output_name = request.output_name or f"{dataset['name']}_split"
    new_dataset_id, output_path = _make_dataset_folder(output_name)

    split_result = settings.dataset_parser.create_split_dataset(
        dataset_path, output_path, dataset["format"],
        train_ratio=request.train_ratio, val_ratio=request.val_ratio,
        test_ratio=request.test_ratio, shuffle=request.shuffle, seed=request.seed,
    )

    new_info = settings.dataset_parser.parse_dataset(output_path, dataset["format"], output_name)
    new_info["id"] = new_dataset_id
    new_info["splits"] = split_result["splits"]
    settings.active_datasets[new_dataset_id] = new_info
    _save_dataset_metadata(new_dataset_id, new_info)

    return {
        "success": True,
        "new_dataset": new_info,
        "splits": split_result["splits"],
        "train_count": split_result["splits"].get("train", 0),
        "val_count": split_result["splits"].get("val", 0),
        "test_count": split_result["splits"].get("test", 0),
    }


@router.post("/api/datasets/{dataset_id}/split-enhanced")
async def split_dataset_enhanced(dataset_id: str, request: EnhancedSplitRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    output_name = request.output_name or f"{dataset['name']}_split"
    new_dataset_id, output_path = _make_dataset_folder(output_name)

    split_result = settings.dataset_parser.create_split_dataset(
        dataset_path, output_path, dataset["format"],
        train_ratio=request.train_ratio, val_ratio=request.val_ratio,
        test_ratio=request.test_ratio, shuffle=request.shuffle, seed=request.seed,
        stratified=request.stratified,
    )

    new_info = settings.dataset_parser.parse_dataset(output_path, dataset["format"], new_dataset_id)
    new_info["id"] = new_dataset_id
    new_info["splits"] = split_result["splits"]
    settings.active_datasets[new_dataset_id] = new_info

    return {
        "success": True,
        "new_dataset": new_info,
        "splits": split_result["splits"],
        "class_distribution": split_result.get("class_distribution", {}),
    }


# ── Class management ──────────────────────────────────────────────────────────

@router.post("/api/datasets/{dataset_id}/extract-classes")
async def extract_classes_to_new_dataset(dataset_id: str, request: ClassExtractRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    source_format = dataset["format"]
    new_dataset_id, output_path = _make_dataset_folder(
        request.output_name or f"{dataset['name']}_extracted"
    )

    extraction_result = settings.annotation_manager.extract_classes(
        dataset_path, output_path, source_format, request.classes_to_extract
    )

    target_format = request.output_format or source_format
    if request.output_format and request.output_format != source_format:
        converted_id = str(uuid.uuid4())
        converted_path = settings.DATASETS_DIR / converted_id
        try:
            settings.format_converter.convert(output_path, converted_path, source_format, request.output_format)
            shutil.rmtree(output_path, ignore_errors=True)
            converted_path.rename(output_path)
        except Exception as exc:
            shutil.rmtree(converted_path, ignore_errors=True)
            logger.warning(
                "Format conversion from %s to %s failed (falling back to %s): %s",
                source_format, request.output_format, source_format, exc,
            )
            target_format = source_format

    new_info = settings.dataset_parser.parse_dataset(output_path, target_format, request.output_name)
    new_info["id"] = new_dataset_id
    settings.active_datasets[new_dataset_id] = new_info
    _save_dataset_metadata(new_dataset_id, new_info)

    return {
        "success": True,
        "new_dataset": new_info,
        "extracted_images": extraction_result["extracted_images"],
        "extracted_annotations": extraction_result["extracted_annotations"],
    }


@router.post("/api/datasets/{dataset_id}/delete-classes")
async def delete_classes_from_dataset(dataset_id: str, request: ClassDeleteRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    delete_result = settings.annotation_manager.delete_classes(
        dataset_path, dataset["format"], request.classes_to_delete
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    return {
        "success": True,
        "updated_dataset": dataset_info,
        "deleted_annotations": delete_result["deleted_annotations"],
        "affected_images": delete_result["affected_images"],
    }


@router.get("/api/datasets/{dataset_id}/unannotated-count")
async def get_unannotated_count(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    fmt = dataset["format"]

    all_images = settings.dataset_parser.get_images_with_annotations(dataset_path, fmt, page=1, limit=999999)
    total = len(all_images)
    unannotated = sum(1 for img in all_images if not img.get("annotations"))
    return {"total": total, "unannotated": unannotated, "annotated": total - unannotated}


@router.post("/api/datasets/{dataset_id}/remove-unannotated")
async def remove_unannotated_images(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    root = settings.dataset_parser._find_dataset_root(dataset_path)
    fmt = dataset["format"]
    removed = 0

    all_images = settings.dataset_parser.get_images_with_annotations(dataset_path, fmt, page=1, limit=999999)
    for img in all_images:
        if img.get("annotations"):
            continue
        img_path = root / img["path"]
        if not img_path.exists():
            img_path = dataset_path / img["path"]
        if not img_path.exists():
            matches = list(root.rglob(img["filename"]))
            if matches:
                img_path = matches[0]
        if img_path.exists():
            img_path.unlink()
            removed += 1
        if fmt in settings.annotation_manager.YOLO_FORMATS:
            for lbl_root in root.rglob("labels"):
                if lbl_root.is_dir():
                    lf = lbl_root / f"{img['id']}.txt"
                    if lf.exists():
                        lf.unlink()

    if dataset_id in settings._images_cache:
        del settings._images_cache[dataset_id]
    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    remaining = dataset_info.get("num_images", 0)
    return {"success": True, "removed": removed, "remaining": remaining, "updated_dataset": dataset_info}


@router.post("/api/datasets/{dataset_id}/merge-classes")
async def merge_classes_in_dataset(dataset_id: str, request: ClassMergeRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    merge_result = settings.annotation_manager.merge_classes(
        dataset_path, dataset["format"], request.source_classes, request.target_class
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    return {
        "success": True,
        "updated_dataset": dataset_info,
        "merged_annotations": merge_result["merged_annotations"],
    }


@router.post("/api/datasets/{dataset_id}/rename-class")
async def rename_class_in_dataset(dataset_id: str, request: ClassRenameRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if not request.old_name or not request.new_name:
        raise HTTPException(status_code=400, detail="Both old_name and new_name are required")
    if request.old_name == request.new_name:
        raise HTTPException(status_code=400, detail="New name must differ from old name")

    rename_result = settings.annotation_manager.rename_class(
        dataset_path, dataset["format"], request.old_name, request.new_name
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    return {
        "success": True,
        "updated_dataset": dataset_info,
        "renamed_annotations": rename_result["renamed_annotations"],
    }


@router.post("/api/datasets/{dataset_id}/add-classes")
async def add_classes_to_dataset(dataset_id: str, request: ClassAddRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if request.use_model and request.model_id:
        results = settings.model_manager.annotate_with_new_classes(
            request.model_id, dataset_path, dataset["format"], request.new_classes
        )
    else:
        settings.annotation_manager.add_classes(dataset_path, dataset["format"], request.new_classes)
        results = {"added_classes": request.new_classes}

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {"success": True, "results": results, "updated_dataset": dataset_info}


@router.get("/api/datasets/{dataset_id}/classes")
async def get_dataset_classes(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        dataset_path = settings.DATASETS_DIR / dataset_id
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        try:
            meta_path = dataset_path / settings.METADATA_FILENAME
            if meta_path.exists():
                with open(meta_path) as f:
                    info = json.load(f)
                info["id"] = dataset_id
            else:
                info = settings.dataset_parser.parse_dataset(dataset_path, name=dataset_path.name)
                info["id"] = dataset_id
                _save_dataset_metadata(dataset_id, info)
            settings.active_datasets[dataset_id] = info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not restore dataset: {e}")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    classes_list = settings.dataset_parser.get_classes_with_distribution(dataset_path, dataset["format"])
    classes_dict = {item["name"]: item["count"] for item in classes_list}
    return {"classes": classes_dict}


# ── Images and annotations ────────────────────────────────────────────────────

@router.get("/api/datasets/{dataset_id}/images")
async def get_dataset_images(
    dataset_id: str,
    page: int = 1,
    limit: int = 50,
    split: str = None,
    class_name: str = None,
    bust_cache: bool = False,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if bust_cache or dataset_id not in settings._images_cache:
        all_images = settings.dataset_parser.get_images_with_annotations(
            dataset_path, dataset["format"], page=1, limit=999999
        )
        settings._images_cache[dataset_id] = all_images
    else:
        all_images = settings._images_cache[dataset_id]

    if split and split != "all":
        all_images = [img for img in all_images if img.get("split") == split]

    if class_name and class_name != "all":
        filtered = []
        for img in all_images:
            if img.get("class_name") == class_name:
                filtered.append(img)
            elif any(ann.get("class_name") == class_name for ann in img.get("annotations", [])):
                filtered.append(img)
        all_images = filtered

    if limit >= 999999:
        images = all_images
    else:
        start = (page - 1) * limit
        images = all_images[start:start + limit]

    return {"images": images, "total": len(all_images), "page": page, "limit": limit}


@router.get("/api/datasets/{dataset_id}/image/{image_id}")
async def get_image_with_annotations(dataset_id: str, image_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    return settings.dataset_parser.get_image_data(dataset_path, dataset["format"], image_id)


@router.put("/api/datasets/{dataset_id}/image/{image_id}/annotations")
async def update_annotations(dataset_id: str, image_id: str, update: AnnotationUpdate):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if dataset.get("format") == "generic-images":
        dataset["format"] = "yolo"
        meta_path = dataset_path / "dataset_metadata.json"
        try:
            import json as _json2
            meta = _json2.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta["format"] = "yolo"
            meta_path.write_text(_json2.dumps(meta, indent=2))
        except Exception:
            pass
        _img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        _images_dir = settings.dataset_parser._find_dataset_root(dataset_path) / "images"
        _images_dir.mkdir(exist_ok=True)
        _root = settings.dataset_parser._find_dataset_root(dataset_path)
        for _f in list(_root.iterdir()):
            if _f.is_file() and _f.suffix.lower() in _img_exts:
                shutil.move(str(_f), _images_dir / _f.name)
        if dataset_id in settings._images_cache:
            del settings._images_cache[dataset_id]

    settings.annotation_manager.update_annotations(
        dataset_path, dataset["format"], image_id, update.annotations
    )

    if dataset_id in settings._images_cache:
        cache = settings._images_cache[dataset_id]
        for img in cache:
            if img.get("id") == image_id:
                img["annotations"] = [a.dict() if hasattr(a, "dict") else a for a in update.annotations]
                img["has_annotations"] = len(update.annotations) > 0
                break

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {"success": True}


@router.post("/api/datasets/{dataset_id}/add-images")
async def add_images_to_dataset(dataset_id: str, files: List[UploadFile] = File(...)):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    added_images = []

    for file in files:
        if file.content_type.startswith("image/"):
            image_path = settings.annotation_manager.add_image(dataset_path, dataset["format"], file)
            added_images.append(image_path)

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {"success": True, "added_images": added_images}


# ── Annotation history ────────────────────────────────────────────────────────

@router.get("/api/datasets/{dataset_id}/history")
async def get_annotation_history(dataset_id: str, image_id: str = None):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    history_key = f"{dataset_id}_{image_id}" if image_id else dataset_id
    return {"history": settings.annotation_history.get(history_key, [])}


@router.post("/api/datasets/{dataset_id}/history/undo")
async def undo_annotation(dataset_id: str, image_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    history_key = f"{dataset_id}_{image_id}"
    history = settings.annotation_history.get(history_key, [])
    if not history:
        raise HTTPException(status_code=400, detail="No history to undo")

    last_entry = history.pop()
    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if last_entry.get("previous_annotations"):
        settings.annotation_manager.restore_annotations(
            dataset_path, dataset["format"], image_id, last_entry["previous_annotations"]
        )

    settings.annotation_history[history_key] = history
    return {"success": True, "restored_to": last_entry.get("previous_annotations")}


@router.post("/api/datasets/{dataset_id}/history/record")
async def record_annotation_action(
    dataset_id: str,
    image_id: str,
    action: str,
    previous_annotations: List[dict],
    new_annotations: List[dict],
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    history_key = f"{dataset_id}_{image_id}"
    if history_key not in settings.annotation_history:
        settings.annotation_history[history_key] = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "previous_annotations": previous_annotations,
        "new_annotations": new_annotations,
    }
    settings.annotation_history[history_key].append(entry)
    if len(settings.annotation_history[history_key]) > 50:
        settings.annotation_history[history_key] = settings.annotation_history[history_key][-50:]

    return {"success": True}


# ── Dataset sorting ───────────────────────────────────────────────────────────

@router.post("/api/sorting/start/{dataset_id}")
async def start_sorting_session(dataset_id: str, filter_classes: List[str] = Query(default=None)):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    images = settings.dataset_parser.get_images_with_annotations(
        dataset_path, dataset["format"], filter_classes=filter_classes
    )

    session_id = str(uuid.uuid4())
    settings.sorting_sessions[session_id] = {
        "dataset_id": dataset_id,
        "images": images,
        "current_index": 0,
        "kept": [],
        "deleted": [],
        "total": len(images),
        "filter_classes": filter_classes,
    }

    return {
        "session_id": session_id,
        "total_images": len(images),
        "current_image": images[0] if images else None,
    }


@router.get("/api/sorting/{session_id}/current")
async def get_current_sorting_image(session_id: str):
    if session_id not in settings.sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = settings.sorting_sessions[session_id]
    idx = session["current_index"]

    if idx >= len(session["images"]):
        return {"complete": True, "kept": len(session["kept"]), "deleted": len(session["deleted"])}

    return {
        "complete": False,
        "current_index": idx,
        "total": session["total"],
        "image": session["images"][idx],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - idx,
        },
    }


@router.post("/api/sorting/{session_id}/action")
async def sorting_action(session_id: str, action: SortingAction):
    if session_id not in settings.sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = settings.sorting_sessions[session_id]
    idx = session["current_index"]

    if idx >= len(session["images"]):
        return {"complete": True}

    image = session["images"][idx]
    if action.action == "keep":
        session["kept"].append(image)
    elif action.action == "delete":
        session["deleted"].append(image)

    session["current_index"] += 1

    if session["current_index"] >= len(session["images"]):
        return {"complete": True, "kept": len(session["kept"]), "deleted": len(session["deleted"])}

    return {
        "complete": False,
        "current_index": session["current_index"],
        "image": session["images"][session["current_index"]],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - session["current_index"],
        },
    }


@router.post("/api/sorting/{session_id}/go-back")
async def go_back_in_sorting(session_id: str):
    if session_id not in settings.sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = settings.sorting_sessions[session_id]
    if session["current_index"] <= 0:
        raise HTTPException(status_code=400, detail="Already at first image")

    session["current_index"] -= 1
    current_image_id = session["images"][session["current_index"]]["id"]
    session["kept"] = [img for img in session["kept"] if img["id"] != current_image_id]
    session["deleted"] = [img for img in session["deleted"] if img["id"] != current_image_id]

    return {
        "complete": False,
        "current_index": session["current_index"],
        "image": session["images"][session["current_index"]],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - session["current_index"],
        },
    }


@router.post("/api/sorting/{session_id}/finalize")
async def finalize_sorting(session_id: str, create_new_dataset: bool = True):
    if session_id not in settings.sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = settings.sorting_sessions[session_id]
    dataset_id = session["dataset_id"]

    if create_new_dataset:
        filter_name = f"{settings.active_datasets[dataset_id]['name']}_filtered"
        new_dataset_id, new_dataset_path = _make_dataset_folder(filter_name)
        original_path = settings.DATASETS_DIR / dataset_id
        original_format = settings.active_datasets[dataset_id]["format"]

        settings.dataset_parser.create_filtered_dataset(
            original_path, new_dataset_path, session["kept"], original_format
        )

        new_info = settings.dataset_parser.parse_dataset(
            new_dataset_path, format_hint=original_format, name=new_dataset_id
        )
        new_info["id"] = new_dataset_id
        settings.active_datasets[new_dataset_id] = new_info
        del settings.sorting_sessions[session_id]
        return {"success": True, "new_dataset": new_info}

    del settings.sorting_sessions[session_id]
    return {"success": True}


# ── Augmentation ──────────────────────────────────────────────────────────────

@router.get("/api/augmentations")
async def list_available_augmentations():
    return {
        "augmentations": [
            {"id": "flip_horizontal", "name": "Horizontal Flip", "description": "Flip images horizontally", "params": {}},
            {"id": "flip_vertical", "name": "Vertical Flip", "description": "Flip images vertically", "params": {}},
            {"id": "rotate", "name": "Rotation", "description": "Rotate images by a random angle", "params": {"angle": {"type": "range", "min": -45, "max": 45, "default": 15}}},
            {"id": "brightness", "name": "Brightness", "description": "Adjust image brightness", "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}},
            {"id": "contrast", "name": "Contrast", "description": "Adjust image contrast", "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}},
            {"id": "saturation", "name": "Saturation", "description": "Adjust color saturation", "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}},
            {"id": "hue", "name": "Hue Shift", "description": "Shift image hue", "params": {"factor": {"type": "range", "min": -0.1, "max": 0.1, "default": 0.05}}},
            {"id": "blur", "name": "Gaussian Blur", "description": "Apply Gaussian blur", "params": {"kernel": {"type": "range", "min": 3, "max": 11, "default": 5}}},
            {"id": "noise", "name": "Gaussian Noise", "description": "Add random noise", "params": {"variance": {"type": "range", "min": 0.01, "max": 0.1, "default": 0.02}}},
            {"id": "crop", "name": "Random Crop", "description": "Randomly crop a portion of the image", "params": {"scale": {"type": "range", "min": 0.7, "max": 0.95, "default": 0.85}}},
            {"id": "scale", "name": "Random Scale", "description": "Randomly scale the image", "params": {"scale": {"type": "range", "min": 0.8, "max": 1.2, "default": 0.1}}},
            {"id": "shear", "name": "Shear", "description": "Apply shear transformation", "params": {"angle": {"type": "range", "min": -15, "max": 15, "default": 10}}},
            {"id": "mosaic", "name": "Mosaic", "description": "Combine 4 images into one", "params": {}},
            {"id": "mixup", "name": "MixUp", "description": "Blend two images together", "params": {"alpha": {"type": "range", "min": 0.1, "max": 0.5, "default": 0.3}}},
            {"id": "cutout", "name": "Cutout", "description": "Randomly cut out rectangles", "params": {"num_holes": {"type": "range", "min": 1, "max": 5, "default": 2}, "size": {"type": "range", "min": 10, "max": 50, "default": 30}}},
            {"id": "grayscale", "name": "Grayscale", "description": "Convert to grayscale", "params": {"probability": {"type": "range", "min": 0, "max": 1, "default": 0.1}}},
            {"id": "elastic", "name": "Elastic Deformation", "description": "Apply elastic transformation", "params": {"alpha": {"type": "range", "min": 50, "max": 200, "default": 100}, "sigma": {"type": "range", "min": 5, "max": 15, "default": 10}}},
        ]
    }


@router.post("/api/datasets/{dataset_id}/augment")
async def augment_dataset(dataset_id: str, config: AugmentationConfig):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    new_dataset_id, output_path = _make_dataset_folder(config.output_name or f"{dataset['name']}_augmented")

    augmentation_result = settings.annotation_manager.augment_dataset(
        dataset_path, output_path, dataset["format"], config.target_size, config.augmentations
    )

    new_info = settings.dataset_parser.parse_dataset(output_path, dataset["format"], new_dataset_id)
    new_info["id"] = new_dataset_id
    settings.active_datasets[new_dataset_id] = new_info

    return {
        "success": True,
        "new_dataset": new_info,
        "augmented_images": augmentation_result["augmented_images"],
        "original_images": augmentation_result["original_images"],
    }


@router.post("/api/datasets/{dataset_id}/augment-enhanced")
async def augment_dataset_enhanced(dataset_id: str, request: EnhancedAugmentationRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    target_size = request.target_size
    if request.target_multiplier:
        target_size = int(dataset["num_images"] * request.target_multiplier)

    new_dataset_id, output_path = _make_dataset_folder(request.output_name or f"{dataset['name']}_augmented")

    result = settings.augmenter.augment_dataset(
        dataset_path, output_path, dataset["format"],
        target_size, request.augmentations, request.preserve_originals,
    )

    if not result["success"]:
        shutil.rmtree(output_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=result.get("error", "Augmentation failed"))

    new_info = settings.dataset_parser.parse_dataset(output_path, dataset["format"], new_dataset_id)
    new_info["id"] = new_dataset_id
    settings.active_datasets[new_dataset_id] = new_info

    return {
        "success": True,
        "new_dataset": new_info,
        "original_images": result["original_images"],
        "augmented_images": result["augmented_images"],
        "total_images": result["total_images"],
    }


# ── Duplicate detection ───────────────────────────────────────────────────────

@router.post("/api/datasets/{dataset_id}/find-duplicates")
async def find_duplicate_images(dataset_id: str, request: DuplicateDetectionRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = settings.DATASETS_DIR / dataset_id

    if request.method == "clip":
        result = settings.clip_manager.find_similar_images(
            dataset_path,
            similarity_threshold=request.threshold / 100.0,
            dataset_id=dataset_id,
        )
        if result.get("success"):
            result["duplicate_groups"] = result.pop("similar_groups", 0)
            result["total_duplicates"] = result.pop("total_similar", 0)
            result["unique_images"] = result.get("total_images", 0) - result["total_duplicates"]
            result["method"] = "clip"
            result["threshold"] = request.threshold
    else:
        result = settings.duplicate_detector.find_duplicates(
            dataset_path,
            method=request.method,
            threshold=request.threshold,
            include_near_duplicates=request.include_near_duplicates,
        )
    return result


@router.post("/api/datasets/{dataset_id}/cancel-scan")
async def cancel_duplicate_scan(dataset_id: str):
    settings.clip_manager.cancel_scan(dataset_id)
    return {"success": True}


@router.post("/api/datasets/{dataset_id}/clip-regroup")
async def clip_regroup_images(dataset_id: str, request: ClipRegroupRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = settings.DATASETS_DIR / dataset_id
    result = settings.clip_manager.regroup_by_threshold(
        dataset_path, similarity_threshold=request.threshold / 100.0
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Regroup failed"))

    result["duplicate_groups"] = result.pop("similar_groups", 0)
    result["total_duplicates"] = result.pop("total_similar", 0)
    result["unique_images"] = result.get("total_images", 0) - result["total_duplicates"]
    result["method"] = "clip"
    result["threshold"] = request.threshold
    return result


@router.post("/api/datasets/{dataset_id}/remove-duplicates")
async def remove_duplicate_images(dataset_id: str, request: RemoveDuplicatesRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    result = settings.duplicate_detector.remove_duplicates(
        dataset_path, dataset["format"], request.groups, request.keep_strategy
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {**result, "updated_dataset": dataset_info}


# ── Batch image operations ────────────────────────────────────────────────────

@router.post("/api/datasets/{dataset_id}/images/batch-delete")
async def batch_delete_images(dataset_id: str, request: BatchDeleteRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    ids_to_delete = set(request.image_ids)

    all_images = settings._images_cache.get(dataset_id) or settings.dataset_parser.get_images_with_annotations(
        dataset_path, fmt, page=1, limit=999999
    )

    deleted = 0
    for img in all_images:
        if img["id"] not in ids_to_delete:
            continue
        img_path = dataset_path / img["path"]
        try:
            if img_path.exists():
                img_path.unlink()
        except Exception:
            pass
        if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11"):
            label_path = img_path.with_suffix(".txt")
            label_path2 = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
            for lp in (label_path, label_path2):
                try:
                    if lp.exists():
                        lp.unlink()
                except Exception:
                    pass
        elif fmt in ("pascal-voc", "voc"):
            ann_path = dataset_path / "Annotations" / (img_path.stem + ".xml")
            try:
                if ann_path.exists():
                    ann_path.unlink()
            except Exception:
                pass
        deleted += 1

    settings._images_cache.pop(dataset_id, None)
    try:
        dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
        dataset_info["id"] = dataset_id
        settings.active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True, "deleted": deleted}


@router.post("/api/datasets/{dataset_id}/images/batch-split")
async def batch_assign_split(dataset_id: str, request: BatchSplitRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    target_split = request.split.strip().lower()
    ids_to_move = set(request.image_ids)

    if not target_split:
        raise HTTPException(status_code=400, detail="split must not be empty")

    all_images = settings._images_cache.get(dataset_id) or settings.dataset_parser.get_images_with_annotations(
        dataset_path, fmt, page=1, limit=999999
    )

    moved = 0
    for img in all_images:
        if img["id"] not in ids_to_move:
            continue
        current_path = dataset_path / img["path"]
        if not current_path.exists():
            continue

        rel = Path(img["path"])
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in ("images", "train", "val", "valid", "test"):
            if parts[0] == "images" and len(parts) >= 3:
                new_rel = Path("images") / target_split / Path(*parts[2:])
                label_old = Path("labels") / parts[1] / Path(*parts[2:])
                label_new = Path("labels") / target_split / Path(*parts[2:])
            else:
                new_rel = Path(target_split) / Path(*parts[1:])
                label_old = None
                label_new = None
        else:
            continue

        new_path = dataset_path / new_rel
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil as _shutil
            _shutil.move(str(current_path), str(new_path))
        except Exception:
            continue

        if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11") and label_old and label_new:
            old_label_path = dataset_path / label_old.with_suffix(".txt")
            new_label_path = dataset_path / label_new.with_suffix(".txt")
            if old_label_path.exists():
                new_label_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import shutil as _shutil
                    _shutil.move(str(old_label_path), str(new_label_path))
                except Exception:
                    pass

        moved += 1

    settings._images_cache.pop(dataset_id, None)
    try:
        dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
        dataset_info["id"] = dataset_id
        settings.active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True, "moved": moved, "split": target_split}


# ── YAML Wizard ───────────────────────────────────────────────────────────────

@router.get("/api/datasets/{dataset_id}/class-samples")
async def get_class_samples(dataset_id: str, samples_per_class: int = 3):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    raw_path = settings.DATASETS_DIR / dataset_id
    root = settings.dataset_parser._find_dataset_root(raw_path)
    fmt = dataset["format"]
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    class_samples: dict = {}

    if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"):
        existing_names: list = []
        for yf in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            try:
                import yaml as _yaml
                with open(yf) as f:
                    cfg = _yaml.safe_load(f)
                if cfg and "names" in cfg:
                    n = cfg["names"]
                    existing_names = list(n.values()) if isinstance(n, dict) else list(n)
                break
            except Exception:
                pass

        for label_file in root.glob("**/labels/*.txt"):
            img_file = None
            img_stem = label_file.stem
            img_dir = label_file.parent.parent / "images"
            for ext in IMAGE_EXTS:
                cand = img_dir / (img_stem + ext)
                if cand.exists():
                    img_file = cand
                    break
            if img_file is None:
                continue
            try:
                annotations = []
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cid = int(parts[0])
                        if len(parts) == 5:
                            annotations.append({
                                "type": "bbox", "class_id": cid,
                                "x_center": float(parts[1]), "y_center": float(parts[2]),
                                "width": float(parts[3]), "height": float(parts[4]),
                                "normalized": True,
                            })
                        else:
                            annotations.append({
                                "type": "polygon", "class_id": cid,
                                "points": [float(p) for p in parts[1:]],
                                "normalized": True,
                            })
                if not annotations:
                    continue
                seen_cids = set(a["class_id"] for a in annotations)
                try:
                    rel_path = str(img_file.relative_to(root)).replace("\\", "/")
                except ValueError:
                    rel_path = str(img_file.relative_to(raw_path)).replace("\\", "/")
                for cid in seen_cids:
                    if cid not in class_samples:
                        class_samples[cid] = []
                    if len(class_samples[cid]) < samples_per_class:
                        class_samples[cid].append({
                            "image_id": img_stem,
                            "image_path": rel_path,
                            "annotations": annotations,
                        })
            except Exception:
                continue

    def _rel(p: Path) -> str:
        return "../" + str(p.relative_to(root.parent)).replace("\\", "/")

    splits: dict = {"train": None, "val": None, "test": None}
    for split, aliases in [("train", ["train"]), ("val", ["val", "valid", "validation"]), ("test", ["test"])]:
        for alias in aliases:
            for sub in ["images", ""]:
                cand = root / alias / sub if sub else root / alias
                if cand.exists() and cand.is_dir():
                    splits[split] = _rel(cand)
                    break
            if splits[split]:
                break

    return {
        "classes": [
            {
                "class_id": cid,
                "existing_name": existing_names[cid] if cid < len(existing_names) else None,
                "samples": class_samples.get(cid, []),
            }
            for cid in sorted(class_samples.keys())
        ],
        "splits": splits,
        "dataset_path": str(root),
        "existing_names": existing_names,
    }


@router.post("/api/datasets/{dataset_id}/generate-yaml")
async def generate_dataset_yaml(dataset_id: str, config: YamlWizardConfig):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_path = settings.DATASETS_DIR / dataset_id
    root = settings.dataset_parser._find_dataset_root(raw_path)

    import yaml as _yaml

    yaml_data: dict = {"path": str(root)}
    if config.train_path:
        yaml_data["train"] = config.train_path
    if config.val_path:
        yaml_data["val"] = config.val_path
    if config.test_path:
        yaml_data["test"] = config.test_path
    yaml_data["nc"] = len(config.class_names)
    yaml_data["names"] = {i: name for i, name in enumerate(config.class_names)}

    out_file = root / "data.yaml"
    with open(out_file, "w") as f:
        _yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    lines = [f"path: {yaml_data['path']}"]
    for k in ("train", "val", "test"):
        if k in yaml_data:
            lines.append(f"{k}: {yaml_data[k]}")
    lines += ["", f"nc: {yaml_data['nc']}", f"names: {config.class_names}"]
    preview = "\n".join(lines)

    dinfo = settings.dataset_parser.parse_dataset(
        raw_path,
        settings.active_datasets[dataset_id]["format"],
        settings.active_datasets[dataset_id]["name"],
    )
    dinfo["id"] = dataset_id
    settings.active_datasets[dataset_id] = dinfo

    return {"success": True, "yaml_path": str(out_file), "preview": preview}


# ── Local folder / browse ─────────────────────────────────────────────────────

@router.post("/api/datasets/load-local")
async def load_local_dataset(request: LocalFolderRequest):
    folder_path = Path(request.folder_path)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    dataset_id, dataset_link = _make_dataset_folder(request.dataset_name or folder_path.name)

    try:
        dataset_link.symlink_to(folder_path.absolute())
    except OSError:
        shutil.copytree(folder_path, dataset_link)

    try:
        dataset_info = settings.dataset_parser.parse_dataset(
            dataset_link, format_hint=request.format_hint, name=dataset_id
        )
        dataset_info["id"] = dataset_id
        dataset_info["local_path"] = str(folder_path.absolute())
        dataset_info["is_local"] = True
        settings.active_datasets[dataset_id] = dataset_info
        _save_dataset_metadata(dataset_id, dataset_info)
        return {"success": True, "dataset": dataset_info}
    except Exception as e:
        if dataset_link.is_symlink():
            dataset_link.unlink()
        else:
            shutil.rmtree(dataset_link, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/datasets/browse-folders")
async def browse_local_folders_dataset(path: str = "."):
    try:
        folder_path = Path(path).expanduser().absolute()
        if not folder_path.exists():
            return {"error": "Path does not exist", "items": []}
        items = []
        for item in sorted(folder_path.iterdir()):
            try:
                if item.is_dir() and not item.name.startswith('.'):
                    is_dataset = False
                    dataset_format = None
                    if list(item.glob("*.yaml")) or list(item.glob("labels/*.txt")):
                        is_dataset = True
                        dataset_format = "yolo"
                    elif list(item.glob("*.json")):
                        is_dataset = True
                        dataset_format = "coco/labelme"
                    elif list(item.glob("*.xml")):
                        is_dataset = True
                        dataset_format = "pascal-voc"
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dataset": is_dataset,
                        "format_hint": dataset_format,
                        "type": "directory",
                    })
            except PermissionError:
                continue
        return {
            "current_path": str(folder_path),
            "parent_path": str(folder_path.parent),
            "items": items,
        }
    except Exception as e:
        return {"error": str(e), "items": []}


@router.post("/api/browse-folders")
async def browse_local_folders_root(path: str = "."):
    try:
        folder_path = Path(path).expanduser().resolve()
        if not folder_path.exists():
            return {"error": "Path does not exist", "items": [], "current_path": str(folder_path)}
        items = []
        if folder_path.parent != folder_path:
            items.append({"name": "..", "path": str(folder_path.parent), "is_directory": True, "is_dataset": False})
        for item in sorted(folder_path.iterdir()):
            try:
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    is_dataset = False
                    format_hint = None
                    if list(item.glob("*.yaml")) or list(item.glob("data.yaml")):
                        is_dataset = True
                        format_hint = "yolo"
                    elif list(item.glob("*.json")):
                        is_dataset = True
                        format_hint = "coco/labelme"
                    elif list(item.glob("*.xml")):
                        is_dataset = True
                        format_hint = "pascal-voc"
                    elif any(d.is_dir() and list(d.glob("*.jpg")) or list(d.glob("*.png")) for d in item.iterdir() if d.is_dir()):
                        is_dataset = True
                        format_hint = "classification-folder"
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "is_directory": True,
                        "is_dataset": is_dataset,
                        "format_hint": format_hint,
                    })
            except PermissionError:
                continue
        return {
            "current_path": str(folder_path),
            "parent_path": str(folder_path.parent) if folder_path.parent != folder_path else None,
            "items": items,
        }
    except Exception as e:
        return {"error": str(e), "items": [], "current_path": path}


# ── Image serving ─────────────────────────────────────────────────────────────

@router.get("/api/image/{dataset_id}/{image_path:path}")
async def serve_image(dataset_id: str, image_path: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_root = settings.dataset_parser._find_dataset_root(settings.DATASETS_DIR / dataset_id)
    full_path = dataset_root / image_path
    if not full_path.exists():
        full_path = settings.DATASETS_DIR / dataset_id / image_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path, headers={"Access-Control-Allow-Origin": "*"})


@router.get("/api/datasets/{dataset_id}/image-file/{image_path:path}")
async def serve_dataset_image(dataset_id: str, image_path: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_root = settings.dataset_parser._find_dataset_root(settings.DATASETS_DIR / dataset_id)
    image_file = dataset_root / image_path

    if not image_file.exists():
        image_file = settings.DATASETS_DIR / dataset_id / image_path
    if not image_file.exists():
        filename = Path(image_path).name
        raw_dataset_dir = settings.DATASETS_DIR / dataset_id
        for ext_variant in [filename, filename.lower(), filename.upper()]:
            matches = list(raw_dataset_dir.rglob(ext_variant))
            image_matches = [m for m in matches if m.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}]
            if image_matches:
                image_file = image_matches[0]
                break
    if not image_file.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_file, headers={"Access-Control-Allow-Origin": "*"})


# ── Merge ─────────────────────────────────────────────────────────────────────

@router.post("/api/merge")
async def merge_datasets(request: MergeRequest):
    for dataset_id in request.dataset_ids:
        if dataset_id not in settings.active_datasets:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    datasets = [
        {
            "path": settings.DATASETS_DIR / did,
            "format": settings.active_datasets[did]["format"],
            "info": settings.active_datasets[did],
        }
        for did in request.dataset_ids
    ]

    new_dataset_id, output_path = _make_dataset_folder(request.output_name or "merged_dataset")
    settings.dataset_merger.merge(
        datasets, output_path, request.output_format, class_mapping=request.class_mapping
    )

    new_info = settings.dataset_parser.parse_dataset(output_path, request.output_format, new_dataset_id)
    new_info["id"] = new_dataset_id
    settings.active_datasets[new_dataset_id] = new_info

    return {"success": True, "merged_dataset": new_info}
