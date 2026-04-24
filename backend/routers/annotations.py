import shutil
import threading
from typing import Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response

from config import settings
from core.startup import _persist_jobs
from core.workspace import _save_dataset_metadata

router = APIRouter()


@router.post("/api/auto-annotate/{dataset_id}")
async def auto_annotate_dataset(
    dataset_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    results = settings.model_manager.auto_annotate(
        model_id, dataset_path, dataset["format"], confidence_threshold
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {"success": True, "annotated_images": results["annotated_count"]}


@router.post("/api/auto-annotate/{dataset_id}/single/{image_id}")
async def auto_annotate_single_image(
    dataset_id: str,
    image_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    point_x: Optional[float] = None,
    point_y: Optional[float] = None,
    text_prompt: Optional[str] = None,
    image_path_hint: Optional[str] = None,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    prompt_point = (point_x, point_y) if point_x is not None and point_y is not None else None

    try:
        annotations = settings.model_manager.annotate_single_image(
            model_id, dataset_path, dataset["format"], image_id, confidence_threshold,
            prompt_point=prompt_point,
            text_prompt=text_prompt or None,
            image_path_hint=image_path_hint,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"success": True, "annotations": annotations}


@router.post("/api/auto-annotate/{dataset_id}/text-batch")
async def auto_annotate_text_batch(
    dataset_id: str,
    model_id: str,
    text_prompt: Optional[str] = None,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    job_id = __import__("uuid").uuid4().hex[:8]

    pause_ev = threading.Event(); pause_ev.set()
    stop_ev = threading.Event()

    settings._text_annotate_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "paused": False,
        "progress": 0,
        "total": 0,
        "processed": 0,
        "annotated": 0,
        "failed": 0,
        "total_annotations": 0,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "confidence_threshold": confidence_threshold,
        "text_prompt": text_prompt or "",
        "started_at": datetime.utcnow().isoformat(),
        "recent_images": [],
        "all_images": [],
        "snapshot": None,
    }
    settings._job_controls[job_id] = {"pause": pause_ev, "stop": stop_ev}
    _persist_jobs()

    def _run_batch(job_id: str):
        try:
            _snap: dict = {"yaml": {}, "labels": {}, "metadata": None}
            for _yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
                try:
                    _snap["yaml"][_yf.name] = _yf.read_text(encoding="utf-8")
                except Exception:
                    pass
            _labels_dir = dataset_path / "labels"
            if _labels_dir.exists():
                for _lf in _labels_dir.rglob("*.txt"):
                    try:
                        _snap["labels"][str(_lf.relative_to(dataset_path))] = _lf.read_text(encoding="utf-8")
                    except Exception:
                        pass
            _meta_path = dataset_path / "dataset_metadata.json"
            if _meta_path.exists():
                try:
                    _snap["metadata"] = _meta_path.read_text(encoding="utf-8")
                except Exception:
                    pass
            settings._text_annotate_jobs[job_id]["snapshot"] = _snap

            if dataset.get("format") == "generic-images":
                dataset["format"] = "yolo"
                meta_path = dataset_path / "dataset_metadata.json"
                try:
                    import json as _json
                    meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    meta["format"] = "yolo"
                    meta_path.write_text(_json.dumps(meta, indent=2))
                except Exception as _me:
                    print(f"[batch:{job_id}] failed to persist format upgrade: {_me}")
                _img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                _root_p = settings.dataset_parser._find_dataset_root(dataset_path)
                _imgs_dir = _root_p / "images"
                _imgs_dir.mkdir(exist_ok=True)
                for _f in list(_root_p.iterdir()):
                    if _f.is_file() and _f.suffix.lower() in _img_exts:
                        shutil.move(str(_f), _imgs_dir / _f.name)
                if dataset_id in settings._images_cache:
                    del settings._images_cache[dataset_id]

            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            _seen_paths: set = set()
            image_files: list = []
            for ext in IMAGE_EXTENSIONS:
                for _f in list(dataset_path.rglob(f"*{ext}")) + list(dataset_path.rglob(f"*{ext.upper()}")):
                    if "labels" in _f.parts:
                        continue
                    _rp = _f.resolve()
                    if _rp not in _seen_paths:
                        _seen_paths.add(_rp)
                        image_files.append(_f)

            settings._text_annotate_jobs[job_id]["total"] = len(image_files)
            classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
            ctrl = settings._job_controls[job_id]

            for i, image_file in enumerate(image_files):
                if ctrl["stop"].is_set():
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    settings._images_cache.pop(dataset_id, None)
                    return

                if not ctrl["pause"].is_set():
                    settings._text_annotate_jobs[job_id]["paused"] = True
                    settings._text_annotate_jobs[job_id]["status"] = "paused"
                    ctrl["pause"].wait()
                    if ctrl["stop"].is_set():
                        settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                        settings._images_cache.pop(dataset_id, None)
                        return
                    settings._text_annotate_jobs[job_id]["paused"] = False
                    settings._text_annotate_jobs[job_id]["status"] = "running"

                try:
                    annotations = settings.model_manager._run_inference(
                        settings.model_manager.loaded_models[model_id]["model"],
                        settings.model_manager.loaded_models[model_id]["type"],
                        image_file,
                        confidence_threshold,
                        text_prompt=text_prompt or None,
                        model_classes=settings.model_manager.loaded_models[model_id].get("classes") or [],
                    )
                except Exception as exc:
                    print(f"[batch:{job_id}] inference failed on {image_file.name}: {exc}")
                    settings._text_annotate_jobs[job_id]["failed"] += 1
                    settings._text_annotate_jobs[job_id]["processed"] = i + 1
                    settings._text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                    continue

                if annotations:
                    try:
                        settings.model_manager._save_annotations(
                            dataset_path, dataset["format"], image_file, annotations, classes
                        )
                        settings._text_annotate_jobs[job_id]["annotated"] += 1
                        settings._text_annotate_jobs[job_id]["total_annotations"] += len(annotations)
                        try:
                            rel_path = str(image_file.relative_to(dataset_path))
                        except Exception:
                            rel_path = image_file.name
                        img_entry = {
                            "filename": image_file.name,
                            "path": rel_path,
                            "abs_path": str(image_file),
                            "image_id": image_file.stem,
                            "annotation_count": len(annotations),
                        }
                        recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
                        recent.append(img_entry)
                        settings._text_annotate_jobs[job_id]["recent_images"] = recent[-10:]
                        all_imgs = settings._text_annotate_jobs[job_id].get("all_images", [])
                        all_imgs.append(img_entry)
                        settings._text_annotate_jobs[job_id]["all_images"] = all_imgs
                        settings._images_cache.pop(dataset_id, None)
                    except Exception as exc:
                        print(f"[batch:{job_id}] save failed on {image_file.name}: {exc}")
                        settings._text_annotate_jobs[job_id]["failed"] += 1
                settings._text_annotate_jobs[job_id]["processed"] = i + 1
                settings._text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                if i % 5 == 4:
                    _persist_jobs()

            settings._images_cache.pop(dataset_id, None)
            settings._text_annotate_jobs[job_id]["status"] = "done"
            _persist_jobs()
        except Exception as e:
            settings._text_annotate_jobs[job_id]["status"] = "error"
            settings._text_annotate_jobs[job_id]["error"] = str(e)
            _persist_jobs()

    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        settings.model_manager._try_autoload(model_id)
    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        raise HTTPException(status_code=400, detail="Model not loaded or unavailable")

    threading.Thread(target=_run_batch, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/auto-annotate/text-batch/{job_id}/status")
async def get_text_batch_status(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return settings._text_annotate_jobs[job_id]


@router.post("/api/auto-annotate/text-batch/{job_id}/pause")
async def pause_text_batch(job_id: str):
    if job_id not in settings._job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] not in ("running",):
        raise HTTPException(status_code=400, detail="Job is not running")
    settings._job_controls[job_id]["pause"].clear()
    settings._text_annotate_jobs[job_id]["paused"] = True
    settings._text_annotate_jobs[job_id]["status"] = "paused"
    _persist_jobs()
    return {"status": "paused"}


@router.post("/api/auto-annotate/text-batch/{job_id}/resume")
async def resume_text_batch(job_id: str):
    if job_id not in settings._job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] not in ("paused",):
        raise HTTPException(status_code=400, detail="Job is not paused")
    settings._text_annotate_jobs[job_id]["paused"] = False
    settings._text_annotate_jobs[job_id]["status"] = "running"
    settings._job_controls[job_id]["pause"].set()
    _persist_jobs()
    return {"status": "running"}


@router.post("/api/auto-annotate/text-batch/{job_id}/cancel")
async def cancel_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    current_status = settings._text_annotate_jobs[job_id]["status"]
    if current_status in ("done", "cancelled", "error"):
        raise HTTPException(status_code=400, detail="Job is already finished")
    if job_id in settings._job_controls:
        ctrl = settings._job_controls[job_id]
        ctrl["stop"].set()
        ctrl["pause"].set()
    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
    _persist_jobs()
    return {"status": "cancelled"}


@router.post("/api/auto-annotate/text-batch/{job_id}/undo")
async def undo_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = settings._text_annotate_jobs[job_id]
    if job["status"] not in ("done", "cancelled", "error", "interrupted"):
        raise HTTPException(status_code=400, detail="Job must be finished to undo")

    snapshot = job.get("snapshot")
    if not snapshot:
        raise HTTPException(status_code=400, detail="No snapshot available — cannot undo this job")

    dataset_id = job["dataset_id"]
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = settings.DATASETS_DIR / dataset_id

    labels_dir = dataset_path / "labels"
    if labels_dir.exists():
        for lf in labels_dir.rglob("*.txt"):
            try:
                lf.unlink()
            except Exception:
                pass

    for rel_path, content in snapshot.get("labels", {}).items():
        restore_path = dataset_path / rel_path
        restore_path.parent.mkdir(parents=True, exist_ok=True)
        restore_path.write_text(content, encoding="utf-8")

    current_yamls = set(yf.name for yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")))
    snap_yamls = snapshot.get("yaml", {})
    for filename, content in snap_yamls.items():
        (dataset_path / filename).write_text(content, encoding="utf-8")
    for yaml_name in current_yamls - set(snap_yamls.keys()):
        try:
            (dataset_path / yaml_name).unlink()
        except Exception:
            pass

    meta_content = snapshot.get("metadata")
    meta_path = dataset_path / "dataset_metadata.json"
    if meta_content:
        meta_path.write_text(meta_content, encoding="utf-8")
        try:
            import json as _jj
            restored_meta = _jj.loads(meta_content)
            settings.active_datasets[dataset_id] = {
                **settings.active_datasets[dataset_id],
                "format": restored_meta.get("format", settings.active_datasets[dataset_id]["format"]),
            }
        except Exception:
            pass

    settings._images_cache.pop(dataset_id, None)
    job["status"] = "undone"
    job["snapshot"] = None
    _persist_jobs()
    return {"status": "undone"}


@router.get("/api/auto-annotate/jobs")
async def list_batch_jobs():
    return {"jobs": list(settings._text_annotate_jobs.values())}


@router.delete("/api/auto-annotate/text-batch/{job_id}")
async def delete_batch_job(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] in ("running", "paused"):
        raise HTTPException(status_code=400, detail="Cancel the job before deleting")
    del settings._text_annotate_jobs[job_id]
    settings._job_controls.pop(job_id, None)
    _persist_jobs()
    return {"status": "deleted"}


@router.post("/api/auto-annotate/text-batch/{job_id}/restart")
async def restart_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = settings._text_annotate_jobs[job_id]
    if job["status"] not in ("interrupted", "cancelled", "error"):
        raise HTTPException(status_code=400, detail="Job is not restartable")

    dataset_id = job["dataset_id"]
    model_id = job.get("model_id")
    confidence_threshold = job.get("confidence_threshold", 0.5)
    text_prompt = job["text_prompt"]

    if not model_id:
        raise HTTPException(status_code=400, detail="Job has no stored model_id; start a new job instead")
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        settings.model_manager._try_autoload(model_id)
    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        raise HTTPException(status_code=400, detail="Model not loaded or unavailable")

    pause_ev = threading.Event(); pause_ev.set()
    stop_ev = threading.Event()
    settings._job_controls[job_id] = {"pause": pause_ev, "stop": stop_ev}
    job["status"] = "running"
    job["paused"] = False
    _persist_jobs()

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    already_done = {img["image_id"] for img in job.get("all_images", [])}

    def _continue_batch(job_id: str):
        try:
            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            _seen_paths2: set = set()
            image_files: list = []
            for ext in IMAGE_EXTENSIONS:
                for _f in list(dataset_path.rglob(f"*{ext}")) + list(dataset_path.rglob(f"*{ext.upper()}")):
                    if "labels" in _f.parts:
                        continue
                    _rp = _f.resolve()
                    if _rp not in _seen_paths2:
                        _seen_paths2.add(_rp)
                        image_files.append(_f)

            all_count = len(image_files)
            remaining = [f for f in image_files if f.stem not in already_done]
            settings._text_annotate_jobs[job_id]["total"] = all_count
            classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
            ctrl = settings._job_controls[job_id]

            for i, image_file in enumerate(remaining):
                if ctrl["stop"].is_set():
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    settings._images_cache.pop(dataset_id, None)
                    return
                if not ctrl["pause"].is_set():
                    settings._text_annotate_jobs[job_id]["paused"] = True
                    settings._text_annotate_jobs[job_id]["status"] = "paused"
                    ctrl["pause"].wait()
                    if ctrl["stop"].is_set():
                        settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                        settings._images_cache.pop(dataset_id, None)
                        return
                    settings._text_annotate_jobs[job_id]["paused"] = False
                    settings._text_annotate_jobs[job_id]["status"] = "running"

                try:
                    annotations = settings.model_manager._run_inference(
                        settings.model_manager.loaded_models[model_id]["model"],
                        settings.model_manager.loaded_models[model_id]["type"],
                        image_file,
                        confidence_threshold,
                        text_prompt=text_prompt,
                        model_classes=settings.model_manager.loaded_models[model_id].get("classes") or [],
                    )
                except Exception as exc:
                    print(f"[batch:{job_id}] inference failed on {image_file.name}: {exc}")
                    settings._text_annotate_jobs[job_id]["failed"] += 1
                    settings._text_annotate_jobs[job_id]["processed"] += 1
                    total_done = len(already_done) + i + 1
                    settings._text_annotate_jobs[job_id]["progress"] = int(total_done / max(all_count, 1) * 100)
                    continue

                if annotations:
                    try:
                        settings.model_manager._save_annotations(
                            dataset_path, dataset["format"], image_file, annotations, classes
                        )
                        settings._text_annotate_jobs[job_id]["annotated"] += 1
                        settings._text_annotate_jobs[job_id]["total_annotations"] += len(annotations)
                        try:
                            rel_path = str(image_file.relative_to(dataset_path))
                        except Exception:
                            rel_path = image_file.name
                        img_entry = {
                            "filename": image_file.name,
                            "path": rel_path,
                            "abs_path": str(image_file),
                            "image_id": image_file.stem,
                            "annotation_count": len(annotations),
                        }
                        recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
                        recent.append(img_entry)
                        settings._text_annotate_jobs[job_id]["recent_images"] = recent[-10:]
                        all_imgs = settings._text_annotate_jobs[job_id].get("all_images", [])
                        all_imgs.append(img_entry)
                        settings._text_annotate_jobs[job_id]["all_images"] = all_imgs
                        settings._images_cache.pop(dataset_id, None)
                    except Exception as exc:
                        print(f"[batch:{job_id}] save failed on {image_file.name}: {exc}")
                        settings._text_annotate_jobs[job_id]["failed"] += 1
                settings._text_annotate_jobs[job_id]["processed"] += 1
                total_done = len(already_done) + i + 1
                settings._text_annotate_jobs[job_id]["progress"] = int(total_done / max(all_count, 1) * 100)
                if i % 5 == 4:
                    _persist_jobs()

            settings._images_cache.pop(dataset_id, None)
            settings._text_annotate_jobs[job_id]["status"] = "done"
            _persist_jobs()
        except Exception as e:
            settings._text_annotate_jobs[job_id]["status"] = "error"
            settings._text_annotate_jobs[job_id]["error"] = str(e)
            _persist_jobs()

    threading.Thread(target=_continue_batch, args=(job_id,), daemon=True).start()
    return {"status": "running"}


@router.get("/api/auto-annotate/text-batch/{job_id}/preview/{idx}")
async def get_batch_job_preview_image(job_id: str, idx: int):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
    if idx < 0 or idx >= len(recent):
        raise HTTPException(status_code=404, detail="Image index out of range")
    abs_path = recent[idx].get("abs_path")
    if not abs_path or not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(abs_path)


@router.get("/api/auto-annotate/text-batch/{job_id}/processed-images")
async def get_batch_job_all_images(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"images": settings._text_annotate_jobs[job_id].get("all_images", [])}


@router.get("/api/auto-annotate/text-batch/{job_id}/image/{image_id}")
async def get_batch_job_image_by_id(job_id: str, image_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    for img in settings._text_annotate_jobs[job_id].get("all_images", []):
        if img.get("image_id") == image_id:
            abs_path = img.get("abs_path")
            if abs_path and Path(abs_path).exists():
                return FileResponse(abs_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/api/auto-annotate/text-batch/{job_id}/annotated/{image_id}")
async def get_batch_job_annotated_image(job_id: str, image_id: str):
    import cv2
    import numpy as np

    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = settings._text_annotate_jobs[job_id]
    dataset_id = job.get("dataset_id")
    if not dataset_id or dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    all_images = job.get("all_images", [])
    img_entry = next((img for img in all_images if img.get("image_id") == image_id), None)
    if not img_entry:
        raise HTTPException(status_code=404, detail="Image not found in job")

    abs_path = img_entry.get("abs_path")
    if not abs_path or not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    img = cv2.imread(abs_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    h, w = img.shape[:2]
    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])

    def get_class_color(idx):
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        ]
        return colors[idx % len(colors)]

    rel_path = img_entry.get("path", "")
    image_file = Path(abs_path)
    label_file = None

    if dataset["format"] == "yolo":
        search_paths = []
        abs_path_str = str(abs_path)
        if '/images/' in abs_path_str or '\\images\\' in abs_path_str:
            label_from_images = abs_path_str.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
            label_from_images = Path(label_from_images).with_suffix('.txt')
            search_paths.append(label_from_images)
        search_paths.append(image_file.with_suffix('.txt'))
        search_paths.append(image_file.parent / 'labels' / (image_file.stem + '.txt'))
        if rel_path:
            rel_label = Path(rel_path).with_suffix('.txt')
            search_paths.append(dataset_path / 'labels' / rel_label.name)
            for split in ['train', 'valid', 'val', 'test']:
                search_paths.append(dataset_path / 'labels' / split / (image_file.stem + '.txt'))
        search_paths.append(dataset_path / 'labels' / (image_file.stem + '.txt'))

        print(f"[batch-annotate] Searching for label file for {image_file.name}")
        for sp in search_paths:
            print(f"[batch-annotate]   Checking: {sp} -> exists={sp.exists() if sp else False}")
            if sp and sp.exists():
                label_file = sp
                print(f"[batch-annotate]   FOUND: {label_file}")
                break
        if not label_file:
            print(f"[batch-annotate]   No label file found for {image_file.name}")

    if label_file and label_file.exists():
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_idx = int(parts[0])
                    color = get_class_color(class_idx)
                    class_name = classes[class_idx] if class_idx < len(classes) else f"class_{class_idx}"
                    coords = parts[1:]
                    if len(coords) == 4:
                        cx, cy, bw, bh = map(float, coords)
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(img, class_name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        polygon_points = []
                        for i_c in range(0, len(coords) - 1, 2):
                            px = int(float(coords[i_c]) * w)
                            py = int(float(coords[i_c + 1]) * h)
                            polygon_points.append([px, py])
                        if len(polygon_points) >= 3:
                            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                            overlay = img.copy()
                            cv2.fillPoly(overlay, [pts], color)
                            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                            M = cv2.moments(pts)
                            if M["m00"] != 0:
                                cx_pt = int(M["m10"] / M["m00"])
                                cy_pt = int(M["m01"] / M["m00"])
                            else:
                                cx_pt, cy_pt = polygon_points[0]
                            (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(img, (cx_pt - 2, cy_pt - th - 4), (cx_pt + tw + 4, cy_pt + 2), color, -1)
                            cv2.putText(img, class_name, (cx_pt, cy_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"[batch] Error drawing annotations: {e}")

    _, buffer = cv2.imencode('.jpg', img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.delete("/api/datasets/{dataset_id}/annotations")
async def delete_all_annotations(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    deleted = 0

    if fmt in settings.annotation_manager.YOLO_FORMATS:
        for label_file in dataset_path.rglob("*.txt"):
            if "labels" not in label_file.parts:
                continue
            try:
                label_file.write_text("")
                deleted += 1
            except Exception:
                pass
    elif fmt in ("pascal-voc", "voc"):
        for xml_file in dataset_path.rglob("*.xml"):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall("object"):
                    root.remove(obj)
                tree.write(xml_file)
                deleted += 1
            except Exception:
                pass

    settings._images_cache.pop(dataset_id, None)
    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    return {"success": True, "cleared_files": deleted, "updated_dataset": dataset_info}
