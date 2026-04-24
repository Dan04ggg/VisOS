"""
Inference Router - Run YOLO inference on images, video frames, and webcam streams.
Supports detection, segmentation, pose estimation, and multi-object tracking.
"""
import base64
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import settings

router = APIRouter()

# ── Smart-model types ─────────────────────────────────────────────────────────
_TEXT_PROMPT_TYPES  = {"yoloworld", "groundingdino", "owlvit", "sam", "sam2", "sam3"}
_POINT_PROMPT_TYPES = {"sam", "sam2", "sam3"}
_ZERO_SHOT_PREFIXES = ("sam_", "sam2_", "mobile_sam", "rtdetr", "fastsam")
_ZERO_SHOT_SUBSTRINGS = ("world", "clip", "groundingdino", "owlvit", "owl_vit")


def _is_plain_yolo(name: str) -> bool:
    """True for YOLO .pt files that work with raw YOLO predict."""
    lower = name.lower()
    if any(lower.startswith(p) for p in _ZERO_SHOT_PREFIXES):
        return False
    if any(s in lower for s in _ZERO_SHOT_SUBSTRINGS):
        return False
    return True


def _render_annotations(image_path: Path, annotations: List[Dict[str, Any]]) -> tuple:
    """Draw bbox/polygon annotations, return (jpeg_bytes, detections_list)."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
    ]
    detections: List[Dict[str, Any]] = []

    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        ann_type = ann.get("type", "bbox")
        cls_name = ann.get("class_name", "object")
        conf = float(ann.get("confidence") or 0.0)
        label = f"{cls_name} {conf:.2f}" if conf > 0 else cls_name

        if ann_type == "bbox":
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1)],
            })

        elif ann_type == "polygon":
            pts_flat = ann.get("points") or []
            if not pts_flat and "segmentation" in ann:
                seg = ann["segmentation"]
                pts_flat = seg[0] if (seg and isinstance(seg[0], list)) else seg
            if len(pts_flat) >= 6:
                pts = (np.array(pts_flat, dtype=np.float32)
                         .reshape(-1, 2).astype(np.int32))
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [pts], True, color, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(img, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "mask_points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                                    for p in pts[:30].tolist()],
                })

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf), detections


# ── Session manager (keeps model instances alive for tracking continuity) ─────
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 300  # seconds of inactivity before cleanup


def _cleanup_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items() if now - s["last_used"] > SESSION_TTL]
        for sid in stale:
            _sessions.pop(sid, None)


def _get_or_create_session(session_id: str, model_path: str, task: str, tracker: str) -> Dict:
    _cleanup_sessions()
    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing and existing["model_path"] == model_path and existing["task"] == task:
            existing["last_used"] = time.time()
            return existing
        # New or stale session — (re)create
        from ultralytics import YOLO
        model = YOLO(model_path)
        s: Dict[str, Any] = {
            "model": model,
            "model_path": model_path,
            "task": task,
            "tracker": tracker,
            "last_used": time.time(),
        }
        _sessions[session_id] = s
        return s


# ── Video processing jobs ─────────────────────────────────────────────────────
_video_jobs: Dict[str, Dict[str, Any]] = {}


# ── Result annotation helper ──────────────────────────────────────────────────

def _build_result(results, task: str, is_tracking: bool):
    """Return (jpeg_bytes, detections_list) from ultralytics Results."""
    import cv2

    result = results[0]
    annotated = result.plot()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_bytes = bytes(buf)

    detections: List[Dict[str, Any]] = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            d: Dict[str, Any] = {
                "class_id":   int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [round(float(v), 1) for v in box.xyxy[0].tolist()],
            }
            if is_tracking and box.id is not None:
                d["track_id"] = int(box.id[0])
            detections.append(d)

    if task == "segment" and result.masks is not None:
        for i, poly in enumerate(result.masks.xy):
            if i < len(detections):
                detections[i]["mask_points"] = [
                    [round(float(p[0]), 1), round(float(p[1]), 1)]
                    for p in poly[:30]
                ]

    if task == "pose" and result.keypoints is not None:
        for i, kp in enumerate(result.keypoints.xy):
            if i < len(detections):
                detections[i]["keypoints"] = [
                    [round(float(p[0]), 1), round(float(p[1]), 1)]
                    for p in kp.tolist()
                ]

    return img_bytes, detections


# ── Frame endpoint (webcam streaming) ─────────────────────────────────────────

class FrameRequest(BaseModel):
    model_path: str
    frame_b64: str          # base64-encoded JPEG
    task: str = "detect"   # detect | segment | pose | track
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    tracker: str = "bytetrack"   # bytetrack | botsort
    session_id: str = ""


@router.post("/api/infer/frame")
async def infer_frame(req: FrameRequest):
    if not req.model_path or not Path(req.model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {req.model_path}")
    try:
        import cv2
        import numpy as np

        img_data = base64.b64decode(req.frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")

        session_id = req.session_id or str(uuid.uuid4())[:8]
        is_tracking = req.task == "track"
        session = _get_or_create_session(session_id, req.model_path, req.task, req.tracker)
        model = session["model"]

        if is_tracking:
            results = model.track(
                frame, conf=req.confidence, iou=req.iou, imgsz=req.imgsz,
                tracker=f"{req.tracker}.yaml", persist=True, verbose=False,
            )
        else:
            results = model.predict(
                frame, conf=req.confidence, iou=req.iou, imgsz=req.imgsz, verbose=False,
            )

        img_bytes, detections = _build_result(results, req.task, is_tracking)
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/infer/session/{session_id}")
async def close_session(session_id: str):
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"success": True}


# ── Single-image endpoint ──────────────────────────────────────────────────────

@router.post("/api/infer/image")
async def infer_image(
    model_path: str = Form(...),
    task: str = Form("detect"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    image: UploadFile = File(...),
):
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO

        img_data = await image.read()
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        model = YOLO(model_path)
        results = model.predict(frame, conf=confidence, iou=iou, imgsz=imgsz, verbose=False)
        img_bytes, detections = _build_result(results, task, False)
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Video processing ───────────────────────────────────────────────────────────

def _run_video_job(job_id: str, model_path: str, video_path: str, task: str,
                   confidence: float, iou: float, imgsz: int, tracker: str) -> None:
    job = _video_jobs[job_id]
    try:
        import cv2
        from ultralytics import YOLO

        job["status"] = "running"
        model = YOLO(model_path)
        is_tracking = task == "track"

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(video_path).parent / f"{job_id}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if is_tracking:
                results = model.track(
                    frame, conf=confidence, iou=iou, imgsz=imgsz,
                    tracker=f"{tracker}.yaml", persist=True, verbose=False,
                )
            else:
                results = model.predict(
                    frame, conf=confidence, iou=iou, imgsz=imgsz, verbose=False,
                )
            out.write(results[0].plot())
            idx += 1
            job["progress"] = round(idx / total * 100, 1)

        cap.release()
        out.release()
        job["status"] = "completed"
        job["result_path"] = str(out_path)
        job["message"] = f"Processed {idx} frames"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Failed: {e}"


@router.post("/api/infer/video")
async def infer_video(
    model_path: str = Form(...),
    task: str = Form("detect"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    tracker: str = Form("bytetrack"),
    video: UploadFile = File(...),
):
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    job_id = str(uuid.uuid4())[:8]
    save_dir = settings.TEMP_DIR / "inference"
    save_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    video_path = save_dir / f"{job_id}_input{suffix}"
    video_path.write_bytes(await video.read())

    _video_jobs[job_id] = {
        "id": job_id, "status": "starting", "progress": 0,
        "message": "Starting…", "result_path": None, "error": None,
    }
    threading.Thread(
        target=_run_video_job,
        args=(job_id, model_path, str(video_path), task, confidence, iou, imgsz, tracker),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.get("/api/infer/video/{job_id}/status")
async def video_status(job_id: str):
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _video_jobs[job_id]
    return {"id": job_id, "status": j["status"], "progress": j["progress"],
            "message": j.get("message", ""), "error": j.get("error")}


@router.get("/api/infer/video/{job_id}/result")
async def video_result(job_id: str):
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _video_jobs[job_id]
    if j["status"] != "completed" or not j.get("result_path"):
        raise HTTPException(status_code=400, detail="Video not ready")
    p = j["result_path"]
    if not Path(p).exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(p, filename=f"inference_{job_id}.mp4", media_type="video/mp4")


# ── List YOLO models (training runs + workspace .pt files) ─────────────────────

@router.get("/api/infer/models")
async def list_infer_models():
    models = []
    for job in settings.training_manager.list_training_jobs():
        mp = job.get("model_path")
        if mp and Path(mp).exists() and _is_plain_yolo(Path(mp).name):
            models.append({
                "id": job["id"],
                "name": job["name"],
                "path": mp,
                "model_type": job["model_type"],
                "status": job["status"],
                "started_at": job["started_at"],
            })
    for pt in sorted(settings.MODELS_DIR.glob("*.pt")):
        if _is_plain_yolo(pt.name):
            models.append({
                "id": pt.stem,
                "name": pt.name,
                "path": str(pt),
                "model_type": "unknown",
                "status": "available",
                "started_at": "",
            })
    return {"models": models}


# ── Smart-model inference (SAM / YOLOWorld / GroundingDINO / OWL-ViT / …) ──────

class SmartFrameRequest(BaseModel):
    model_id: str
    frame_b64: str
    confidence: float = 0.5
    text_prompt: Optional[str] = None
    point_x: Optional[float] = None
    point_y: Optional[float] = None


@router.get("/api/infer/smart/models")
async def list_smart_models():
    """All loaded smart models ready for inference."""
    out = []
    for model_id, info in settings.model_manager.loaded_models.items():
        if info.get("model") is None:
            continue
        mtype = info.get("type", "unknown")
        out.append({
            "id":             model_id,
            "name":           info.get("name", model_id),
            "type":           mtype,
            "supports_text":  mtype in _TEXT_PROMPT_TYPES,
            "supports_point": mtype in _POINT_PROMPT_TYPES,
        })
    return {"models": out}


def _smart_infer_sync(model_id: str, tmp: Path, confidence: float,
                      text_prompt: Optional[str], point_x: Optional[float],
                      point_y: Optional[float]):
    """Blocking inference using model_manager. Returns (img_bytes, detections)."""
    model_info = settings.model_manager.loaded_models.get(model_id)
    if not model_info or model_info.get("model") is None:
        raise ValueError(f"Model '{model_id}' not loaded. Load it in the Models tab first.")
    model = model_info["model"]
    model_type = model_info.get("type", "unknown")
    prompt_point = (point_x, point_y) if point_x is not None and point_y is not None else None
    annotations = settings.model_manager._run_inference(
        model, model_type, tmp, confidence, prompt_point, text_prompt or None
    )
    return _render_annotations(tmp, annotations)


@router.post("/api/infer/smart/image")
async def infer_smart_image(
    model_id: str = Form(...),
    confidence: float = Form(0.5),
    text_prompt: Optional[str] = Form(None),
    point_x: Optional[float] = Form(None),
    point_y: Optional[float] = Form(None),
    image: UploadFile = File(...),
):
    save_dir = settings.TEMP_DIR / "inference"
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "img.jpg").suffix or ".jpg"
    tmp = save_dir / f"{str(uuid.uuid4())[:8]}_smart{suffix}"
    tmp.write_bytes(await image.read())
    try:
        img_bytes, detections = _smart_infer_sync(
            model_id, tmp, confidence, text_prompt, point_x, point_y
        )
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/api/infer/smart/frame")
async def infer_smart_frame(req: SmartFrameRequest):
    try:
        import cv2, numpy as np
        img_data = base64.b64decode(req.frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")

        save_dir = settings.TEMP_DIR / "inference"
        save_dir.mkdir(parents=True, exist_ok=True)
        tmp = save_dir / f"{str(uuid.uuid4())[:8]}_sframe.jpg"
        cv2.imwrite(str(tmp), frame)

        img_bytes, detections = _smart_infer_sync(
            req.model_id, tmp, req.confidence,
            req.text_prompt, req.point_x, req.point_y
        )
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "tmp" in dir() and tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Smart-model catalog + inline load/download ────────────────────────────────

# All smart models surfaced in the inference tab
_SMART_CATALOG: List[Dict[str, Any]] = [
    {"id": "sam_vit_b",           "name": "SAM Base",              "type": "sam"},
    {"id": "sam_vit_l",           "name": "SAM Large",             "type": "sam"},
    {"id": "sam2_tiny",           "name": "SAM 2 Tiny",            "type": "sam2"},
    {"id": "sam2_small",          "name": "SAM 2 Small",           "type": "sam2"},
    {"id": "sam2_base",           "name": "SAM 2 Base+",           "type": "sam2"},
    {"id": "sam2_large",          "name": "SAM 2 Large",           "type": "sam2"},
    {"id": "sam21_tiny",          "name": "SAM 2.1 Tiny",          "type": "sam2"},
    {"id": "sam21_small",         "name": "SAM 2.1 Small",         "type": "sam2"},
    {"id": "sam21_base",          "name": "SAM 2.1 Base+",         "type": "sam2"},
    {"id": "sam21_large",         "name": "SAM 2.1 Large",         "type": "sam2"},
    {"id": "sam3",                "name": "SAM 3",                  "type": "sam3"},
    {"id": "yoloworld_s",         "name": "YOLO-World S",          "type": "yoloworld"},
    {"id": "yoloworld_m",         "name": "YOLO-World M",          "type": "yoloworld"},
    {"id": "yoloworld_l",         "name": "YOLO-World L",          "type": "yoloworld"},
    {"id": "groundingdino_t",     "name": "GroundingDINO Tiny",    "type": "groundingdino"},
    {"id": "groundingdino_b",     "name": "GroundingDINO Base",    "type": "groundingdino"},
    {"id": "owlvit_base_patch32", "name": "OWL-ViT Base/32",       "type": "owlvit"},
    {"id": "owlvit_base_patch16", "name": "OWL-ViT Base/16",       "type": "owlvit"},
    {"id": "owlvit_large_patch14","name": "OWL-ViT Large/14",      "type": "owlvit"},
]


@router.get("/api/infer/smart/catalog")
async def smart_catalog():
    """Full smart-model catalog with live download/load status."""
    # Build a lookup from list_models() which already checks disk + HF cache
    all_models = {m["id"]: m for m in settings.model_manager.list_models()}

    out = []
    for entry in _SMART_CATALOG:
        mid   = entry["id"]
        mtype = entry["type"]
        info  = settings.model_manager.loaded_models.get(mid, {})
        dl    = settings._download_status.get(mid, {})
        disk  = all_models.get(mid, {})

        if info.get("model") is not None:
            status = "loaded"
        elif dl.get("status") == "downloading":
            status = "downloading"
        elif disk.get("downloaded"):
            status = "downloaded"
        else:
            status = "not_downloaded"

        out.append({
            "id":               mid,
            "name":             entry["name"],
            "type":             mtype,
            "status":           status,
            "download_progress": dl.get("progress", 0),
            "error":            dl.get("error") or info.get("error"),
            "supports_text":    mtype in _TEXT_PROMPT_TYPES,
            "supports_point":   mtype in _POINT_PROMPT_TYPES,
        })
    return {"models": out}


@router.post("/api/infer/smart/load")
async def load_smart_model(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    model_type: str = Form(...),
    hf_token: Optional[str] = Form(None),
):
    """Download (if needed) and load a smart model into memory."""
    info = settings.model_manager.loaded_models.get(model_id, {})
    if info.get("model") is not None:
        return {"status": "already_loaded"}
    if settings._download_status.get(model_id, {}).get("status") == "downloading":
        return {"status": "already_loading"}

    settings._download_status[model_id] = {
        "status": "downloading", "progress": 5,
        "phase": "starting", "message": f"Loading {model_id}…",
    }

    def _do_load(mtype: str, mname: str, token: Optional[str]):
        try:
            def _phase(phase: str):
                if phase == "loading_memory":
                    settings._download_status[mname] = {
                        "status": "downloading", "progress": 80,
                        "phase": "loading_memory",
                        "message": "Loading into memory…",
                    }
            settings.model_manager.load_pretrained(mtype, mname, hf_token=token, _status_hook=_phase)
            loaded = settings.model_manager.loaded_models.get(mname, {})
            if loaded.get("error") and not loaded.get("path"):
                settings._download_status[mname] = {
                    "status": "error", "progress": 0, "error": loaded["error"]
                }
            else:
                settings._download_status[mname] = {"status": "done", "progress": 100}
        except Exception as e:
            settings._download_status[mname] = {
                "status": "error", "progress": 0, "error": str(e)
            }

    background_tasks.add_task(_do_load, model_type, model_id, hf_token)
    return {"status": "started"}


@router.get("/api/infer/smart/load-status/{model_id}")
async def smart_load_status(model_id: str):
    return settings._download_status.get(model_id, {"status": "idle", "progress": 0})
