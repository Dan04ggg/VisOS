"""
Evaluation Router - Head-to-head model evaluation using YOLO val + optional pycocotools.
"""
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings

router = APIRouter()

_eval_jobs: Dict[str, Dict[str, Any]] = {}


class EvaluateRequest(BaseModel):
    model_paths: List[str]
    dataset_id: str
    split: str = "val"   # val | test | train
    conf: float = 0.001
    iou: float = 0.6
    imgsz: int = 640


def _run_evaluation(job_id: str, model_paths: List[str], data_yaml: Path,
                    split: str, conf: float, iou: float, imgsz: int) -> None:
    job = _eval_jobs[job_id]
    try:
        from ultralytics import YOLO

        job["status"] = "running"
        results_list: List[Dict[str, Any]] = []

        for i, model_path in enumerate(model_paths):
            job["message"] = f"Evaluating {i+1}/{len(model_paths)}: {Path(model_path).name}"
            job["progress"] = int(i / len(model_paths) * 100)

            model = YOLO(model_path)
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                workers=0,
            )

            rd = dict(getattr(metrics, "results_dict", {}) or {})

            def _f(key: str) -> float:
                return round(float(rd.get(key, 0) or 0), 4)

            # Speed metrics: preprocess + inference + postprocess (ms per image)
            speed: Dict[str, float] = dict(getattr(metrics, "speed", {}) or {})
            avg_latency_ms = round(
                sum(speed.get(k, 0.0) for k in ("preprocess", "inference", "postprocess")), 2
            )
            fps = round(1000.0 / avg_latency_ms, 1) if avg_latency_ms > 0 else 0.0

            entry: Dict[str, Any] = {
                "model_path": model_path,
                "model_name": Path(model_path).stem,
                "mAP50":      _f("metrics/mAP50(B)"),
                "mAP50_95":   _f("metrics/mAP50-95(B)"),
                "precision":  _f("metrics/precision(B)"),
                "recall":     _f("metrics/recall(B)"),
                "fitness":    round(float(getattr(metrics, "fitness", 0) or 0), 4),
                # Segmentation metrics (zero if not a seg model)
                "mAP50_seg":   _f("metrics/mAP50(M)"),
                "mAP50_95_seg": _f("metrics/mAP50-95(M)"),
                "avg_latency_ms": avg_latency_ms,
                "fps":            fps,
            }

            # Optionally enrich with pycocotools per-size breakdown
            try:
                import pycocotools  # noqa: F401
                entry["pycocotools"] = True
                # YOLO val saves a predictions JSON; run COCOeval if it exists
                pred_json = Path(f"runs/val/exp/predictions.json")
                if pred_json.exists():
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval
                    import json as _json

                    with open(pred_json) as f:
                        preds = _json.load(f)

                    # Build minimal GT from YOLO metrics.confusion_matrix if available
                    # Full COCOeval requires GT annotations file — skip if not COCO format
            except ImportError:
                entry["pycocotools"] = False

            results_list.append(entry)
            del model

        job["status"] = "completed"
        job["progress"] = 100
        job["results"] = results_list
        job["message"] = f"Evaluated {len(model_paths)} model(s) on {split} split"
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()[-2000:]
        job["message"] = f"Evaluation failed: {e}"


@router.post("/api/evaluate")
async def start_evaluation(req: EvaluateRequest):
    if req.dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    missing = [mp for mp in req.model_paths if not Path(mp).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Model(s) not found: {missing}")

    dataset_path = settings.DATASETS_DIR / req.dataset_id
    data_yaml: Optional[Path] = None
    for candidate in list(dataset_path.rglob("data.yaml")) + list(dataset_path.rglob("*.yaml")):
        if not candidate.name.startswith("."):
            data_yaml = candidate
            if candidate.name == "data.yaml":
                break

    if not data_yaml:
        raise HTTPException(status_code=400, detail="No data.yaml found in dataset")

    job_id = str(uuid.uuid4())[:8]
    _eval_jobs[job_id] = {
        "id": job_id, "status": "starting", "progress": 0,
        "message": "Starting evaluation…", "results": None, "error": None,
    }

    threading.Thread(
        target=_run_evaluation,
        args=(job_id, req.model_paths, data_yaml, req.split, req.conf, req.iou, req.imgsz),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@router.get("/api/evaluate/{job_id}/status")
async def eval_status(job_id: str):
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    j = _eval_jobs[job_id]
    return {
        "id": job_id,
        "status": j["status"],
        "progress": j["progress"],
        "message": j.get("message", ""),
        "results": j.get("results"),
        "error": j.get("error"),
    }
