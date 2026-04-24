import threading
import uuid
import yaml
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from config import settings
from core.workspace import _make_dataset_folder, _save_dataset_metadata
from schemas.common import VideoExtractRequest

router = APIRouter()


@router.post("/api/videos/upload")
async def upload_video(video: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    video_filename = video.filename or f"video_{video_id}.mp4"
    video_path = settings.VIDEOS_DIR / f"{video_id}_{video_filename}"

    content = await video.read()
    with open(video_path, "wb") as f:
        f.write(content)

    info = settings.video_extractor.get_video_info(video_path)

    video_data = {
        "id": video_id,
        "filename": video_filename,
        "path": str(video_path),
        "url": f"/api/videos/{video_id}/stream",
        "duration":     info.get("duration", 0)       if info.get("success") else 60,
        "fps":          info.get("fps", 30)            if info.get("success") else 30,
        "total_frames": info.get("total_frames", 1800) if info.get("success") else 1800,
        "width":        info.get("width", 1920)        if info.get("success") else 1920,
        "height":       info.get("height", 1080)       if info.get("success") else 1080,
        "thumbnail": None,
    }

    settings._uploaded_videos[video_id] = video_data
    return video_data


@router.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str):
    if video_id not in settings._uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(settings._uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(video_path, media_type="video/mp4")


@router.post("/api/video/extract")
@router.post("/api/videos/extract-frames")
async def extract_video_frames(request: VideoExtractRequest):
    if request.video_id and request.video_id in settings._uploaded_videos:
        video_path = Path(settings._uploaded_videos[request.video_id]["path"])
    elif request.video_path:
        video_path = Path(request.video_path)
    else:
        raise HTTPException(status_code=400, detail="No video specified")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    nth_frame = request.nth_frame
    if request.frame_interval:
        nth_frame = request.frame_interval

    # Determine target output dir
    if request.existing_dataset_id and request.existing_dataset_id in settings.active_datasets:
        dataset_id = request.existing_dataset_id
        dataset_path = settings.DATASETS_DIR / dataset_id
        output_path = dataset_path / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        is_existing = True
    else:
        dataset_id, _dataset_dir = _make_dataset_folder(request.output_name or "video_frames")
        dataset_path = settings.DATASETS_DIR / dataset_id
        output_path = dataset_path / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        is_existing = False

    if request.mode == "uniform":
        effective_max_frames = request.max_frames or request.uniform_count
    else:
        effective_max_frames = request.max_frames or None

    job_id = str(uuid.uuid4())[:8]
    settings._extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "extracted_frames": 0,
        "dataset_id": None,
        "new_dataset": None,
        "error": None,
    }

    def _run_extraction():
        def _progress(pct: int, count: int):
            settings._extraction_jobs[job_id]["progress"] = pct
            settings._extraction_jobs[job_id]["extracted_frames"] = count

        result = settings.video_extractor.extract_frames(
            video_path, output_path,
            nth_frame=nth_frame,
            max_frames=effective_max_frames,
            start_time=request.start_time,
            end_time=request.end_time,
            progress_callback=_progress,
        )

        if not result["success"]:
            if not is_existing:
                import shutil
                shutil.rmtree(dataset_path, ignore_errors=True)
            settings._extraction_jobs[job_id]["status"] = "error"
            settings._extraction_jobs[job_id]["error"] = result.get("error", "Extraction failed")
            return

        if not is_existing:
            (dataset_path / "labels").mkdir(exist_ok=True)
            config = {
                "path": str(dataset_path.absolute()),
                "train": "images",
                "val": "images",
                "names": {},
                "nc": 0,
            }
            with open(dataset_path / "data.yaml", "w") as f:
                yaml.dump(config, f)

        dataset_name = (
            settings.active_datasets[dataset_id]["name"]
            if is_existing
            else (request.output_name or "video_frames")
        )
        fmt = settings.active_datasets[dataset_id]["format"] if is_existing else "yolo"
        dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset_name)
        dataset_info["id"] = dataset_id
        dataset_info["source_video"] = str(video_path)
        settings.active_datasets[dataset_id] = dataset_info
        _save_dataset_metadata(dataset_id, dataset_info)
        if dataset_id in settings._images_cache:
            del settings._images_cache[dataset_id]

        settings._extraction_jobs[job_id]["status"] = "done"
        settings._extraction_jobs[job_id]["progress"] = 100
        settings._extraction_jobs[job_id]["extracted_frames"] = result["extracted_frames"]
        settings._extraction_jobs[job_id]["dataset_id"] = dataset_id
        settings._extraction_jobs[job_id]["new_dataset"] = dataset_info
        settings._extraction_jobs[job_id]["is_existing"] = is_existing

    threading.Thread(target=_run_extraction, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/videos/extract/{job_id}/status")
async def get_extraction_status(job_id: str):
    if job_id not in settings._extraction_jobs:
        raise HTTPException(status_code=404, detail="Extraction job not found")
    return settings._extraction_jobs[job_id]


@router.get("/api/video/info")
async def get_video_info(video_path: str):
    result = settings.video_extractor.get_video_info(Path(video_path))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to get video info"))
    return result
