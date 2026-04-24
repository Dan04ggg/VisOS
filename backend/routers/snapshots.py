import json
import zipfile
import shutil
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from schemas.common import SnapshotRequest

router = APIRouter()


@router.get("/api/datasets/{dataset_id}/snapshots")
async def list_snapshots(dataset_id: str):
    snap_dir = settings.SNAPSHOTS_DIR / dataset_id
    snapshots = []
    if snap_dir.exists():
        for snap_zip in sorted(snap_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            snapshot_id = snap_zip.stem
            stat = snap_zip.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
            meta_path = snap_dir / f"{snapshot_id}.json"
            name = snapshot_id
            description = ""
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    name = meta.get("name", snapshot_id)
                    description = meta.get("description", "")
                    created_at = meta.get("created_at", created_at)
                except Exception:
                    pass
            snapshots.append({
                "id": snapshot_id,
                "name": name,
                "description": description,
                "dataset_id": dataset_id,
                "created_at": created_at,
                "size_mb": round(size_mb, 2),
            })
    return {"snapshots": snapshots}


@router.post("/api/datasets/{dataset_id}/snapshot")
async def create_snapshot(dataset_id: str, request: SnapshotRequest):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    import time as _time

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    snapshot_id = f"snap_{int(_time.time() * 1000)}"
    snap_dir = settings.SNAPSHOTS_DIR / dataset_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_zip = snap_dir / f"{snapshot_id}.zip"

    try:
        with zipfile.ZipFile(snap_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_path.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(dataset_path))
        size_mb = snap_zip.stat().st_size / (1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {e}")

    created_at = datetime.now().isoformat()
    meta = {
        "name": request.name or snapshot_id,
        "description": request.description or "",
        "created_at": created_at,
        "num_images": dataset.get("num_images", 0),
        "num_annotations": dataset.get("num_annotations", 0),
    }
    try:
        meta_path = snap_dir / f"{snapshot_id}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    except Exception:
        pass

    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "size_mb": round(size_mb, 2),
        "created_at": created_at,
        "name": meta["name"],
    }


@router.get("/api/datasets/{dataset_id}/snapshot/{snapshot_id}/download")
async def download_snapshot(dataset_id: str, snapshot_id: str):
    snap_zip = settings.SNAPSHOTS_DIR / dataset_id / f"{snapshot_id}.zip"
    if not snap_zip.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path=str(snap_zip), media_type="application/zip", filename=f"{snapshot_id}.zip")


@router.post("/api/datasets/{dataset_id}/snapshot/{snapshot_id}/restore")
async def restore_snapshot(dataset_id: str, snapshot_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    snap_zip = settings.SNAPSHOTS_DIR / dataset_id / f"{snapshot_id}.zip"
    if not snap_zip.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")

    dataset_path = settings.DATASETS_DIR / dataset_id
    dataset = settings.active_datasets[dataset_id]
    fmt = dataset["format"]
    name = dataset["name"]

    try:
        shutil.rmtree(dataset_path)
        dataset_path.mkdir(parents=True)
        with zipfile.ZipFile(snap_zip, "r") as zf:
            zf.extractall(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")

    settings._images_cache.pop(dataset_id, None)
    try:
        dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, name)
        dataset_info["id"] = dataset_id
        settings.active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True}


@router.delete("/api/datasets/{dataset_id}/snapshot/{snapshot_id}")
async def delete_snapshot(dataset_id: str, snapshot_id: str):
    snap_dir = settings.SNAPSHOTS_DIR / dataset_id
    snap_zip = snap_dir / f"{snapshot_id}.zip"
    meta_path = snap_dir / f"{snapshot_id}.json"
    if not snap_zip.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    try:
        snap_zip.unlink()
        if meta_path.exists():
            meta_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"success": True}
