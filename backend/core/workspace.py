"""Workspace path helpers and dataset-restore logic."""

import re
import json
import logging
from pathlib import Path
from typing import Tuple

from config import settings

logger = logging.getLogger(__name__)


def _save_dataset_metadata(dataset_id: str, info: dict) -> None:
    """Persist dataset info alongside dataset files so it survives restarts."""
    try:
        meta_path = settings.DATASETS_DIR / dataset_id / settings.METADATA_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(info, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not save metadata for dataset %s: %s", dataset_id, exc)


def _make_dataset_folder(name: str) -> Tuple[str, Path]:
    """Return (folder_name, full_path) using the dataset name.
    Sanitizes the name and appends _2, _3 … to avoid collisions."""
    safe = re.sub(r'[^\w\-.]', '_', name).strip('_') or "dataset"
    candidate = safe
    counter = 2
    while (settings.DATASETS_DIR / candidate).exists():
        candidate = f"{safe}_{counter}"
        counter += 1
    return candidate, settings.DATASETS_DIR / candidate


def _restore_datasets() -> None:
    """On startup: scan workspace/datasets/ for metadata sidecars and re-register datasets."""
    if not settings.DATASETS_DIR.exists():
        return
    restored = 0
    for entry in settings.DATASETS_DIR.iterdir():
        if not entry.is_dir():
            continue
        dataset_id = entry.name
        meta_path = entry / settings.METADATA_FILENAME
        try:
            if meta_path.exists():
                with open(meta_path) as f:
                    info = json.load(f)
                info["id"] = dataset_id
                settings.active_datasets[dataset_id] = info
                restored += 1
            else:
                info = settings.dataset_parser.parse_dataset(entry, name=entry.name)
                info["id"] = dataset_id
                settings.active_datasets[dataset_id] = info
                _save_dataset_metadata(dataset_id, info)
                restored += 1
        except Exception as exc:
            logger.warning("[startup] Skipping dataset %s — could not load: %s", dataset_id, exc)
    if restored:
        print(f"[startup] Restored {restored} dataset(s) from workspace.")
