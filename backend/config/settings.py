"""
Global configuration: path constants, singleton service instances, and shared mutable state.

All mutable dicts/lists are defined here so every router accesses the same objects.
Routers import this module as:   from config import settings
then reference globals as:       settings.active_datasets, settings.DATASETS_DIR, etc.
This ensures that reassignments (e.g. settings.DATASETS_DIR = new_path) are visible
across all modules without needing to re-import.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
# Anchored to this file so paths work regardless of the process working directory.
WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"
DATASETS_DIR  = WORKSPACE_DIR / "datasets"
MODELS_DIR    = WORKSPACE_DIR / "models"
EXPORTS_DIR   = WORKSPACE_DIR / "exports"
SNAPSHOTS_DIR = WORKSPACE_DIR / "snapshots"
TEMP_DIR      = WORKSPACE_DIR / "temp"
VIDEOS_DIR    = WORKSPACE_DIR / "videos"
JOBS_FILE     = WORKSPACE_DIR / "batch_jobs.json"

METADATA_FILENAME = "dataset_metadata.json"

# Create directories on import
for _d in [WORKSPACE_DIR, DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, SNAPSHOTS_DIR, TEMP_DIR, VIDEOS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Service singletons ────────────────────────────────────────────────────────
from dataset_parsers import DatasetParser
from format_converter import FormatConverter
from annotation_tools import AnnotationManager
from model_integration import ModelManager
from training import TrainingManager
from dataset_merger import DatasetMerger
from video_utils import VideoFrameExtractor, DuplicateDetector, CLIPEmbeddingManager
from augmentation import DatasetAugmenter

dataset_parser    = DatasetParser()
format_converter  = FormatConverter()
annotation_manager = AnnotationManager()
model_manager     = ModelManager(MODELS_DIR)
training_manager  = TrainingManager(jobs_file=WORKSPACE_DIR / "training_jobs.json")
dataset_merger    = DatasetMerger()
video_extractor   = VideoFrameExtractor()
duplicate_detector = DuplicateDetector()
clip_manager      = CLIPEmbeddingManager()
augmenter         = DatasetAugmenter()

# ── Shared mutable state ──────────────────────────────────────────────────────
active_datasets: Dict[str, Any] = {}
sorting_sessions: Dict[str, Dict] = {}
annotation_history: Dict[str, List[Dict]] = {}
_images_cache: Dict[str, List] = {}
_download_status: Dict[str, Dict] = {}
_uploaded_videos: Dict[str, Dict[str, Any]] = {}

_gpu_status: dict = {
    "state": "unknown",
    "message": "",
    "gpu_name": "",
}

_text_annotate_jobs: Dict[str, Dict] = {}
_job_controls: Dict[str, Dict] = {}
_extraction_jobs: Dict[str, Dict] = {}
_augmentation_jobs: Dict[str, Dict] = {}

_app_settings: dict = {
    "models_path":   str(MODELS_DIR),
    "datasets_path": str(DATASETS_DIR),
    "output_path":   str(EXPORTS_DIR),
    "use_gpu":       True,
    "gpu_device":    "0",
}
