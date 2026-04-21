# ── CHANGES vs PREVIOUS VERSION ───────────────────────────────────────────────
# • list_models(): added "pretrained" key to the loaded-models dict so the
#   frontend can correctly exclude newly-downloaded pretrained models from the
#   "loaded files" section (previously the key was missing → treated as falsy).
# • list_models(): changed name fallback from .get("name", "Unknown") to
#   `model_info.get("name") or "Unknown"` so an explicit None is also caught.
# ──────────────────────────────────────────────────────────────────────────────
"""
Model Integration - Load and use external models for auto-annotation
"""

import os
import json
import ssl
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
from PIL import Image
import yaml


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that validates certificates correctly on macOS.

    On macOS, Python installed from python.org ships without the system CA
    bundle wired in.  certifi provides a bundled CA store that works
    everywhere; we fall back to the default context if certifi is absent.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class ModelManager:
    """Manage external models for auto-annotation"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
    
    # Legacy Meta-format checkpoint stems — incompatible with ultralytics, hidden from UI
    _SAM_LEGACY_STEMS = {
        "sam_vit_b_01ec64", "sam_vit_l_0b3195", "sam_vit_h_4b8939",
        "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large",
    }

    # Full pretrained catalog — single source of truth used by list_models()
    _PRETRAINED_CATALOG = [
        # YOLOv8
        {"id": "yolov8n",  "name": "YOLOv8 Nano",   "type": "yolo", "pretrained": True},
        {"id": "yolov8s",  "name": "YOLOv8 Small",  "type": "yolo", "pretrained": True},
        {"id": "yolov8m",  "name": "YOLOv8 Medium", "type": "yolo", "pretrained": True},
        {"id": "yolov8l",  "name": "YOLOv8 Large",  "type": "yolo", "pretrained": True},
        {"id": "yolov8x",  "name": "YOLOv8 XLarge", "type": "yolo", "pretrained": True},
        # YOLOv5
        {"id": "yolov5n",  "name": "YOLOv5 Nano",   "type": "yolo", "pretrained": True},
        {"id": "yolov5s",  "name": "YOLOv5 Small",  "type": "yolo", "pretrained": True},
        # YOLOv9
        {"id": "yolov9n",  "name": "YOLOv9 Nano",     "type": "yolo", "pretrained": True},
        {"id": "yolov9s",  "name": "YOLOv9 Small",    "type": "yolo", "pretrained": True},
        {"id": "yolov9m",  "name": "YOLOv9 Medium",   "type": "yolo", "pretrained": True},
        {"id": "yolov9c",  "name": "YOLOv9 Compact",  "type": "yolo", "pretrained": True},
        {"id": "yolov9e",  "name": "YOLOv9 Extended", "type": "yolo", "pretrained": True},
        # YOLOv10
        {"id": "yolov10n", "name": "YOLOv10 Nano",     "type": "yolo", "pretrained": True},
        {"id": "yolov10s", "name": "YOLOv10 Small",    "type": "yolo", "pretrained": True},
        {"id": "yolov10m", "name": "YOLOv10 Medium",   "type": "yolo", "pretrained": True},
        {"id": "yolov10b", "name": "YOLOv10 Balanced", "type": "yolo", "pretrained": True},
        {"id": "yolov10l", "name": "YOLOv10 Large",    "type": "yolo", "pretrained": True},
        {"id": "yolov10x", "name": "YOLOv10 XLarge",   "type": "yolo", "pretrained": True},
        # YOLO11
        {"id": "yolo11n",  "name": "YOLO11 Nano",   "type": "yolo", "pretrained": True},
        {"id": "yolo11s",  "name": "YOLO11 Small",  "type": "yolo", "pretrained": True},
        {"id": "yolo11m",  "name": "YOLO11 Medium", "type": "yolo", "pretrained": True},
        {"id": "yolo11l",  "name": "YOLO11 Large",  "type": "yolo", "pretrained": True},
        {"id": "yolo11x",  "name": "YOLO11 XLarge", "type": "yolo", "pretrained": True},
        # YOLO12
        {"id": "yolo12n",  "name": "YOLO12 Nano",   "type": "yolo", "pretrained": True},
        {"id": "yolo12s",  "name": "YOLO12 Small",  "type": "yolo", "pretrained": True},
        {"id": "yolo12m",  "name": "YOLO12 Medium", "type": "yolo", "pretrained": True},
        {"id": "yolo12l",  "name": "YOLO12 Large",  "type": "yolo", "pretrained": True},
        {"id": "yolo12x",  "name": "YOLO12 XLarge", "type": "yolo", "pretrained": True},
        # RT-DETR (ultralytics)
        {"id": "rtdetr-l", "name": "RT-DETR Large",  "type": "rtdetr", "pretrained": True},
        {"id": "rtdetr-x", "name": "RT-DETR XLarge", "type": "rtdetr", "pretrained": True},
        {"id": "sam_vit_b",   "name": "SAM Base",      "type": "sam",   "pretrained": True},
        {"id": "sam_vit_l",   "name": "SAM Large",     "type": "sam",   "pretrained": True},
        {"id": "sam2_tiny",   "name": "SAM 2 Tiny",    "type": "sam2",  "pretrained": True},
        {"id": "sam2_small",  "name": "SAM 2 Small",   "type": "sam2",  "pretrained": True},
        {"id": "sam2_base",   "name": "SAM 2 Base+",   "type": "sam2",  "pretrained": True},
        {"id": "sam2_large",  "name": "SAM 2 Large",   "type": "sam2",  "pretrained": True},
        {"id": "sam21_tiny",  "name": "SAM 2.1 Tiny",  "type": "sam2",  "pretrained": True},
        {"id": "sam21_small", "name": "SAM 2.1 Small", "type": "sam2",  "pretrained": True},
        {"id": "sam21_base",  "name": "SAM 2.1 Base+", "type": "sam2",  "pretrained": True},
        {"id": "sam21_large", "name": "SAM 2.1 Large", "type": "sam2",  "pretrained": True},
        {"id": "sam3",        "name": "SAM 3",         "type": "sam3",  "pretrained": True},
        # RF-DETR — Roboflow real-time detection transformer
        {"id": "rfdetr_base",  "name": "RF-DETR Base",  "type": "rfdetr", "pretrained": True},
        {"id": "rfdetr_large", "name": "RF-DETR Large", "type": "rfdetr", "pretrained": True},
        # Zero-shot open-vocabulary models
        {"id": "yoloworld_s", "name": "YOLO-World S (zero-shot)", "type": "yoloworld", "pretrained": True},
        {"id": "yoloworld_m", "name": "YOLO-World M (zero-shot)", "type": "yoloworld", "pretrained": True},
        {"id": "yoloworld_l", "name": "YOLO-World L (zero-shot)", "type": "yoloworld", "pretrained": True},
        {"id": "groundingdino_t", "name": "GroundingDINO Tiny (zero-shot)", "type": "groundingdino", "pretrained": True},
        {"id": "groundingdino_b", "name": "GroundingDINO Base (zero-shot)", "type": "groundingdino", "pretrained": True},
        # OWL-ViT — open-vocabulary detection via image-text matching
        {"id": "owlvit_base_patch32", "name": "OWL-ViT Base (Patch32) (zero-shot)", "type": "owlvit", "pretrained": True},
        {"id": "owlvit_base_patch16", "name": "OWL-ViT Base (Patch16) (zero-shot)", "type": "owlvit", "pretrained": True},
        {"id": "owlvit_large_patch14", "name": "OWL-ViT Large (Patch14) (zero-shot)", "type": "owlvit", "pretrained": True},
    ]

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        seen_ids: set = set()

        # Build reverse map: file-stem → catalog entry
        # e.g. "yolo11n" → catalog entry for "yolov11n"
        stem_to_catalog: Dict[str, Dict] = {}
        for url_dict in (self._YOLO_URLS, self._SAM_URLS):
            for entry in self._PRETRAINED_CATALOG:
                url_entry = url_dict.get(entry["id"])
                if url_entry:
                    stem = Path(url_entry[0]).stem
                    stem_to_catalog[stem] = entry
        # RF-DETR filenames (downloaded to workspace/models/ via chdir)
        for rfdetr_id, rfdetr_stem in self._RFDETR_FILENAMES.items():
            catalog = next((e for e in self._PRETRAINED_CATALOG if e["id"] == rfdetr_id), None)
            if catalog:
                stem_to_catalog[Path(rfdetr_stem).stem] = catalog

        # 1. In-memory loaded models (highest priority)
        for model_id, info in self.loaded_models.items():
            seen_ids.add(model_id)
            models.append({
                "id": model_id,
                "name": info.get("name") or model_id,
                "type": info.get("type", "unknown"),
                "loaded": True,
                "pretrained": info.get("pretrained", False),
                "classes": info.get("classes", []),
                "error": info.get("error"),
                "downloaded": True,
            })

        # 2. Files present on disk in workspace/models/
        for model_file in sorted(self.models_dir.glob("*")):
            if model_file.suffix not in {".pt", ".pth", ".onnx", ".weights"}:
                continue
            stem = model_file.stem
            # Skip legacy Meta-format SAM checkpoints — not loadable by ultralytics
            if stem in self._SAM_LEGACY_STEMS:
                continue
            # Map file stem back to catalog ID if possible
            catalog = stem_to_catalog.get(stem)
            model_id = catalog["id"] if catalog else stem
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            models.append({
                "id": model_id,
                "name": catalog["name"] if catalog else model_file.name,
                "type": catalog["type"] if catalog else self._detect_model_type(model_file),
                "loaded": False,
                "pretrained": bool(catalog),
                "downloaded": True,
                "path": str(model_file),
            })

        # 2b. HuggingFace-cached models (GroundingDINO, OWL-ViT)
        #     (these are never stored as .pt files in models_dir, so the disk
        #     scan above misses them entirely)
        try:
            from huggingface_hub import scan_cache_info
            cached_repos = {repo.repo_id for repo in scan_cache_info().repos}
            _hf_model_maps = [
                (self._GROUNDINGDINO_HF_IDS, "groundingdino"),
                (self._OWLVIT_HF_IDS, "owlvit"),
            ]
            for hf_ids_dict, mtype in _hf_model_maps:
                for model_id, hf_id in hf_ids_dict.items():
                    if model_id in seen_ids:
                        continue
                    # Also check local hf_cache directory used by snapshot_download
                    local_cache = self.models_dir / "hf_cache" / hf_id.replace("/", "--")
                    is_local = local_cache.exists() and any(local_cache.iterdir())
                    if hf_id in cached_repos or is_local:
                        catalog = next(
                            (e for e in self._PRETRAINED_CATALOG if e["id"] == model_id), None
                        )
                        seen_ids.add(model_id)
                        models.append({
                            "id": model_id,
                            "name": catalog["name"] if catalog else model_id,
                            "type": mtype,
                            "loaded": False,
                            "pretrained": True,
                            "downloaded": True,
                        })
        except Exception:
            pass

        # 3. Pretrained catalog entries not yet downloaded
        for entry in self._PRETRAINED_CATALOG:
            if entry["id"] not in seen_ids:
                models.append({**entry, "loaded": False, "downloaded": False})

        return models
    
    def _detect_model_type(self, model_path: Path) -> str:
        """Detect model type from file"""
        name = model_path.name.lower()
        
        if "world" in name and "yolo" in name:
            return "yoloworld"
        elif "yolo" in name:
            return "yolo"
        elif "sam" in name:
            return "sam"
        elif "rfdetr" in name or "rf-detr" in name or "rf_detr" in name:
            return "rfdetr"
        elif "detr" in name:
            return "rfdetr"  # assume RF-DETR for generic detr files
        elif "rcnn" in name:
            return "rcnn"
        else:
            return "unknown"
    
    def load_model(self, model_path: str, model_type: str) -> str:
        """Load a model from file"""
        model_path = Path(model_path)
        model_id = str(uuid.uuid4())[:8]
        
        model_info = {
            "id": model_id,
            "name": model_path.name,
            "type": model_type,
            "path": str(model_path),
            "model": None,
            "classes": []
        }
        
        try:
            if model_type == "yolo":
                model_info = self._load_yolo_model(model_path, model_info)
            elif model_type in ["sam", "sam2", "sam3"]:
                model_info = self._load_sam_model(model_path, model_info)
            elif model_type == "rfdetr":
                model_info = self._load_rfdetr_from_checkpoint(model_path, model_info)
            elif model_type == "yoloworld":
                try:
                    from ultralytics import YOLOWorld
                    m = YOLOWorld(str(model_path))
                    self._apply_device_to_model(m)
                    model_info["model"] = m
                    model_info["classes"] = []
                except Exception as e:
                    model_info["error"] = str(e)
            else:
                model_info["model"] = None
        except Exception as e:
            model_info["error"] = str(e)
        
        self.loaded_models[model_id] = model_info
        return model_id
    
    def load_pretrained(self, model_type: str, model_name: str, hf_token: Optional[str] = None,
                        _status_hook=None) -> str:
        """Load a pretrained model. _status_hook(phase: str) is called at key loading phases."""
        model_id = model_name
        model_info = {
            "id": model_id,
            "name": model_name,
            "type": model_type,
            "pretrained": True,
            "loaded": True,
            "model": None,
            "classes": []
        }
        try:
            if model_type in ("yolo", "rtdetr"):
                model_info = self._load_pretrained_yolo(model_name, model_info, _status_hook=_status_hook)
            elif model_type in ["sam", "sam2", "sam3"]:
                model_info = self._load_pretrained_sam(model_name, model_info, hf_token=hf_token,
                                                       _status_hook=_status_hook)
            elif model_type == "rfdetr":
                model_info = self._load_pretrained_rfdetr(model_name, model_info)
            elif model_type == "yoloworld":
                model_info = self._load_pretrained_yoloworld(model_name, model_info)
            elif model_type == "groundingdino":
                model_info = self._load_pretrained_groundingdino(model_name, model_info)
            elif model_type == "owlvit":
                model_info = self._load_pretrained_owlvit(model_name, model_info)
        except Exception as e:
            model_info["error"] = str(e)
        self.loaded_models[model_id] = model_info
        return model_id
    
    def _load_yolo_model(self, model_path: Path, model_info: Dict) -> Dict:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            self._apply_device_to_model(model)
            model_info["model"] = model
            model_info["classes"] = list(model.names.values()) if hasattr(model, 'names') else []
        except ImportError:
            model_info["error"] = "ultralytics package not installed"

        return model_info
    
    # Direct download URLs for pretrained SAM / SAM2 models
    # Using ultralytics-packaged weights — compatible with `from ultralytics import SAM`
    _SAM_URLS = {
        # SAM (v1)
        "sam_vit_b":   ("sam_b.pt",      "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt"),
        "sam_vit_l":   ("sam_l.pt",      "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt"),
        # SAM 2
        "sam2_tiny":   ("sam2_t.pt",     "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt"),
        "sam2_small":  ("sam2_s.pt",     "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_s.pt"),
        "sam2_base":   ("sam2_b.pt",     "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt"),
        "sam2_large":  ("sam2_l.pt",     "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt"),
        # SAM 2.1
        "sam21_tiny":  ("sam2.1_t.pt",   "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt"),
        "sam21_small": ("sam2.1_s.pt",   "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt"),
        "sam21_base":  ("sam2.1_b.pt",   "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt"),
        "sam21_large": ("sam2.1_l.pt",   "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt"),
        # SAM 3
        "sam3":        ("sam3.pt",       "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true"),
    }

    # Direct download URLs for pretrained YOLO/RT-DETR models
    _BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
    _YOLO_URLS = {
        # YOLOv8
        "yolov8n":  ("yolov8n.pt",  f"{_BASE}/yolov8n.pt"),
        "yolov8s":  ("yolov8s.pt",  f"{_BASE}/yolov8s.pt"),
        "yolov8m":  ("yolov8m.pt",  f"{_BASE}/yolov8m.pt"),
        "yolov8l":  ("yolov8l.pt",  f"{_BASE}/yolov8l.pt"),
        "yolov8x":  ("yolov8x.pt",  f"{_BASE}/yolov8x.pt"),
        # YOLOv5 (ultralytics repack)
        "yolov5n":  ("yolov5nu.pt", f"{_BASE}/yolov5nu.pt"),
        "yolov5s":  ("yolov5su.pt", f"{_BASE}/yolov5su.pt"),
        # YOLOv9
        "yolov9n":  ("yolov9n.pt",  f"{_BASE}/yolov9n.pt"),
        "yolov9s":  ("yolov9s.pt",  f"{_BASE}/yolov9s.pt"),
        "yolov9m":  ("yolov9m.pt",  f"{_BASE}/yolov9m.pt"),
        "yolov9c":  ("yolov9c.pt",  f"{_BASE}/yolov9c.pt"),
        "yolov9e":  ("yolov9e.pt",  f"{_BASE}/yolov9e.pt"),
        # YOLOv10
        "yolov10n": ("yolov10n.pt", f"{_BASE}/yolov10n.pt"),
        "yolov10s": ("yolov10s.pt", f"{_BASE}/yolov10s.pt"),
        "yolov10m": ("yolov10m.pt", f"{_BASE}/yolov10m.pt"),
        "yolov10b": ("yolov10b.pt", f"{_BASE}/yolov10b.pt"),
        "yolov10l": ("yolov10l.pt", f"{_BASE}/yolov10l.pt"),
        "yolov10x": ("yolov10x.pt", f"{_BASE}/yolov10x.pt"),
        # YOLO11
        "yolo11n":  ("yolo11n.pt",  f"{_BASE}/yolo11n.pt"),
        "yolo11s":  ("yolo11s.pt",  f"{_BASE}/yolo11s.pt"),
        "yolo11m":  ("yolo11m.pt",  f"{_BASE}/yolo11m.pt"),
        "yolo11l":  ("yolo11l.pt",  f"{_BASE}/yolo11l.pt"),
        "yolo11x":  ("yolo11x.pt",  f"{_BASE}/yolo11x.pt"),
        # YOLO12
        "yolo12n":  ("yolo12n.pt",  f"{_BASE}/yolo12n.pt"),
        "yolo12s":  ("yolo12s.pt",  f"{_BASE}/yolo12s.pt"),
        "yolo12m":  ("yolo12m.pt",  f"{_BASE}/yolo12m.pt"),
        "yolo12l":  ("yolo12l.pt",  f"{_BASE}/yolo12l.pt"),
        "yolo12x":  ("yolo12x.pt",  f"{_BASE}/yolo12x.pt"),
        # RT-DETR (ultralytics)
        "rtdetr-l": ("rtdetr-l.pt", f"{_BASE}/rtdetr-l.pt"),
        "rtdetr-x": ("rtdetr-x.pt", f"{_BASE}/rtdetr-x.pt"),
        # YOLO-World v2 — zero-shot open-vocabulary detection
        "yoloworld_s": ("yolov8s-worldv2.pt", f"{_BASE}/yolov8s-worldv2.pt"),
        "yoloworld_m": ("yolov8m-worldv2.pt", f"{_BASE}/yolov8m-worldv2.pt"),
        "yoloworld_l": ("yolov8l-worldv2.pt", f"{_BASE}/yolov8l-worldv2.pt"),
    }
    del _BASE  # class-level cleanup

    # RF-DETR expected filenames in workspace/models/
    _RFDETR_FILENAMES = {
        "rfdetr_base":  "rf-detr-base.pth",
        "rfdetr_large": "rf-detr-large.pth",
    }

    # GroundingDINO HuggingFace model IDs (downloaded on first use via transformers)
    _GROUNDINGDINO_HF_IDS = {
        "groundingdino_t": "IDEA-Research/grounding-dino-tiny",
        "groundingdino_b": "IDEA-Research/grounding-dino-base",
    }

    # OWL-ViT HuggingFace model IDs
    _OWLVIT_HF_IDS = {
        "owlvit_base_patch32":  "google/owlvit-base-patch32",
        "owlvit_base_patch16":  "google/owlvit-base-patch16",
        "owlvit_large_patch14": "google/owlvit-large-patch14",
    }

    def _load_pretrained_yolo(self, model_name: str, model_info: Dict, _status_hook=None) -> Dict:
        """Load pretrained YOLO model — downloads directly to workspace/models/."""
        import sys
        import subprocess
        import urllib.request

        # Restore venv sys.path in case this is called from a thread
        try:
            from training import _VENV_SYSPATH
            for p in _VENV_SYSPATH:
                if p not in sys.path:
                    sys.path.insert(0, p)
        except ImportError:
            pass

        entry = self._YOLO_URLS.get(model_name)
        if entry is None:
            model_info["error"] = f"Unknown YOLO model: {model_name}"
            model_info["loaded"] = False
            return model_info

        model_file, download_url = entry
        local_model_path = self.models_dir / model_file

        # Download directly to workspace/models/ if not already present
        if not local_model_path.exists():
            try:
                req = urllib.request.Request(
                    download_url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as resp, \
                        open(local_model_path, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)  # 64 KB
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as dl_err:
                # Clean up partial download
                if local_model_path.exists():
                    local_model_path.unlink()
                model_info["error"] = f"Download failed: {dl_err}"
                model_info["loaded"] = False
                return model_info

        # File is on disk — now load into memory
        if _status_hook:
            _status_hook("loading_memory")

        # Try to load with ultralytics (optional — gives class names)
        try:
            from ultralytics import YOLO
            model = YOLO(str(local_model_path))
            self._apply_device_to_model(model)
            model_info["model"] = model
            model_info["classes"] = list(model.names.values()) if hasattr(model, "names") else []
        except Exception:
            # Still mark as available even without ultralytics loaded in memory
            model_info["model"] = None
            model_info["classes"] = []

        model_info["loaded"] = True
        model_info["path"] = str(local_model_path)
        return model_info
    
    def _load_sam_model(self, model_path: Path, model_info: Dict) -> Dict:
        """Load a custom SAM model file — tries ultralytics first, then native packages."""
        loaded = self._try_load_sam_ultralytics(str(model_path), model_info)
        if loaded:
            return model_info
        # Fall back to segment_anything / sam2 packages
        try:
            from segment_anything import SamPredictor, sam_model_registry
            name = str(model_path).lower()
            if "vit_h" in name:
                vit = "vit_h"
            elif "vit_l" in name:
                vit = "vit_l"
            else:
                vit = "vit_b"
            sam = sam_model_registry[vit](checkpoint=str(model_path))
            model_info["model"] = SamPredictor(sam)
            model_info["sam_type"] = vit
            model_info["sam_backend"] = "native"
        except ImportError:
            model_info["error"] = "SAM package not installed (ultralytics also failed)"
        return model_info

    def _try_load_sam_ultralytics(self, model_path: str, model_info: Dict) -> bool:
        """Try loading a SAM/SAM2/SAM2.1/SAM3 checkpoint via ultralytics. Returns True on success."""
        try:
            name = Path(model_path).name.lower()
            if "sam3" in name:
                try:
                    from ultralytics.models.sam import SAM3
                    model = SAM3(model_path)
                    model_info["sam_backend"] = "ultralytics-sam3"
                except Exception:
                    from ultralytics import SAM
                    model = SAM(model_path)
                    model_info["sam_backend"] = "ultralytics"
            elif "sam2" in name:
                try:
                    from ultralytics.models.sam import SAM2
                    model = SAM2(model_path)
                    model_info["sam_backend"] = "ultralytics-sam2"
                except Exception:
                    from ultralytics import SAM
                    model = SAM(model_path)
                    model_info["sam_backend"] = "ultralytics"
            else:
                from ultralytics import SAM
                model = SAM(model_path)
                model_info["sam_backend"] = "ultralytics"
            self._apply_device_to_model(model)
            model_info["model"] = model
            return True
        except Exception:
            return False

    # Models that require HuggingFace authentication
    _HF_GATED_MODELS = {
        "sam3": ("facebook/sam3", "sam3.pt"),
    }

    def _load_pretrained_sam(self, model_name: str, model_info: Dict, hf_token: Optional[str] = None,
                             _status_hook=None) -> Dict:
        """Download SAM/SAM2/SAM3 checkpoint to workspace/models/, then load via ultralytics."""
        import urllib.request

        entry = self._SAM_URLS.get(model_name)
        if entry is None:
            model_info["error"] = f"Unknown SAM model: {model_name}"
            model_info["loaded"] = False
            return model_info

        model_file, download_url = entry
        local_model_path = self.models_dir / model_file

        # Clean up any legacy Meta-format files for this model slot
        for legacy_stem in self._SAM_LEGACY_STEMS:
            for ext in (".pt", ".pth"):
                legacy_path = self.models_dir / f"{legacy_stem}{ext}"
                if legacy_path.exists():
                    try:
                        legacy_path.unlink()
                    except OSError:
                        pass

        # Download file if not already present
        if not local_model_path.exists():
            hf_gated = self._HF_GATED_MODELS.get(model_name)

            if hf_gated:
                # Gated HuggingFace model — requires huggingface_hub + token
                repo_id, filename = hf_gated
                if not hf_token:
                    model_info["error"] = (
                        f"{model_name} is a gated HuggingFace model. "
                        "Provide a HuggingFace token (hf.co/settings/tokens) in the Models page to download it."
                    )
                    model_info["loaded"] = False
                    return model_info
                try:
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        import subprocess, sys
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"],
                            check=True,
                        )
                        from huggingface_hub import hf_hub_download
                    dl_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        token=hf_token,
                        local_dir=str(self.models_dir),
                    )
                    # hf_hub_download may save to a cache subfolder; copy to models_dir root
                    dl_path = Path(dl_path)
                    if dl_path != local_model_path:
                        import shutil
                        shutil.copy2(dl_path, local_model_path)
                except Exception as dl_err:
                    if local_model_path.exists():
                        local_model_path.unlink()
                    model_info["error"] = f"HuggingFace download failed: {dl_err}"
                    model_info["loaded"] = False
                    return model_info
            else:
                # Regular public download via urllib
                try:
                    req = urllib.request.Request(
                        download_url,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=600, context=_ssl_context()) as resp, \
                            open(local_model_path, "wb") as f:
                        while True:
                            chunk = resp.read(1 << 16)  # 64 KB
                            if not chunk:
                                break
                            f.write(chunk)
                except Exception as dl_err:
                    if local_model_path.exists():
                        local_model_path.unlink()
                    model_info["error"] = f"Download failed: {dl_err}"
                    model_info["loaded"] = False
                    return model_info

        model_info["path"] = str(local_model_path)
        model_info["loaded"] = True

        # File is on disk — now load into memory (can be slow for large SAM models)
        if _status_hook:
            _status_hook("loading_memory")

        # Try ultralytics SAM first (no extra packages needed)
        if self._try_load_sam_ultralytics(str(local_model_path), model_info):
            return model_info

        # Fall back to native sam2 / segment_anything packages
        try:
            if "sam2" in model_name:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                config = model_file.replace(".pt", ".yaml")
                sam2_model = build_sam2(config, str(local_model_path))
                model_info["model"] = SAM2ImagePredictor(sam2_model)
                model_info["sam_type"] = "sam2"
                model_info["sam_backend"] = "native"
            else:
                from segment_anything import SamPredictor, sam_model_registry
                vit_map = {"sam_vit_b": "vit_b", "sam_vit_l": "vit_l", "sam_vit_h": "vit_h"}
                vit = vit_map.get(model_name, "vit_b")
                sam = sam_model_registry[vit](checkpoint=str(local_model_path))
                model_info["model"] = SamPredictor(sam)
                model_info["sam_type"] = vit
                model_info["sam_backend"] = "native"
        except ImportError as e:
            # File is on disk; package just isn't installed
            model_info["model"] = None
            model_info["error"] = f"Package not installed (model file saved): {e}"

        return model_info

    def _load_pretrained_yoloworld(self, model_name: str, model_info: Dict) -> Dict:
        """Load a pretrained YOLO-World v2 zero-shot detection model."""
        import sys, urllib.request
        try:
            from training import _VENV_SYSPATH
            for p in _VENV_SYSPATH:
                if p not in sys.path:
                    sys.path.insert(0, p)
        except ImportError:
            pass

        entry = self._YOLO_URLS.get(model_name)
        if entry is None:
            model_info["error"] = f"Unknown YOLO-World model: {model_name}"
            model_info["loaded"] = False
            return model_info

        model_file, download_url = entry
        local_path = self.models_dir / model_file

        if not local_path.exists():
            try:
                req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as resp, open(local_path, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as dl_err:
                if local_path.exists():
                    local_path.unlink()
                model_info["error"] = f"Download failed: {dl_err}"
                model_info["loaded"] = False
                return model_info

        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(str(local_path))
            self._apply_device_to_model(model)
            model_info["model"] = model
            model_info["classes"] = []  # zero-shot: classes set via text prompt
            model_info["loaded"] = True
            model_info["path"] = str(local_path)
        except Exception as e:
            model_info["error"] = str(e)
            model_info["loaded"] = False
        return model_info

    def _load_pretrained_groundingdino(self, model_name: str, model_info: Dict) -> Dict:
        """Load GroundingDINO via HuggingFace transformers (zero-shot text-grounded detection)."""
        import sys, subprocess
        try:
            from training import _VENV_SYSPATH
            for p in _VENV_SYSPATH:
                if p not in sys.path:
                    sys.path.insert(0, p)
        except ImportError:
            pass

        hf_id = self._GROUNDINGDINO_HF_IDS.get(model_name)
        if hf_id is None:
            model_info["error"] = f"Unknown GroundingDINO model: {model_name}"
            model_info["loaded"] = False
            return model_info

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "transformers", "-q"], timeout=300
                )
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            except Exception as e:
                model_info["error"] = f"Failed to install transformers: {e}"
                model_info["loaded"] = False
                return model_info

        try:
            import os
            # On Windows, huggingface_hub tries to create symlinks which fails without
            # Developer Mode enabled, causing [Errno 22] Invalid argument. Fix: download
            # to a local directory with symlinks disabled, then load from that path.
            local_model_dir = self.models_dir / "hf_cache" / hf_id.replace("/", "--")
            local_model_dir.mkdir(parents=True, exist_ok=True)
            local_dir_str = str(local_model_dir)

            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    hf_id,
                    local_dir=local_dir_str,
                    local_dir_use_symlinks=False,
                )
                load_path = local_dir_str
            except TypeError:
                # Older huggingface_hub without local_dir_use_symlinks — fall back to cache_dir
                load_path = hf_id
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            except Exception:
                # snapshot_download failed; try loading directly (may work if already cached)
                load_path = hf_id
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            processor = AutoProcessor.from_pretrained(load_path)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(load_path)
            # Store processor as attribute on model for use in _run_inference
            model._grounding_processor = processor
            try:
                import torch
                device = self._get_device()
                if device != "cpu":
                    model = model.to(f"cuda:{device}" if isinstance(device, int) else device)
            except Exception:
                pass
            model_info["model"] = model
            model_info["classes"] = []  # zero-shot
            model_info["loaded"] = True
        except Exception as e:
            model_info["error"] = str(e)
            model_info["loaded"] = False
        return model_info

    def _load_pretrained_owlvit(self, model_name: str, model_info: Dict) -> Dict:
        """Load OWL-ViT via HuggingFace transformers (zero-shot text-grounded detection)."""
        import sys, subprocess
        try:
            from training import _VENV_SYSPATH
            for p in _VENV_SYSPATH:
                if p not in sys.path:
                    sys.path.insert(0, p)
        except ImportError:
            pass

        hf_id = self._OWLVIT_HF_IDS.get(model_name)
        if hf_id is None:
            model_info["error"] = f"Unknown OWL-ViT model: {model_name}"
            model_info["loaded"] = False
            return model_info

        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "transformers", "-q"], timeout=300
                )
                from transformers import OwlViTProcessor, OwlViTForObjectDetection
            except Exception as e:
                model_info["error"] = f"Failed to install transformers: {e}"
                model_info["loaded"] = False
                return model_info

        try:
            import os
            local_model_dir = self.models_dir / "hf_cache" / hf_id.replace("/", "--")
            local_model_dir.mkdir(parents=True, exist_ok=True)
            local_dir_str = str(local_model_dir)

            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    hf_id,
                    local_dir=local_dir_str,
                    local_dir_use_symlinks=False,
                )
                load_path = local_dir_str
            except TypeError:
                load_path = hf_id
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            except Exception:
                load_path = hf_id
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            processor = OwlViTProcessor.from_pretrained(load_path)
            model = OwlViTForObjectDetection.from_pretrained(load_path)
            model._owlvit_processor = processor
            try:
                import torch
                device = self._get_device()
                if device != "cpu":
                    model = model.to(f"cuda:{device}" if isinstance(device, int) else device)
            except Exception:
                pass
            model_info["model"] = model
            model_info["classes"] = []  # zero-shot
            model_info["loaded"] = True
        except Exception as e:
            model_info["error"] = str(e)
            model_info["loaded"] = False
        return model_info

    def _load_rfdetr_from_checkpoint(self, model_path: Path, model_info: Dict) -> Dict:
        """Load a locally trained or pretrained RF-DETR checkpoint (.pth) for inference."""
        import sys, subprocess
        try:
            import rfdetr as _r  # noqa: F401
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 'rfdetr[train,loggers]', "-q"], timeout=300)
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
            # Choose the right class based on the filename
            name_lower = str(model_path).lower()
            is_large = "large" in name_lower
            cls = RFDETRLarge if is_large else RFDETRBase
            model = cls(pretrain_weights=str(model_path))
            model_info["model"]      = model
            model_info["loaded"]     = True
            model_info["downloaded"] = True
        except Exception as e:
            model_info["error"]  = str(e)
            model_info["loaded"] = False
        return model_info

    def _load_pretrained_rfdetr(self, model_name: str, model_info: Dict) -> Dict:
        """Install rfdetr if needed, then load RF-DETR Base or Large (weights auto-downloaded by rfdetr)."""
        import sys, subprocess
        # Restore venv sys.path when called from a background thread
        try:
            from training import _VENV_SYSPATH
            for p in _VENV_SYSPATH:
                if p not in sys.path:
                    sys.path.insert(0, p)
        except ImportError:
            pass

        try:
            import rfdetr as _rfdetr_pkg  # noqa: F401
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", 'rfdetr[train,loggers]', "-q"],
                timeout=300,
            )

        try:
            from rfdetr import RFDETRBase, RFDETRLarge
            # rfdetr downloads weights to cwd — chdir to models_dir so they land there
            _prev_cwd = os.getcwd()
            os.chdir(str(self.models_dir))
            try:
                if model_name == "rfdetr_large":
                    model = RFDETRLarge()
                    model_info["name"] = "RF-DETR Large"
                else:
                    model = RFDETRBase()
                    model_info["name"] = "RF-DETR Base"
            finally:
                os.chdir(_prev_cwd)
            model_info["model"] = model
            model_info["loaded"] = True
            model_info["downloaded"] = True
            # Ensure the downloaded .pth is stored under the canonical filename so
            # list_models() can find it on disk.  rfdetr may use a versioned name
            # (e.g. rf-detr-large-2026.pth) — rename to the canonical one.
            canonical = self._RFDETR_FILENAMES[model_name]
            canonical_path = self.models_dir / canonical
            if not canonical_path.exists():
                import glob as _glob
                pattern = str(self.models_dir / "rf-detr-large*.pth") if model_name == "rfdetr_large" else str(self.models_dir / "rf-detr-base*.pth")
                matches = [p for p in _glob.glob(pattern) if Path(p).name != canonical]
                if matches:
                    import shutil
                    shutil.copy2(matches[0], canonical_path)
            # RF-DETR is COCO-pretrained — 80 standard classes
            model_info["classes"] = [
                "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
                "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
                "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
                "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
                "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
                "clock","vase","scissors","teddy bear","hair drier","toothbrush",
            ]
        except Exception as e:
            model_info["error"] = str(e)
            model_info["loaded"] = False
        return model_info

    def _get_device(self):
        """Return GPU device index (0) if CUDA is available, else 'cpu'.
        Uses the same logic as training.py so inference and training share a device."""
        try:
            import torch
            if torch.cuda.is_available():
                return 0  # GPU 0 — ultralytics wants an int, not "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _apply_device_to_model(self, model) -> None:
        """Set the device override on an ultralytics model so all predict calls use it."""
        device = self._get_device()
        if device == "cpu":
            return  # CPU is the default; nothing to do
        try:
            if hasattr(model, "overrides"):
                model.overrides["device"] = device
            if hasattr(model, "predictor") and model.predictor is not None:
                if hasattr(model.predictor, "args"):
                    model.predictor.args.device = device
        except Exception:
            pass

    def auto_annotate(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        confidence_threshold: float = 0.5,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Auto-annotate all images in a dataset"""
        if model_id not in self.loaded_models:
            self._try_autoload(model_id)
        if model_id not in self.loaded_models:
            return {"error": "Model not loaded"}

        model_info = self.loaded_models[model_id]
        model = model_info.get("model")
        model_type = model_info.get("type")

        if not model:
            return {"error": "Model not available"}

        dataset_path = Path(dataset_path)
        results = {
            "annotated_count": 0,
            "failed_count": 0,
            "images": []
        }

        # Find all images
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))

        # Load class mapping
        classes = self._get_dataset_classes(dataset_path, format_name)

        for image_file in image_files:
            try:
                annotations = self._run_inference(
                    model, model_type, image_file, confidence_threshold,
                    text_prompt=text_prompt,
                    model_classes=model_info.get("classes") or [],
                )
                
                if annotations:
                    self._save_annotations(
                        dataset_path, format_name, image_file, annotations, classes
                    )
                    results["annotated_count"] += 1
                    results["images"].append({
                        "filename": image_file.name,
                        "annotations": len(annotations)
                    })
                
            except Exception as e:
                results["failed_count"] += 1
        
        return results
    
    def _try_autoload(self, model_id: str) -> None:
        """Load a model into memory on demand, given its catalog ID or file stem."""
        # Build combined stem→url map for all known pretrained models
        all_urls: dict = {**self._YOLO_URLS, **self._SAM_URLS}

        # 0. RF-DETR pretrained catalog — uses _RFDETR_FILENAMES, not _YOLO_URLS
        rfdetr_filename = self._RFDETR_FILENAMES.get(model_id)
        if rfdetr_filename:
            local_path = self.models_dir / rfdetr_filename
            if local_path.exists():
                info: Dict[str, Any] = {
                    "id": model_id, "name": rfdetr_filename,
                    "type": "rfdetr", "pretrained": True,
                    "path": str(local_path), "model": None, "classes": [],
                }
                self._load_rfdetr_from_checkpoint(local_path, info)
                self.loaded_models[model_id] = info
                return

        # 1. Try catalog entry (pretrained model with known URL)
        entry = all_urls.get(model_id)
        if entry:
            filename, _ = entry
            local_path = self.models_dir / filename
            if local_path.exists():
                mtype = "sam3"       if "sam3" in model_id else \
                        "sam2"       if "sam2" in model_id or model_id.startswith("sam21") else \
                        "sam"        if "sam"  in model_id else \
                        "yoloworld"  if "yoloworld" in model_id else "yolo"
                info: Dict[str, Any] = {
                    "id": model_id, "name": filename,
                    "type": mtype, "pretrained": True,
                    "path": str(local_path), "model": None, "classes": [],
                }
                if mtype in ("sam", "sam2", "sam3"):
                    self._try_load_sam_ultralytics(str(local_path), info)
                elif mtype == "yoloworld":
                    try:
                        from ultralytics import YOLOWorld
                        m = YOLOWorld(str(local_path))
                        self._apply_device_to_model(m)
                        info["model"] = m
                        info["classes"] = []
                    except Exception:
                        pass
                else:
                    try:
                        from ultralytics import YOLO
                        m = YOLO(str(local_path))
                        self._apply_device_to_model(m)
                        info["model"] = m
                        info["classes"] = list(m.names.values()) if hasattr(m, "names") else []
                    except Exception:
                        pass
                self.loaded_models[model_id] = info
                return

        # 2. Scan models_dir for any file whose stem matches model_id
        for f in sorted(self.models_dir.glob("*")):
            if f.suffix not in {".pt", ".pth", ".onnx", ".weights"}:
                continue
            if f.stem in self._SAM_LEGACY_STEMS:
                continue
            if f.stem != model_id and f.name != model_id:
                continue
            mtype = self._detect_model_type(f)
            info = {
                "id": model_id, "name": f.name,
                "type": mtype, "path": str(f),
                "model": None, "classes": [],
            }
            if mtype in ("sam", "sam2"):
                self._try_load_sam_ultralytics(str(f), info)
                if not info.get("model"):
                    self._load_sam_model(f, info)
            elif mtype == "rfdetr":
                self._load_rfdetr_from_checkpoint(f, info)
            elif mtype == "yoloworld":
                try:
                    from ultralytics import YOLOWorld
                    m = YOLOWorld(str(f))
                    self._apply_device_to_model(m)
                    info["model"] = m
                    info["classes"] = []
                except Exception:
                    self._load_yolo_model(f, info)  # fallback
            else:
                self._load_yolo_model(f, info)
            self.loaded_models[model_id] = info
            return

    def annotate_single_image(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        image_id: str,
        confidence_threshold: float = 0.5,
        prompt_point: Optional[tuple] = None,
        text_prompt: Optional[str] = None,
        image_path_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Auto-annotate a single image"""
        # Auto-load from disk if not already in memory
        if model_id not in self.loaded_models or not self.loaded_models[model_id].get("model"):
            self._try_autoload(model_id)

        if model_id not in self.loaded_models:
            raise RuntimeError(f"Model '{model_id}' is not loaded. Please load it first from the Models page.")

        model_info = self.loaded_models[model_id]
        model = model_info.get("model")
        model_type = model_info.get("type")

        if not model:
            load_error = model_info.get("error", "unknown error")
            raise RuntimeError(f"Model '{model_id}' failed to load: {load_error}")

        dataset_path = Path(dataset_path)

        # Find image file
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_file = None

        # Use path hint if provided (avoids expensive glob on large datasets)
        if image_path_hint:
            candidate = dataset_path / image_path_hint
            if candidate.exists():
                image_file = candidate

        if not image_file:
            for ext in IMAGE_EXTENSIONS:
                for found in dataset_path.glob(f"**/{image_id}{ext}"):
                    image_file = found
                    break
                if image_file:
                    break

        if not image_file:
            raise RuntimeError(f"Image file not found for id '{image_id}'")

        import traceback
        try:
            annotations = self._run_inference(
                model, model_type, image_file, confidence_threshold,
                prompt_point=prompt_point,
                text_prompt=text_prompt,
                model_classes=model_info.get("classes") or [],
            )
            return annotations
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Inference error ({model_type}): {exc}") from exc
    
    def _run_inference(
        self,
        model,
        model_type: str,
        image_path: Path,
        confidence_threshold: float,
        prompt_point: Optional[tuple] = None,
        text_prompt: Optional[str] = None,
        model_classes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on a single image"""
        annotations = []
        device = self._get_device()

        if model_type == "yolo":
            results = model(str(image_path), conf=confidence_threshold, device=device, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        # Get bounding box (xyxy format)
                        xyxy = box.xyxy[0].tolist()
                        
                        annotations.append({
                            "type": "bbox",
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": [
                                xyxy[0],
                                xyxy[1],
                                xyxy[2] - xyxy[0],
                                xyxy[3] - xyxy[1]
                            ]
                        })
                
                # Check for segmentation masks
                masks = result.masks
                if masks is not None:
                    for i, mask in enumerate(masks):
                        if i < len(boxes):
                            cls_id = int(boxes[i].cls[0])
                            cls_name = model.names[cls_id]
                            
                            # Get polygon from mask
                            xy = mask.xy[0] if mask.xy else []
                            if len(xy) > 0:
                                points = [coord for point in xy for coord in point]
                                annotations.append({
                                    "type": "polygon",
                                    "class_id": cls_id,
                                    "class_name": cls_name,
                                    "segmentation": [points]
                                })
        
        elif model_type in ["sam", "sam2", "sam3"]:
            try:
                import numpy as np
                import cv2

                image = Image.open(image_path).convert("RGB")
                image_array = np.array(image)
                h, w = image_array.shape[:2]

                def _mask_xy_to_annotation(mask_xy, cls_name="object") -> Optional[Dict]:
                    pts = np.array(mask_xy, dtype=np.float32)
                    if len(pts) < 3:
                        return None
                    return {
                        "type": "polygon",
                        "class_id": 0,
                        "class_name": cls_name,
                        "points": pts.flatten().tolist(),
                    }

                # ── ultralytics SAM path (preferred) ────────────────────────
                if hasattr(model, 'predict') and hasattr(model, 'info'):

                    if text_prompt is not None:
                        texts = [t.strip() for t in text_prompt.split(",") if t.strip()] or [text_prompt]

                        # ── Stage 1: try SAM native text grounding (low conf) ──
                        # SAM2.1 + GroundingDINO works here; SAM3 might not have
                        # a text encoder — we fall through to Stage 2 if empty.
                        try:
                            native_results = model(
                                str(image_path), texts=texts,
                                conf=0.1, verbose=False, device=device
                            )
                            for result in native_results:
                                if result.masks is None:
                                    continue
                                result_boxes = result.boxes
                                for i, mask_xy in enumerate(result.masks.xy):
                                    cls_name = texts[0] if len(texts) == 1 else "object"
                                    if result_boxes is not None and i < len(result_boxes):
                                        try:
                                            cls_id = int(result_boxes[i].cls[0])
                                            cls_name = texts[cls_id] if cls_id < len(texts) else texts[0]
                                        except Exception:
                                            pass
                                    ann = _mask_xy_to_annotation(mask_xy, cls_name)
                                    if ann:
                                        annotations.append(ann)
                        except Exception:
                            pass  # Native text not supported by this model

                        # ── Stage 2: YOLO-World → boxes → SAM masks ────────────
                        # Reliable two-stage pipeline: YOLO-World detects regions
                        # matching the text labels; SAM turns those boxes into masks.
                        if not annotations:
                            try:
                                from ultralytics import YOLOWorld
                                # Use the small variant (~100 MB); ultralytics caches it
                                world = YOLOWorld("yolov8s-worldv2.pt")
                                self._apply_device_to_model(world)
                                world.set_classes(texts)
                                det = world(
                                    str(image_path), conf=0.1,
                                    verbose=False, device=device
                                )
                                if det and det[0].boxes is not None and len(det[0].boxes):
                                    boxes_xyxy = det[0].boxes.xyxy.tolist()
                                    cls_ids    = [int(c) for c in det[0].boxes.cls.tolist()]
                                    box_names  = [
                                        texts[cid] if cid < len(texts) else texts[0]
                                        for cid in cls_ids
                                    ]
                                    # Use SAM to generate masks from detected boxes
                                    sam_results = model(
                                        str(image_path), bboxes=boxes_xyxy,
                                        verbose=False, device=device
                                    )
                                    for result in sam_results:
                                        if result.masks is None:
                                            continue
                                        for i, mask_xy in enumerate(result.masks.xy):
                                            cls_name = box_names[i] if i < len(box_names) else texts[0]
                                            ann = _mask_xy_to_annotation(mask_xy, cls_name)
                                            if ann:
                                                annotations.append(ann)
                            except Exception as yolo_world_err:
                                print(f"[text_seg] YOLO-World stage failed: {yolo_world_err}")

                    elif prompt_point is not None:
                        # Point-prompted: clicked pixel coordinates (normalized → pixel)
                        px = int(prompt_point[0] * w)
                        py = int(prompt_point[1] * h)
                        results = model(str(image_path), points=[[px, py]], labels=[1], verbose=False, device=device)
                        for result in results:
                            if result.masks is None:
                                continue
                            for mask_xy in result.masks.xy:
                                ann = _mask_xy_to_annotation(mask_xy)
                                if ann:
                                    annotations.append(ann)

                    else:
                        # No prompt — segment everything (grid/automatic mode)
                        results = model(str(image_path), verbose=False, device=device)
                        for result in results:
                            if result.masks is None:
                                continue
                            for mask_xy in result.masks.xy:
                                pts = np.array(mask_xy, dtype=np.float32)
                                if len(pts) < 3:
                                    continue
                                # Filter out tiny noise and full-image background masks
                                x1, y1 = pts.min(axis=0)
                                x2, y2 = pts.max(axis=0)
                                area_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
                                if area_ratio < 0.005 or area_ratio > 0.95:
                                    continue
                                annotations.append({
                                    "type": "polygon",
                                    "class_id": 0,
                                    "class_name": "object",
                                    "points": pts.flatten().tolist(),
                                })

                # ── SamAutomaticMaskGenerator path ───────────────────────────
                elif hasattr(model, 'generate'):
                    masks = model.generate(image_array)
                    for mask_data in masks:
                        binary = mask_data.get('segmentation')
                        if binary is None:
                            continue
                        score = mask_data.get('predicted_iou', 1.0)
                        if score < confidence_threshold:
                            continue
                        contours, _ = cv2.findContours(
                            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for contour in contours:
                            if contour.shape[0] < 3:
                                continue
                            pts = contour.squeeze().astype(float)
                            if pts.ndim != 2:
                                continue
                            annotations.append({
                                "type": "polygon",
                                "class_id": 0,
                                "class_name": "object",
                                "confidence": float(score),
                                "points": pts.flatten().tolist(),
                            })

                # ── SamPredictor / SAM2ImagePredictor — grid prompts ─────────
                elif hasattr(model, 'set_image'):
                    model.set_image(image_array)
                    # Sample a sparse 4×4 grid of foreground prompts
                    grid_pts = np.array([
                        [w * (c + 0.5) / 4, h * (r + 0.5) / 4]
                        for r in range(4) for c in range(4)
                    ], dtype=np.float32)
                    seen_masks: list = []
                    for pt in grid_pts:
                        masks_out, scores_out, _ = model.predict(
                            point_coords=pt[None],
                            point_labels=np.array([1]),
                            multimask_output=False,
                        )
                        if scores_out[0] < confidence_threshold:
                            continue
                        binary = masks_out[0].astype(np.uint8)
                        # Deduplicate by IoU with already-seen masks
                        duplicate = False
                        for prev in seen_masks:
                            inter = np.logical_and(binary, prev).sum()
                            union = np.logical_or(binary, prev).sum()
                            if union > 0 and inter / union > 0.5:
                                duplicate = True
                                break
                        if duplicate:
                            continue
                        seen_masks.append(binary)
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for contour in contours:
                            if contour.shape[0] < 3:
                                continue
                            pts = contour.squeeze().astype(float)
                            if pts.ndim != 2:
                                continue
                            annotations.append({
                                "type": "polygon",
                                "class_id": 0,
                                "class_name": "object",
                                "confidence": float(scores_out[0]),
                                "points": pts.flatten().tolist(),
                            })

            except Exception:
                pass

        elif model_type == "rfdetr":
            try:
                detections = model.predict(str(image_path), threshold=confidence_threshold)
                # detections is a supervision.Detections object
                if detections is not None and len(detections) > 0:
                    xyxy_arr   = detections.xyxy          if detections.xyxy is not None else []
                    conf_arr   = detections.confidence    if detections.confidence is not None else []
                    cls_arr    = detections.class_id      if detections.class_id is not None else []
                    # class names may live in detections.data["class_name"]
                    data       = detections.data or {}
                    names_arr  = data.get("class_name", [])

                    for i in range(len(xyxy_arr)):
                        x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]
                        conf = float(conf_arr[i]) if i < len(conf_arr) else 1.0
                        cls_id = int(cls_arr[i]) if i < len(cls_arr) else 0
                        if i < len(names_arr):
                            raw = names_arr[i]
                            cls_name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        elif model_classes and cls_id < len(model_classes):
                            cls_name = model_classes[cls_id]
                        else:
                            cls_name = str(cls_id)
                        annotations.append({
                            "type": "bbox",
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })
            except Exception as e:
                print(f"[rfdetr] inference error: {e}")

        elif model_type == "yoloworld":
            try:
                import torch
                texts = [t.strip() for t in text_prompt.split(",") if t.strip()] if text_prompt else ["object"]
                model.set_classes(texts)
                # set_classes computes text embeddings via CLIP on CPU; move the entire
                # model (including all text tensors) to the inference device.
                if device != "cpu":
                    target_device = f"cuda:{device}" if isinstance(device, int) else str(device)
                    try:
                        # Move the full model graph — this covers txt_feats, txt_pe and
                        # any other tensors that set_classes() computed on CPU.
                        for container in [
                            getattr(model, "model", None),
                            getattr(getattr(model, "predictor", None), "model", None),
                        ]:
                            if container is not None:
                                try:
                                    container.to(target_device)
                                except Exception:
                                    pass
                                # Belt-and-suspenders: also move individual text attrs
                                for attr in ("txt_feats", "txt_pe"):
                                    val = getattr(container, attr, None)
                                    if val is not None:
                                        try:
                                            setattr(container, attr, val.to(target_device))
                                        except Exception:
                                            pass
                    except Exception:
                        pass
                results = model(str(image_path), conf=confidence_threshold, verbose=False, device=device)
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = texts[cls_id] if cls_id < len(texts) else texts[0]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        annotations.append({
                            "type": "bbox",
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        })
            except Exception as e:
                import traceback; traceback.print_exc()
                raise RuntimeError(f"YOLO World inference failed: {e}") from e

        elif model_type == "groundingdino":
            try:
                import torch
                processor = getattr(model, "_grounding_processor", None)
                if processor is None:
                    print("[groundingdino] processor not found on model object")
                else:
                    texts = [t.strip() for t in text_prompt.split(",") if t.strip()] if text_prompt else ["object"]
                    # GroundingDINO expects ". "-separated labels ending with "."
                    text_input = ". ".join(texts) + "."
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, text=text_input, return_tensors="pt")
                    # Determine inference device and move ALL model tensors there
                    # (lazy-initialized buffers may still be on CPU after the initial .to())
                    model_device = next(model.parameters()).device
                    model.to(model_device)
                    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Post-processing mixes logit-derived CUDA indices with CPU tokenizer
                    # lookups — do it entirely on CPU to avoid device mismatches.
                    cpu_outputs = outputs.__class__(
                        **{k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                           for k, v in outputs.items()}
                    )
                    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
                    results = processor.post_process_grounded_object_detection(
                        cpu_outputs,
                        inputs["input_ids"].cpu(),
                        threshold=confidence_threshold,
                        text_threshold=max(0.1, confidence_threshold * 0.5),
                        target_sizes=target_sizes,
                    )
                    result = results[0]
                    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                        score_val = float(score)
                        if score_val < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in box]
                        # label may be a string phrase from the text prompt
                        cls_name = str(label).strip() if label else "object"
                        # Map to closest text prompt label
                        cls_id = 0
                        for idx, t in enumerate(texts):
                            if t.lower() in cls_name.lower() or cls_name.lower() in t.lower():
                                cls_id = idx
                                cls_name = t
                                break
                        annotations.append({
                            "type": "bbox",
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": score_val,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })
            except Exception as e:
                import traceback; traceback.print_exc()
                raise RuntimeError(f"GroundingDINO inference failed: {e}") from e

        elif model_type == "owlvit":
            try:
                import torch
                processor = getattr(model, "_owlvit_processor", None)
                if processor is None:
                    print("[owlvit] processor not found on model object")
                else:
                    texts = [t.strip() for t in text_prompt.split(",") if t.strip()] if text_prompt else ["object"]
                    # OWL-ViT expects a list of query strings per image
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(text=[texts], images=image, return_tensors="pt")
                    model_device = next(model.parameters()).device
                    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
                    # post_process_object_detection lives on the image_processor in
                    # most transformers versions; fall back to processor itself if missing.
                    _post_fn = getattr(processor, "post_process_object_detection", None) \
                        or getattr(processor.image_processor, "post_process_object_detection", None)
                    results = _post_fn(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=confidence_threshold,
                    )
                    result = results[0]
                    for score, label_idx, box in zip(result["scores"], result["labels"], result["boxes"]):
                        score_val = float(score)
                        if score_val < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                        cls_id = int(label_idx)
                        cls_name = texts[cls_id] if cls_id < len(texts) else "object"
                        annotations.append({
                            "type": "bbox",
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": score_val,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })
            except Exception as e:
                import traceback; traceback.print_exc()
                raise RuntimeError(f"OWL-ViT inference failed: {e}") from e

        return annotations

    def _get_dataset_classes(self, dataset_path: Path, format_name: str) -> List[str]:
        """Get classes from dataset"""
        classes = []
        
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            for yaml_file in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
                try:
                    with open(yaml_file) as f:
                        config = yaml.safe_load(f)
                        if "names" in config:
                            if isinstance(config["names"], dict):
                                classes = list(config["names"].values())
                            else:
                                classes = config["names"]
                        break
                except Exception:
                    pass
        
        elif format_name == "coco":
            for json_file in list(dataset_path.glob("*.json")) + list(dataset_path.glob("annotations/*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "categories" in data:
                            classes = [cat["name"] for cat in data["categories"]]
                        break
                except Exception:
                    pass
        
        return classes
    
    def _save_annotations(
        self,
        dataset_path: Path,
        format_name: str,
        image_path: Path,
        annotations: List[Dict],
        classes: List[str]
    ):
        """Save annotations to dataset format"""
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        image_id = image_path.stem

        # generic-images datasets (raw upload, no format yet) are promoted to YOLO on
        # first annotation save — labels go next to the images in a sibling labels/ dir
        if format_name == "generic-images":
            format_name = "yolo"

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            # Determine labels directory — mirror the image's split/images folder.
            # Works for flat, nested, and multi-depth datasets:
            #   images/x.jpg              → labels/x.txt
            #   train/images/x.jpg        → train/labels/x.txt
            #   myds/train/images/x.jpg   → myds/train/labels/x.txt
            try:
                rel_parts = image_path.relative_to(dataset_path).parts
                # Find the 'images' or 'imgs' directory anywhere in the path
                img_idx = next(
                    (i for i, p in enumerate(rel_parts) if p.lower() in ("images", "imgs")),
                    None
                )
                if img_idx is not None:
                    labels_dir = dataset_path.joinpath(*rel_parts[:img_idx]) / "labels"
                else:
                    raise ValueError("no 'images' segment in path")
            except Exception:
                # Fallback: use train/labels if it exists, otherwise root labels
                labels_dir = dataset_path / "labels"
                if (dataset_path / "train" / "labels").exists():
                    labels_dir = dataset_path / "train" / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Re-read the yaml every call so classes accumulate correctly across
            # multiple images saved in the same batch (each call sees what prior
            # calls already wrote rather than using a stale in-memory list).
            live_classes = list(classes)
            for _yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
                try:
                    with open(_yf) as _f:
                        _cfg = yaml.safe_load(_f) or {}
                        if "names" in _cfg:
                            _n = _cfg["names"]
                            live_classes = list(_n.values()) if isinstance(_n, dict) else list(_n)
                except Exception:
                    pass
                break

            class_to_id = {name: idx for idx, name in enumerate(live_classes)}
            new_classes = []
            for ann in annotations:
                if ann["class_name"] not in class_to_id:
                    class_to_id[ann["class_name"]] = len(live_classes) + len(new_classes)
                    new_classes.append(ann["class_name"])

            if new_classes:
                self._update_yolo_classes(dataset_path, live_classes + new_classes)
            
            # Write label file
            label_file = labels_dir / f"{image_id}.txt"
            with open(label_file, "w") as f:
                for ann in annotations:
                    class_id = class_to_id.get(ann["class_name"], 0)
                    
                    if ann.get("type") == "polygon":
                        # Accept both "points" (flat list) and legacy "segmentation" [[...]]
                        raw_pts = ann.get("points") or (ann.get("segmentation", [[]])[0] if ann.get("segmentation") else [])
                        normalized = []
                        for i in range(0, len(raw_pts) - 1, 2):
                            normalized.append(raw_pts[i] / width)
                            normalized.append(raw_pts[i + 1] / height)
                        if normalized:
                            f.write(f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized) + "\n")
                    elif ann.get("bbox"):
                        bbox = ann["bbox"]
                        x_center = (bbox[0] + bbox[2]/2) / width
                        y_center = (bbox[1] + bbox[3]/2) / height
                        w = bbox[2] / width
                        h = bbox[3] / height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    def _update_yolo_classes(self, dataset_path: Path, classes: List[str]):
        """Update YOLO classes in yaml file"""
        yaml_file = None
        for yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
            yaml_file = yf
            break
        
        if not yaml_file:
            yaml_file = dataset_path / "data.yaml"
        
        if yaml_file.exists():
            with open(yaml_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {
                "path": str(dataset_path.absolute()),
                "train": "images",
                "val": "images"
            }
        
        config["names"] = {i: name for i, name in enumerate(classes)}
        config["nc"] = len(classes)
        
        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def annotate_with_new_classes(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        new_classes: List[str]
    ) -> Dict[str, Any]:
        """Annotate dataset with new classes while keeping existing annotations"""
        if model_id not in self.loaded_models:
            return {"error": "Model not loaded"}
        
        model_info = self.loaded_models[model_id]
        model_classes = model_info.get("classes", [])
        
        # Filter to only detect new classes
        target_class_ids = []
        for i, cls in enumerate(model_classes):
            if cls in new_classes:
                target_class_ids.append(i)
        
        if not target_class_ids:
            return {"error": "None of the new classes found in model"}
        
        # Run auto-annotation for new classes only
        results = self.auto_annotate(
            model_id, dataset_path, format_name, confidence_threshold=0.5
        )
        
        results["added_classes"] = new_classes
        return results
