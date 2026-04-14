![Hero Screenshot](assets/images/hero.png)

# VisOS

The all-in-one local workbench for computer vision datasets. Annotate, convert, augment, merge, and train — without touching a cloud service or writing a line of code.

---

![Annotation canvas with bounding boxes, polygon tool active, and class panel on the right](assets/images/annotate.png)

---

## Why

Managing CV datasets is a grind. You download a COCO dataset, realise your framework expects YOLO, spend an hour writing a conversion script, discover half your classes are duplicates with different names, and write another script to fix it. Repeat forever.

VisOS wraps all of that in a local UI. No accounts, no uploads, no bill.

---

## Getting Started

**Prerequisites:** Python 3.10+, Node.js 18+, npm or pnpm

```bash
git clone https://github.com/your-org/cv-dataset-manager.git
cd cv-dataset-manager
python3 run.py start
```

`run.py` creates a virtualenv, installs all dependencies, starts the FastAPI backend on `:8000` and the Next.js frontend on `:3000`, health-checks both, and opens your browser.

First run takes 2–5 minutes while PyTorch and Ultralytics download (~1.5 GB).

---

## Features

### Datasets

Load from a local folder or ZIP. Format is auto-detected on load. Datasets persist across restarts via metadata sidecar files.

**Supported formats (load & export):**  
YOLO · COCO · Pascal VOC · LabelMe · CreateML · TensorFlow CSV · ImageNet classification · YOLO OBB · COCO Panoptic · Cityscapes · ADE20K · DOTA · TFRecord

![Loaded datasets list with format badges, image counts, and action buttons](assets/images/datasets_view.png)

---

### Dashboard

Per-dataset overview: image count, annotation coverage, class distribution chart, train/val/test split breakdown.

![Dataset dashboard showing class distribution chart and split breakdown](assets/images/dataset_dashboard.png)

---

### Sort & Filter

Review images one at a time with annotations overlaid. Keyboard-driven — right arrow to keep, left to mark for deletion. Apply bulk changes when done. Filter by annotation status or class. Shift-click for range selection.

![Sort and filter view with image displayed, bounding box overlays, and keep/delete hint](assets/images/sort&filter.png)

---

### Annotation

Canvas-based editor with six tools:

| Tool | Shortcut |
|---|---|
| Select / Edit | V |
| Bounding Box | B |
| Polygon | P |
| Keypoint | L |
| Brush | R |
| SAM Wand | auto-activates when a SAM model is loaded |

Full undo/redo. Annotations save automatically.

**Auto-annotation:** load any YOLO, RT-DETR, RF-DETR, SAM, SAM 2/2.1/3, or GroundingDINO model and run inference directly on your dataset with a configurable confidence threshold. GroundingDINO supports zero-shot annotation via text prompt.

---

### Class Management

Extract, delete, merge, or rename classes without touching JSON. Shows per-class annotation counts.

![Class management table with counts, colour swatches, and bulk operation checkboxes](assets/images/class_management.png)

---

### Format Conversion

Convert any supported format to any other. Optionally copy images alongside annotations or annotations only.

![Convert view with source format auto-detected as COCO and target set to YOLO](assets/images/convert_format.png)

---

### Train / Val / Test Split

Dedicated split view with configurable ratios, optional stratification by class, and a fixed random seed for reproducibility.

---

### Augmentation

Toggle-based pipeline builder. Preview sample outputs before applying. Output to a target image count or a multiplier.

**Transforms:** horizontal/vertical flip · rotation · scale · translation · shear · perspective · random crop · brightness · contrast · saturation · hue shift · grayscale · Gaussian blur · Gaussian noise · sharpen · JPEG compression · cutout · mosaic · MixUp · elastic deformation · grid distortion · histogram equalisation · channel shuffle · invert · posterize · solarize

![Augmentation view with pipeline configured and four preview samples shown](assets/images/augmentation.png)

---

### Video Frame Extraction

Turn video files into annotatable image datasets.

- Every Nth frame
- N frames uniformly distributed across the video
- Keyframes on scene change
- Manual frame selection with scrubber

Supports MP4, AVI, MOV, MKV, WebM.

![Video extraction view with frame interval set and estimated frame count shown](assets/images/frame_extraction.png)

---

### Duplicate Detection

| Method | What it finds |
|---|---|
| MD5 Hash | Exact byte-for-byte duplicates |
| Perceptual Hash (pHash) | Visually similar images |
| Average Hash (aHash) | Fast approximate similarity |
| CLIP Embeddings | Semantically similar content |

Configurable similarity threshold. Keep strategy: first, largest resolution, or smallest file.

---

### Dataset Merging

Combine multiple datasets with a class-mapping UI to resolve naming conflicts before merging.

![Merge view with two datasets loaded and a class conflict resolved via mapping table](assets/images/merge_datasets.png)

---

### Model Management

Download pretrained weights from inside the app or import your own `.pt`, `.pth`, or `.onnx` file. Load and unload models to manage GPU memory.

**Available pretrained models:**  
YOLOv5 (n/s) · YOLOv8 (n/s/m/l/x, seg, cls variants) · YOLOv9 (n/s/m/c/e) · YOLOv10 (n/s/m/b/l/x) · RT-DETR (L/X) · RF-DETR (Base/Large) · SAM ViT-B/L · SAM 2 (Tiny/Small/Base+/Large) · SAM 2.1 (Tiny/Small/Base+/Large) · SAM 3 · GroundingDINO (Tiny/Base, zero-shot)

![Model management view with download and import options, loaded model shown with GPU memory usage](assets/images/model_management.png)

---

### Training

Train locally with live metric monitoring. Supports detection, instance segmentation, and classification tasks.

**Architectures:** YOLOv8 · YOLOv9 · YOLOv10 · RF-DETR  
**Configurable:** epochs, batch size, image size, learning rate, early-stopping patience  
**Live:** loss, accuracy, validation loss, GPU usage, ETA  
**Controls:** pause, resume, stop with checkpoint saving  
**Export:** PyTorch, ONNX, TensorRT

![Training view mid-run with loss and accuracy charts updating live and GPU usage visible](assets/images/training_view.png)

---

### Batch Jobs

Track and manage background auto-annotation jobs. Resume interrupted jobs, preview annotated images inline, and monitor per-image progress.

---

### Additional Views

**Gallery** — grid browser with annotation overlays and full-size click-through  
**Compare** — side-by-side stats between two datasets  
**Snapshots** — save and restore named dataset states before destructive operations  
**YAML Config** — GUI editor for `data.yaml` files  
**Health Check** — backend API status, Python dependencies, GPU availability, workspace disk usage

---

## Usage

```bash
python3 run.py start           # Start both servers
python3 run.py stop            # Stop cleanly
python3 run.py restart         # Full restart
python3 run.py restart-back    # Backend only
python3 run.py restart-front   # Frontend only
python3 run.py status          # Show PIDs and ports
python3 run.py logs            # Tail live output
```

No required environment variables for local use. Backend URL defaults to `http://localhost:8000` and is configurable in Settings.

---

## Architecture

```
cv-dataset-manager/
├── run.py                    # Cross-platform process manager
├── backend/
│   ├── main.py               # FastAPI entrypoint and all routes
│   ├── dataset_parsers.py    # Format auto-detection and parsing
│   ├── format_converter.py   # Cross-format conversion
│   ├── annotation_tools.py   # Annotation read/write logic
│   ├── augmentation.py       # Augmentation pipeline engine
│   ├── dataset_merger.py     # Merge with class mapping
│   ├── model_integration.py  # Model download, load/unload, inference
│   ├── training.py           # Training job management and metric streaming
│   ├── video_utils.py        # Frame extraction, duplicate detection, CLIP
│   └── requirements.txt
└── components/               # React views (one per sidebar section)
```

**Proxy pattern:** Next.js API routes in `app/api/backend/` forward all requests to FastAPI, eliminating CORS issues. The frontend only ever talks to `localhost:3000`.

**Persistence:** Datasets survive restarts via `dataset_metadata.json` sidecars. On startup the backend scans `workspace/datasets/` and re-registers everything it finds.

**Process manager:** `run.py` handles cross-platform PID tracking, port cleanup, and log streaming — no Docker required.

---

## API

Base URL: `http://localhost:8000/api`  
Interactive docs: `http://localhost:8000/docs`

| Resource | Endpoints |
|---|---|
| Datasets | `GET /datasets` · `POST /datasets/load-local` · `POST /datasets/upload` · `GET/DELETE /datasets/{id}` |
| Images | `GET /datasets/{id}/images` · `GET /datasets/{id}/images/{image_id}` · `PUT .../annotations` |
| Classes | `POST /datasets/{id}/extract-classes` · `/delete-classes` · `/merge-classes` |
| Conversion | `POST /datasets/{id}/convert` · `POST /datasets/merge` · `GET /formats` |
| Augmentation | `POST /datasets/{id}/augment-enhanced` |
| Video | `POST /video/extract` |
| Duplicates | `POST /datasets/{id}/find-duplicates` · `/remove-duplicates` |
| Models | `GET /models` · `POST /models/download` · `POST /models/import` · `POST /models/{id}/load` · `POST /models/{id}/unload` |
| Auto-annotation | `POST /datasets/{id}/auto-annotate` · `GET /api/auto-annotate/jobs` |
| Training | `POST /training/start` · `GET /training/{job_id}/status` · `/pause` · `/resume` · `/stop` |
| System | `GET /api/health` |

---

## Troubleshooting

**"Backend not connected"** — Python failed to start. Check `.logs/backend.log`. Common causes: Python < 3.10, port 8000 in use, missing OpenCV system dependency.

**First startup hangs** — Normal. PyTorch is large. Check `.logs/backend.log` to watch pip progress.

**"Dataset format not recognized"** — Auto-detection looks for specific files (`data.yaml`, `instances_train.json`, `Annotations/*.xml`, etc). Match the folder structure exactly. Nested ZIPs aren't supported — extract first.

**OOM during training** — Lower batch size (try 4–8), lower image size (try 320), or use a smaller arch (`yolov8n`). Check VRAM with `nvidia-smi`.

**Port still in use after crash** — `python3 run.py stop`. If that fails: `lsof -ti:3000 | xargs kill -9` (macOS/Linux) or `netstat -ano | findstr :3000` → `taskkill /F /PID <pid>` (Windows).

**Blank frontend / 500** — Run `npm install` manually in the project root, then `python3 run.py restart-front`.

---

> ⚠️ The FastAPI backend serves files directly from your local filesystem. Don't expose port 8000 to the public internet without authentication. For remote GPU servers, use SSH port forwarding: `ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@server`