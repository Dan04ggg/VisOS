"""
Training Module - Local model training with comprehensive logging
"""

import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Capture sys.path at import time so daemon threads can find venv site-packages
_VENV_SYSPATH = list(sys.path)


def _restore_syspath():
    for p in _VENV_SYSPATH:
        if p not in sys.path:
            sys.path.insert(0, p)


# ── Device resolution ────────────────────────────────────────────────────────

def _resolve_device(config: Dict, job: Dict) -> Any:
    """Return the best available device (int, 'cpu', or 'mps')."""
    pref = config.get("device", "auto")
    if pref not in ("auto", ""):
        job["logs"].append(f"Using user-specified device: {pref}")
        return pref

    # 1. torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            job["logs"].append(f"CUDA available — {n} GPU(s): {names}")
            job["device_info"] = names
            return 0
        else:
            job["logs"].append("torch.cuda.is_available() = False (PyTorch may be CPU-only)")
    except ImportError:
        job["logs"].append("PyTorch not found")
    except Exception as e:
        job["logs"].append(f"CUDA check error: {e}")

    # 2. nvidia-smi confirms GPU hardware present
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and "GPU" in r.stdout:
            lines = r.stdout.strip().splitlines()
            job["logs"].append(f"GPU hardware detected by nvidia-smi: {lines[0]}")
            job["logs"].append(
                "⚠ Install CUDA-enabled PyTorch to use your GPU:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                "Falling back to CPU for this run."
            )
            job["device_info"] = lines[0]
    except Exception:
        pass

    # 3. Apple MPS
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            job["logs"].append("Using Apple MPS (Metal Performance Shaders)")
            job["device_info"] = "Apple MPS"
            return "mps"
    except Exception:
        pass

    job["logs"].append("Using CPU (no GPU acceleration available)")
    return "cpu"


# ── Comprehensive callbacks ──────────────────────────────────────────────────

def _make_callbacks(job: Dict) -> Dict:
    """Create ultralytics callbacks that write structured metrics + human logs."""

    def on_train_start(trainer):
        device = str(getattr(trainer, "device", "unknown"))
        job["logs"].append(f"Training started on device: {device}")
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    job["logs"].append(f"  GPU {i}: {name}  ({total:.1f} GB total VRAM)")
        except Exception:
            pass
        job["logs"].append(
            f"{'Epoch':>6}  {'GPU-Mem':>8}  {'box_loss':>10}  {'cls_loss':>10}  "
            f"{'dfl_loss':>10}  {'mAP50':>8}  {'mAP50-95':>10}  {'Speed':>10}"
        )
        job["logs"].append("─" * 90)

    def on_fit_epoch_end(trainer):
        # Store trainer reference so stop/pause can signal it
        job["_trainer"] = trainer

        # Check stop/pause flag — stops after current epoch completes
        if job.get("_stop_requested"):
            # Save last checkpoint path before stopping
            project = Path(getattr(trainer, "save_dir", "") or "")
            last_pt = project / "weights" / "last.pt"
            if last_pt.exists():
                job["last_checkpoint"] = str(last_pt)
                job["model_path"] = str(last_pt)
            # Transition "pausing" → "paused" now that the epoch is fully done
            if job.get("status") == "pausing":
                job["status"] = "paused"
                job["logs"].append("Epoch complete — training paused. Checkpoint saved.")
            trainer.stop = True  # ultralytics flag to stop after epoch

        epoch = trainer.epoch + 1
        total = job["total_epochs"]
        raw_metrics = dict(trainer.metrics) if trainer.metrics else {}

        # ── Training losses ──────────────────────────────────────────────────
        tloss = getattr(trainer, "tloss", None)
        loss_names = list(getattr(trainer, "loss_names", []) or [])
        losses: Dict[str, float] = {}

        if tloss is not None:
            try:
                # tloss may be a tensor, list, or scalar
                if hasattr(tloss, "__len__"):
                    for i, name in enumerate(loss_names):
                        try:
                            losses[name] = float(tloss[i])
                        except Exception:
                            pass
                else:
                    losses["total"] = float(tloss)
            except Exception:
                pass

        box_loss = losses.get("box_loss", losses.get("box", 0.0)) or 0.0
        cls_loss = losses.get("cls_loss", losses.get("cls", 0.0)) or 0.0
        dfl_loss = losses.get("dfl_loss", losses.get("dfl", 0.0)) or 0.0
        seg_loss = losses.get("seg_loss", losses.get("seg", 0.0)) or 0.0

        # ── Val losses ───────────────────────────────────────────────────────
        val_box = float(raw_metrics.get("val/box_loss", 0) or 0)
        val_cls = float(raw_metrics.get("val/cls_loss", 0) or 0)
        val_dfl = float(raw_metrics.get("val/dfl_loss", 0) or 0)
        val_seg = float(raw_metrics.get("val/seg_loss", 0) or 0)

        # ── mAP / P / R ──────────────────────────────────────────────────────
        mAP50     = float(raw_metrics.get("metrics/mAP50(B)",    0) or 0)
        mAP50_95  = float(raw_metrics.get("metrics/mAP50-95(B)", 0) or 0)
        mAP50_seg = float(raw_metrics.get("metrics/mAP50(M)",    0) or 0)
        precision = float(raw_metrics.get("metrics/precision(B)", 0) or 0)
        recall    = float(raw_metrics.get("metrics/recall(B)",    0) or 0)

        # ── Speed ─────────────────────────────────────────────────────────────
        speed = dict(getattr(trainer, "speed", {}) or {})
        pre_ms  = float(speed.get("preprocess",  0) or 0)
        inf_ms  = float(speed.get("inference",   0) or 0)
        loss_ms = float(speed.get("loss",        0) or 0)
        post_ms = float(speed.get("postprocess", 0) or 0)
        total_ms = pre_ms + inf_ms + loss_ms + post_ms

        # ── GPU memory ────────────────────────────────────────────────────────
        gpu_mem_gb: Optional[float] = None
        gpu_mem_str = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.memory_reserved(0) / 1e9
                gpu_mem_str = f"{gpu_mem_gb:.2f}G"
                job["gpu_mem_gb"] = round(gpu_mem_gb, 2)
        except Exception:
            pass

        # ── Instance count ────────────────────────────────────────────────────
        batch = getattr(trainer, "batch", None)
        inst_count = 0
        if isinstance(batch, dict):
            cls_tensor = batch.get("cls")
            if cls_tensor is not None:
                try:
                    inst_count = int(len(cls_tensor))
                except Exception:
                    pass

        # ── Human-readable log line ───────────────────────────────────────────
        epoch_line = (
            f"{epoch:>4}/{total:<4}  "
            f"{gpu_mem_str:>8}  "
            f"{box_loss:>10.4f}  {cls_loss:>10.4f}  {dfl_loss:>10.4f}  "
            f"{mAP50:>8.4f}  {mAP50_95:>10.4f}  "
            f"{total_ms:>8.1f}ms"
        )
        job["logs"].append(epoch_line)

        if val_box > 0 or mAP50 > 0:
            val_line = (
                f"  {'Val':>8}  {' ':>8}  "
                f"{val_box:>10.4f}  {val_cls:>10.4f}  {val_dfl:>10.4f}  "
                f"{mAP50:>8.4f}  {mAP50_95:>10.4f}"
            )
            if precision > 0:
                val_line += f"  P:{precision:.3f}  R:{recall:.3f}"
            job["logs"].append(val_line)

        if total_ms > 0:
            job["logs"].append(
                f"  Speed — pre:{pre_ms:.1f}ms  inf:{inf_ms:.1f}ms  "
                f"loss:{loss_ms:.1f}ms  post:{post_ms:.1f}ms  "
                + (f"inst:{inst_count}" if inst_count else "")
            )

        # ── Structured epoch record for charting ──────────────────────────────
        record: Dict[str, Any] = {
            "epoch":           epoch,
            "train_box_loss":  round(box_loss,   5),
            "train_cls_loss":  round(cls_loss,   5),
            "train_dfl_loss":  round(dfl_loss,   5),
            "val_box_loss":    round(val_box,     5),
            "val_cls_loss":    round(val_cls,     5),
            "val_dfl_loss":    round(val_dfl,     5),
            "mAP50":           round(mAP50,       5),
            "mAP50_95":        round(mAP50_95,    5),
            "precision":       round(precision,   5),
            "recall":          round(recall,      5),
            "speed_ms":        round(total_ms,    2),
        }
        if seg_loss > 0:
            record["train_seg_loss"] = round(seg_loss, 5)
        if val_seg > 0:
            record["val_seg_loss"] = round(val_seg, 5)
        if mAP50_seg > 0:
            record["mAP50_seg"] = round(mAP50_seg, 5)
        if gpu_mem_gb is not None:
            record["gpu_mem_gb"] = round(gpu_mem_gb, 2)

        job["epoch_history"].append(record)
        job["metrics"]       = record
        job["progress"]      = epoch / total * 100
        job["current_epoch"] = epoch

    def on_train_end(trainer):
        job["logs"].append("─" * 90)
        job["logs"].append("Training finished.")
        try:
            if hasattr(trainer, "best") and trainer.best:
                job["logs"].append(f"Best weights: {trainer.best}")
        except Exception:
            pass

    return {
        "on_train_start":   on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end":     on_train_end,
    }


# ── Training manager ─────────────────────────────────────────────────────────

class TrainingManager:
    """Manage local model training"""

    def __init__(self, jobs_file: Optional[Path] = None):
        self.training_jobs:    Dict[str, Dict[str, Any]] = {}
        self.training_threads: Dict[str, threading.Thread] = {}
        self.jobs_file = jobs_file
        if self.jobs_file:
            self._restore_jobs()

    # ── Persistence ───────────────────────────────────────────────────────────

    # Fields that cannot be serialised to JSON (live objects / internal flags)
    _SKIP_KEYS = {"model", "_trainer", "_stop_requested"}

    def _persist_jobs(self) -> None:
        """Write serialisable job metadata to disk so the frontend can restore it."""
        if not self.jobs_file:
            return
        try:
            import json as _json
            snapshot = {}
            for job_id, job in self.training_jobs.items():
                snapshot[job_id] = {
                    k: v for k, v in job.items()
                    if k not in self._SKIP_KEYS and not callable(v)
                }
            self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jobs_file, "w") as f:
                _json.dump(snapshot, f, indent=2, default=str)
        except Exception as exc:
            print(f"[training] Could not persist jobs: {exc}")

    def _restore_jobs(self) -> None:
        """Restore job metadata from disk on startup.
        Jobs that were 'running' or 'starting' are marked as 'interrupted'
        because their training thread died with the previous process."""
        if not self.jobs_file or not self.jobs_file.exists():
            return
        try:
            import json as _json
            with open(self.jobs_file) as f:
                saved: Dict[str, Any] = _json.load(f)
            for job_id, job in saved.items():
                if job.get("status") in ("running", "starting"):
                    job["status"] = "interrupted"
                    logs = list(job.get("logs", []))
                    logs.append("⚠ Training was interrupted — the backend process was restarted.")
                    job["logs"] = logs
                self.training_jobs[job_id] = job
            print(f"[training] Restored {len(saved)} training job(s) from disk.")
        except Exception as exc:
            print(f"[training] Could not restore jobs: {exc}")

    def start_training(
        self,
        dataset_path: Path,
        format_name: str,
        model_type: str,
        config: Dict[str, Any],
        name: str = "",
    ) -> str:
        training_id = str(uuid.uuid4())[:8]

        job: Dict[str, Any] = {
            "id":            training_id,
            "name":          name.strip() or f"Run {training_id}",
            "dataset_path":  str(dataset_path),
            "format":        format_name,
            "model_type":    model_type,
            "config":        config,
            "status":        "starting",
            "progress":      0,
            "current_epoch": 0,
            "total_epochs":  config.get("epochs", 100),
            "metrics":       {},
            "epoch_history": [],
            "started_at":    datetime.now().isoformat(),
            "model_path":    None,
            "device_info":   None,
            "gpu_mem_gb":    None,
            "logs":          [],
        }

        self.training_jobs[training_id] = job
        self._persist_jobs()  # record job creation immediately

        thread = threading.Thread(target=self._run_training, args=(training_id,), daemon=True)
        thread.start()
        self.training_threads[training_id] = thread

        return training_id

    @staticmethod
    def _release_memory(job: Dict) -> None:
        """Release GPU VRAM and Python heap after training ends or is stopped."""
        import gc
        job.pop("_trainer", None)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        job["logs"].append("GPU VRAM and RAM released.")

    def _run_training(self, training_id: str):
        _restore_syspath()
        job = self.training_jobs[training_id]
        try:
            mt = job["model_type"]
            if mt == "rfdetr":
                self._train_rfdetr(job)
            elif mt == "segmentation":
                self._train_segmentation(job)
            elif mt == "classification":
                self._train_classification(job)
            elif mt == "rtdetr":
                # RT-DETR from ultralytics is trained via the YOLO class
                self._train_yolo(job)
            else:
                # "yolo" — may need seg→bbox conversion if dataset has polygon labels
                self._train_yolo(job)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Don't overwrite a deliberate pause/stop with "failed"
            if job.get("status") not in ("paused", "pausing", "stopped"):
                job["status"] = "failed"
                job["error"]  = str(e)
            job["logs"].append(f"Fatal error: {e}")
            short_tb = "\n".join(tb.strip().splitlines()[-8:])
            job["logs"].append(short_tb)
        finally:
            self._persist_jobs()  # always persist final status

    # ── RF-DETR ───────────────────────────────────────────────────────────────

    def _train_rfdetr(self, job: Dict):
        import subprocess
        try:
            import rfdetr as _rfdetr_pkg  # noqa: F401
            # Also ensure train/loggers extras are available
            from rfdetr.detr import RFDETRBase as _  # triggers extras check
        except (ImportError, Exception):
            job["logs"].append("Installing rfdetr[train,loggers]…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", 'rfdetr[train,loggers]', "-q"]
            )

        from rfdetr import RFDETRBase, RFDETRLarge

        job["status"] = "running"
        config       = job["config"]
        dataset_path = Path(job["dataset_path"])

        # RF-DETR requires a COCO-format dataset.
        # Detect the dataset root by looking for a COCO annotations JSON.
        coco_json = self._find_coco_json(dataset_path, job)
        if coco_json is None:
            return  # error already set inside _find_coco_json

        device = _resolve_device(config, job)

        arch = config.get("base_model", "rfdetr_base")
        job["logs"].append(f"RF-DETR architecture: {arch}")
        job["logs"].append(f"Dataset (COCO): {dataset_path}")

        output_dir = dataset_path / "runs" / f"rfdetr_{job['id']}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if "large" in arch:
            model = RFDETRLarge()
            job["logs"].append("Loaded RF-DETR Large backbone")
        else:
            model = RFDETRBase()
            job["logs"].append("Loaded RF-DETR Base backbone")

        epochs    = config.get("epochs", 50)
        batch     = config.get("batch_size", 4)
        lr        = config.get("lr0", 1e-4)
        grad_acc  = max(1, 16 // max(batch, 1))  # accumulate to effective batch 16
        job["total_epochs"] = epochs
        job["logs"].append(f"Epochs: {epochs}  Batch: {batch}  LR: {lr}  GradAcc: {grad_acc}")

        # Build RF-DETR callbacks that feed the standard job metrics dict
        rfdetr_job_ref = job  # capture for closure

        def on_fit_epoch_end(logs: dict):
            epoch      = int(logs.get("epoch", rfdetr_job_ref["current_epoch"] + 1))
            map50      = float(logs.get("mAP50",    logs.get("map50",    0)) or 0)
            map50_95   = float(logs.get("mAP50_95", logs.get("map",      0)) or 0)
            train_loss = float(logs.get("train_loss", logs.get("loss",   0)) or 0)
            val_loss   = float(logs.get("val_loss",   0) or 0)

            gpu_mem_gb: Optional[float] = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.memory_reserved(0) / 1e9
                    rfdetr_job_ref["gpu_mem_gb"] = round(gpu_mem_gb, 2)
            except Exception:
                pass

            log_line = (
                f"{epoch:>4}/{epochs:<4}  "
                f"{'%.2fG' % gpu_mem_gb if gpu_mem_gb else '':>8}  "
                f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                f"mAP50: {map50:.4f}  mAP50-95: {map50_95:.4f}"
            )
            rfdetr_job_ref["logs"].append(log_line)

            record: Dict[str, Any] = {
                "epoch":          epoch,
                "train_box_loss": round(train_loss, 5),
                "train_cls_loss": 0.0,
                "train_dfl_loss": 0.0,
                "val_box_loss":   round(val_loss,   5),
                "val_cls_loss":   0.0,
                "val_dfl_loss":   0.0,
                "mAP50":          round(map50,      5),
                "mAP50_95":       round(map50_95,   5),
                "precision":      float(logs.get("precision", 0) or 0),
                "recall":         float(logs.get("recall",    0) or 0),
                "speed_ms":       0.0,
            }
            if gpu_mem_gb is not None:
                record["gpu_mem_gb"] = round(gpu_mem_gb, 2)

            rfdetr_job_ref["epoch_history"].append(record)
            rfdetr_job_ref["metrics"]       = record
            rfdetr_job_ref["progress"]      = epoch / epochs * 100
            rfdetr_job_ref["current_epoch"] = epoch

        try:
            model.train(
                dataset_dir=str(dataset_path),
                epochs=epochs,
                batch_size=batch,
                grad_accum_steps=grad_acc,
                lr=lr,
                output_dir=str(output_dir),
                callbacks={"on_fit_epoch_end": on_fit_epoch_end},
            )
            # RF-DETR saves checkpoint as checkpoint.pth in output_dir
            best_ckpt = output_dir / "checkpoint_best_total.pth"
            if not best_ckpt.exists():
                # fallback to any .pth
                ptfiles = list(output_dir.glob("*.pth"))
                best_ckpt = ptfiles[0] if ptfiles else None

            job["status"]     = "completed"
            job["progress"]   = 100
            job["model_path"] = str(best_ckpt) if best_ckpt else None
            job["logs"].append("✓ RF-DETR training complete")
            if best_ckpt:
                job["logs"].append(f"Weights saved: {best_ckpt}")
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)
            job["logs"].append(f"RF-DETR training error: {e}")
        finally:
            del model
            self._release_memory(job)

    @staticmethod
    def _find_coco_json(dataset_path: Path, job: Dict) -> Optional[Path]:
        """Locate a COCO-format annotations JSON file. Returns None and sets job error if not found."""
        # Common COCO layouts: annotations/instances_train2017.json, _annotations.coco.json, etc.
        candidates: List[Path] = (
            list(dataset_path.rglob("*instances_train*.json")) +
            list(dataset_path.rglob("*_annotations.coco.json")) +
            list(dataset_path.rglob("annotations/*.json")) +
            list(dataset_path.rglob("*.json"))
        )
        for p in candidates:
            if p.stat().st_size < 100:
                continue
            try:
                import json as _json
                with open(p) as f:
                    data = _json.load(f)
                if "images" in data and "annotations" in data and "categories" in data:
                    job["logs"].append(f"COCO annotations: {p}")
                    return p
            except Exception:
                continue
        job["status"] = "failed"
        job["error"]  = (
            "No COCO-format annotations JSON found. "
            "RF-DETR training requires a COCO dataset. "
            "Use Convert to transform your dataset to COCO format first."
        )
        job["logs"].append(job["error"])
        return None

    # ── Segmentation label helpers ────────────────────────────────────────────

    @staticmethod
    def _get_label_dir(img_dir: Path) -> Optional[Path]:
        """Return the label directory that corresponds to an images directory."""
        candidates = [
            # Standard YOLO layout: .../images/train → .../labels/train
            img_dir.parent.parent / "labels" / img_dir.name,
            # Flat layout: .../train/images → .../train/labels  (path ends with 'images')
            img_dir.parent / "labels",
            # String-replace variants for mixed separator environments
            Path(str(img_dir).replace(f"{Path.sep}images{Path.sep}", f"{Path.sep}labels{Path.sep}")),
            Path(str(img_dir).replace("/images/", "/labels/")),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _validate_seg_labels(data_yaml: Path, job: Dict) -> None:
        """Scan YOLO segmentation label files and ensure every annotation line is a
        valid polygon (≥ 7 tokens: class + at least 3 x/y pairs).

        Bbox-only lines (exactly 5 tokens) are removed — mixing bbox and polygon
        annotations in the same dataset triggers YOLO's 'segment counts ≠ box counts'
        warning which strips ALL segments and causes an index-out-of-bounds crash
        in the segmentation loss.  Truncated / malformed lines are also dropped.

        NOTE: We scan the dataset root directly (not via yaml paths) because many
        Roboflow-exported yamls use '../train/images' which resolves outside the dataset
        folder when parsed in Python, causing the yaml-based scan to miss all files.
        """
        dataset_root = data_yaml.parent
        total_removed = 0
        total_checked = 0

        # Collect all candidate label directories under the dataset root
        lbl_dirs: List[Path] = []
        for split_name in ("train", "valid", "val", "test"):
            candidate = dataset_root / split_name / "labels"
            if candidate.is_dir():
                lbl_dirs.append(candidate)
        # Also honour any labels/ dir sitting directly at dataset_root
        if (dataset_root / "labels").is_dir():
            lbl_dirs.append(dataset_root / "labels")

        if not lbl_dirs:
            job["logs"].append("[Validate] No label directories found — skipping validation")
            return

        total_files = sum(len(list(d.glob("*.txt"))) for d in lbl_dirs)
        job["logs"].append(f"[Validate] Checking {total_files} label file(s) across {len(lbl_dirs)} split(s)…")

        for lbl_dir in lbl_dirs:
            for lbl_file in lbl_dir.glob("*.txt"):
                total_checked += 1
                try:
                    raw = lbl_file.read_text(encoding="utf-8", errors="replace").strip()
                    if not raw:
                        continue  # empty = background image, OK
                    good, removed = [], 0
                    for line in raw.splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) >= 7:
                            good.append(line)   # valid polygon — keep
                        else:
                            removed += 1        # bbox-only or truncated — drop
                    if removed:
                        lbl_file.write_text("\n".join(good) + ("\n" if good else ""))
                        total_removed += removed
                except Exception as ex:
                    job["logs"].append(f"[Validate] {lbl_file.name}: {ex}")

        # Delete YOLO label cache files so the fixed labels are rescanned by YOLO
        # (stale .cache files make YOLO use the pre-fix data regardless of label changes)
        deleted_caches = 0
        for cache_file in dataset_root.rglob("labels.cache"):
            try:
                cache_file.unlink()
                deleted_caches += 1
            except Exception:
                pass

        if total_removed:
            job["logs"].append(
                f"[Validate] Removed {total_removed} non-polygon lines from {total_checked} files "
                f"(bbox-only / truncated lines cause mixed-dataset crash in segmentation training)"
            )
        else:
            job["logs"].append(f"[Validate] Labels OK — {total_checked} label files checked")
        if deleted_caches:
            job["logs"].append(f"[Validate] Cleared {deleted_caches} stale label cache(s)")

    @staticmethod
    def _labels_are_segmentation(data_yaml: Path) -> bool:
        """Return True if the dataset labels appear to contain polygon (segmentation) annotations."""
        try:
            import yaml as _yaml
            with open(data_yaml) as f:
                cfg = _yaml.safe_load(f)
            base = data_yaml.parent
            train_val = cfg.get("train") or cfg.get("val")
            if not train_val:
                return False
            img_dir = Path(train_val) if Path(train_val).is_absolute() else base / train_val
            lbl_dir = TrainingManager._get_label_dir(img_dir)
            if not lbl_dir:
                return False
            # Sample up to 20 label files
            for lbl_file in list(lbl_dir.glob("*.txt"))[:20]:
                content = lbl_file.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    first_line = content.splitlines()[0].split()
                    if len(first_line) > 5:
                        return True  # polygon format
            return False
        except Exception:
            return False

    @staticmethod
    def _convert_seg_to_bbox_yaml(data_yaml: Path, job: Dict) -> Optional[Path]:
        """Create a new data.yaml whose labels are bounding boxes derived from the
        segmentation polygons. Images are hard-linked (or copied) into a temp directory
        so that YOLO's images→labels path derivation still works.
        Returns the new data.yaml path, or None on failure."""
        import os, shutil
        try:
            import yaml as _yaml
            with open(data_yaml) as f:
                cfg = _yaml.safe_load(f)
        except Exception as e:
            job["logs"].append(f"[Seg→BBox] Cannot read data.yaml: {e}")
            return None

        base = data_yaml.parent
        out_dir = base / "_det_from_seg"
        out_dir.mkdir(exist_ok=True)
        new_cfg = dict(cfg)

        for split_key in ("train", "val", "test"):
            split_val = cfg.get(split_key)
            if not split_val:
                continue
            img_dir = Path(split_val) if Path(split_val).is_absolute() else base / split_val
            if not img_dir.exists():
                continue
            lbl_dir = TrainingManager._get_label_dir(img_dir)
            if not lbl_dir:
                job["logs"].append(f"[Seg→BBox] No label dir found for {img_dir}, skipping {split_key}")
                continue

            new_img_dir = out_dir / "images" / split_key
            new_lbl_dir = out_dir / "labels" / split_key
            new_img_dir.mkdir(parents=True, exist_ok=True)
            new_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Hard-link (or copy) image files so YOLO can find them
            IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in IMG_EXTS:
                    continue
                dest = new_img_dir / img_file.name
                if not dest.exists():
                    try:
                        os.link(img_file, dest)
                    except (OSError, NotImplementedError, AttributeError):
                        shutil.copy2(img_file, dest)

            # Convert labels: polygon → bbox
            converted = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                try:
                    content = lbl_file.read_text(encoding="utf-8", errors="replace").strip()
                    new_lines: List[str] = []
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) == 5:
                            new_lines.append(line)   # already bbox
                        elif len(parts) >= 7:
                            class_id = parts[0]
                            coords = list(map(float, parts[1:]))
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            cx = (x_min + x_max) / 2
                            cy = (y_min + y_max) / 2
                            w = x_max - x_min
                            h = y_max - y_min
                            new_lines.append(
                                f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                            )
                            converted += 1
                    (new_lbl_dir / lbl_file.name).write_text(
                        "\n".join(new_lines) + ("\n" if new_lines else "")
                    )
                except Exception as ex:
                    job["logs"].append(f"[Seg→BBox] {lbl_file.name}: {ex}")

            new_cfg[split_key] = str(new_img_dir)
            job["logs"].append(
                f"[Seg→BBox] {split_key}: converted {converted} polygons → bboxes"
            )

        new_yaml = out_dir / "data.yaml"
        try:
            import yaml as _yaml
            with open(new_yaml, "w") as f:
                _yaml.dump(new_cfg, f, default_flow_style=False)
            job["logs"].append(f"[Seg→BBox] Detection data.yaml → {new_yaml}")
            return new_yaml
        except Exception as e:
            job["logs"].append(f"[Seg→BBox] Failed to write data.yaml: {e}")
            return None

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _train_yolo(self, job: Dict):
        try:
            from ultralytics import YOLO
        except ImportError:
            import subprocess
            job["logs"].append("Installing ultralytics…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
            from ultralytics import YOLO

        config        = job["config"]
        dataset_path  = Path(job["dataset_path"])

        # ── Dataset preparation phase (status stays "starting" so UI shows "Preparing dataset…")
        job["logs"].append("Scanning dataset…")
        data_yaml = self._find_yaml(dataset_path, job)
        if not data_yaml:
            return

        # Auto-convert segmentation dataset to detection bbox format
        if self._labels_are_segmentation(data_yaml):
            job["logs"].append(
                "Dataset has segmentation (polygon) labels — auto-converting to "
                "bounding boxes for detection training."
            )
            converted_yaml = self._convert_seg_to_bbox_yaml(data_yaml, job)
            if converted_yaml:
                data_yaml = converted_yaml
            else:
                job["logs"].append(
                    "Seg→BBox conversion failed; attempting to train with original labels."
                )

        job["logs"].append("Dataset ready — starting training.")
        job["status"] = "running"

        device = _resolve_device(config, job)

        arch = config.get("base_model", "yolov8n.pt")
        job["logs"].append(f"Base model: {arch}")

        model = YOLO(arch)
        cbs = _make_callbacks(job)
        for name, fn in cbs.items():
            model.add_callback(name, fn)

        output_dir = dataset_path / "runs" / f"train_{job['id']}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=config.get("epochs", 100),
                batch=config.get("batch_size", 16),
                imgsz=config.get("image_size", 640),
                device=device,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                verbose=False,
                workers=0,  # avoid multiprocessing issues in daemon threads on Windows
                # advanced hypers
                lr0=config.get("lr0", 0.01),
                lrf=config.get("lrf", 0.01),
                optimizer=config.get("optimizer", "SGD"),
                patience=config.get("patience", 50),
                cos_lr=config.get("cos_lr", False),
                warmup_epochs=config.get("warmup_epochs", 3),
                weight_decay=config.get("weight_decay", 0.0005),
                mosaic=config.get("mosaic", 1.0),
                hsv_h=config.get("hsv_h", 0.015),
                hsv_s=config.get("hsv_s", 0.7),
                hsv_v=config.get("hsv_v", 0.4),
                flipud=config.get("flipud", 0.0),
                fliplr=config.get("fliplr", 0.5),
                amp=config.get("amp", True),
                dropout=config.get("dropout", 0.0),
            )
            best_pt = output_dir / "train" / "weights" / "best.pt"
            last_pt = output_dir / "train" / "weights" / "last.pt"
            if job.get("status") not in ("paused", "pausing", "stopped"):
                job["model_path"] = str(best_pt if best_pt.exists() else last_pt)
                job["status"]     = "completed"
                job["progress"]   = 100
                job["logs"].append("✓ Detection training complete")
                job["logs"].append(f"Weights: {job['model_path']}")
                if hasattr(results, "results_dict"):
                    job["metrics"].update(results.results_dict)
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)
            job["logs"].append(f"Training error: {e}")
            for pt_name in ("best.pt", "last.pt"):
                pt = output_dir / "train" / "weights" / pt_name
                if pt.exists():
                    job["model_path"] = str(pt)
                    job["logs"].append(f"Partial weights available: {pt}")
                    break
        finally:
            del model
            self._release_memory(job)

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _train_segmentation(self, job: Dict):
        try:
            from ultralytics import YOLO
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
            from ultralytics import YOLO

        config        = job["config"]
        dataset_path  = Path(job["dataset_path"])

        # ── Dataset preparation phase (status stays "starting" so UI shows "Preparing dataset…")
        job["logs"].append("Scanning dataset…")
        data_yaml = self._find_yaml(dataset_path, job)
        if not data_yaml:
            return

        # Validate and fix segmentation labels before training to prevent DataLoader IndexErrors
        job["logs"].append("Validating segmentation labels…")
        self._validate_seg_labels(data_yaml, job)
        job["logs"].append("Dataset ready — starting training.")

        job["status"] = "running"

        device = _resolve_device(config, job)

        arch = config.get("base_model", "yolov8n-seg.pt")
        job["logs"].append(f"Base model: {arch}")

        model = YOLO(arch)
        cbs = _make_callbacks(job)
        for name, fn in cbs.items():
            model.add_callback(name, fn)

        output_dir = dataset_path / "runs" / f"segment_{job['id']}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=config.get("epochs", 100),
                batch=config.get("batch_size", 16),
                imgsz=config.get("image_size", 640),
                device=device,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                verbose=False,
                workers=0,  # avoid multiprocessing issues in daemon threads on Windows
                lr0=config.get("lr0", 0.01),
                lrf=config.get("lrf", 0.01),
                optimizer=config.get("optimizer", "SGD"),
                patience=config.get("patience", 50),
                cos_lr=config.get("cos_lr", False),
                warmup_epochs=config.get("warmup_epochs", 3),
                weight_decay=config.get("weight_decay", 0.0005),
                mosaic=config.get("mosaic", 1.0),
                hsv_h=config.get("hsv_h", 0.015),
                hsv_s=config.get("hsv_s", 0.7),
                hsv_v=config.get("hsv_v", 0.4),
                flipud=config.get("flipud", 0.0),
                fliplr=config.get("fliplr", 0.5),
                amp=config.get("amp", True),
                dropout=config.get("dropout", 0.0),
            )
            # Use last.pt if best.pt wasn't saved (e.g. early stop)
            best_pt = output_dir / "train" / "weights" / "best.pt"
            last_pt = output_dir / "train" / "weights" / "last.pt"
            if job.get("status") not in ("paused", "pausing", "stopped"):
                job["model_path"] = str(best_pt if best_pt.exists() else last_pt)
                job["status"]     = "completed"
                job["progress"]   = 100
                job["logs"].append("✓ Segmentation training complete")
                job["logs"].append(f"Weights: {job['model_path']}")
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)
            job["logs"].append(f"Training error: {e}")
            # Still expose any partial weights
            for pt_name in ("best.pt", "last.pt"):
                pt = output_dir / "train" / "weights" / pt_name
                if pt.exists():
                    job["model_path"] = str(pt)
                    job["logs"].append(f"Partial weights available: {pt}")
                    break
        finally:
            del model
            self._release_memory(job)

    # ── Classification ────────────────────────────────────────────────────────

    def _train_classification(self, job: Dict):
        try:
            from ultralytics import YOLO
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
            from ultralytics import YOLO

        job["status"] = "running"
        config        = job["config"]
        dataset_path  = Path(job["dataset_path"])
        device        = _resolve_device(config, job)

        arch = config.get("base_model", "yolov8n-cls.pt")
        job["logs"].append(f"Base model: {arch}")

        model = YOLO(arch)
        cbs = _make_callbacks(job)
        for name, fn in cbs.items():
            model.add_callback(name, fn)

        output_dir = dataset_path / "runs" / f"classify_{job['id']}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            model.train(
                data=str(dataset_path),
                epochs=config.get("epochs", 100),
                batch=config.get("batch_size", 16),
                imgsz=config.get("image_size", 224),
                device=device,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                verbose=False,
                workers=0,  # avoid multiprocessing issues in daemon threads on Windows
            )
            if job.get("status") not in ("paused", "pausing", "stopped"):
                job["status"]     = "completed"
                job["progress"]   = 100
                job["model_path"] = str(output_dir / "train" / "weights" / "best.pt")
                job["logs"].append("✓ Classification training complete")
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)
            job["logs"].append(f"Training error: {e}")
        finally:
            del model
            self._release_memory(job)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_yaml(dataset_path: Path, job: Dict) -> Optional[Path]:
        data_yaml = None
        for candidate in list(dataset_path.rglob("data.yaml")) + list(dataset_path.rglob("*.yaml")):
            if candidate.name.startswith("."):
                continue
            data_yaml = candidate
            if candidate.name == "data.yaml":
                break
        if not data_yaml:
            job["status"] = "failed"
            job["error"]  = "No data.yaml found — use the YAML Wizard to create one"
            job["logs"].append(job["error"])
        else:
            job["logs"].append(f"Data config: {data_yaml}")
        return data_yaml

    # ── Public API ────────────────────────────────────────────────────────────

    def get_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        if training_id not in self.training_jobs:
            return None
        job = self.training_jobs[training_id]
        return {
            "id":            job.get("id", training_id),
            "name":          job.get("name", f"Run {training_id}"),
            "status":        job.get("status", "unknown"),
            "progress":      job.get("progress", 0),
            "current_epoch": job.get("current_epoch", 0),
            "total_epochs":  job.get("total_epochs", 0),
            "metrics":       job.get("metrics", {}),
            "epoch_history": job.get("epoch_history", []),
            "started_at":    job.get("started_at", ""),
            "model_path":    job.get("model_path"),
            "device_info":   job.get("device_info"),
            "gpu_mem_gb":    job.get("gpu_mem_gb"),
            "logs":          job.get("logs", [])[-50:],
            "error":         job.get("error"),
        }

    def stop_training(self, training_id: str) -> bool:
        if training_id not in self.training_jobs:
            return False
        job = self.training_jobs[training_id]
        job["_stop_requested"] = True
        job["status"] = "stopped"
        job["logs"].append("Training stopped by user — waiting for current epoch to finish.")
        # Also try to set trainer.stop directly if trainer is accessible
        trainer = job.get("_trainer")
        if trainer is not None:
            try:
                trainer.stop = True
            except Exception:
                pass
        self._persist_jobs()
        return True

    def pause_training(self, training_id: str) -> bool:
        """Stop training after current epoch and mark as paused (checkpoint saved)."""
        if training_id not in self.training_jobs:
            return False
        job = self.training_jobs[training_id]
        if job.get("status") not in ("running", "starting"):
            return False
        job["_stop_requested"] = True
        job["status"] = "pausing"   # intermediate — transitions to "paused" in on_fit_epoch_end
        job["logs"].append("Pausing… finishing current epoch before stopping.")
        trainer = job.get("_trainer")
        if trainer is not None:
            try:
                trainer.stop = True
            except Exception:
                pass
        self._persist_jobs()
        return True

    def resume_training(self, training_id: str) -> Optional[str]:
        """Resume a paused or interrupted training job from the last saved checkpoint."""
        if training_id not in self.training_jobs:
            return None
        job = self.training_jobs[training_id]
        if job.get("status") not in ("paused", "interrupted"):
            return None

        # Find the checkpoint to resume from
        last_checkpoint = job.get("last_checkpoint") or job.get("model_path")
        if not last_checkpoint or not Path(last_checkpoint).exists():
            dataset_path = Path(job["dataset_path"])
            job_id = job["id"]
            for pattern in [f"runs/segment_{job_id}/train/weights/last.pt",
                             f"runs/train_{job_id}/train/weights/last.pt",
                             f"runs/classify_{job_id}/train/weights/last.pt"]:
                candidate = dataset_path / pattern
                if candidate.exists():
                    last_checkpoint = str(candidate)
                    break

        if not last_checkpoint or not Path(last_checkpoint).exists():
            return None

        mt = job["model_type"]
        if mt == "rfdetr":
            job["logs"].append("RF-DETR does not support checkpoint resume.")
            return None

        # Clear stop flag and reset to running
        job["_stop_requested"] = False
        job["_trainer"] = None
        job["status"] = "running"
        job["last_checkpoint"] = last_checkpoint
        job["logs"].append(f"Resuming from checkpoint: {last_checkpoint}")
        self._persist_jobs()

        def _resume():
            _restore_syspath()
            try:
                from ultralytics import YOLO
                model = YOLO(last_checkpoint)
                cbs = _make_callbacks(job)
                for cb_name, fn in cbs.items():
                    model.add_callback(cb_name, fn)
                model.train(resume=True, workers=0)
                if job.get("status") not in ("paused", "pausing", "stopped"):
                    job["status"]   = "completed"
                    job["progress"] = 100
                    job["logs"].append("✓ Training resumed and completed.")
            except Exception as e:
                if job.get("status") not in ("paused", "pausing", "stopped"):
                    job["status"] = "failed"
                    job["error"]  = str(e)
                    job["logs"].append(f"Resume error: {e}")
            finally:
                try:
                    del model
                except Exception:
                    pass
                TrainingManager._release_memory(job)
                TrainingManager._instance_persist(job, self)

        thread = threading.Thread(target=_resume, daemon=True)
        thread.start()
        self.training_threads[training_id] = thread
        return training_id

    @staticmethod
    def _instance_persist(job: Dict, manager: "TrainingManager") -> None:
        """Helper so the _resume closure can call _persist_jobs without capturing self."""
        try:
            manager._persist_jobs()
        except Exception:
            pass

    def export_model_format(self, training_id: str, fmt: str) -> Optional[str]:
        """Export a trained model to the given format (onnx, tflite, coreml, engine)."""
        if training_id not in self.training_jobs:
            return None
        job = self.training_jobs[training_id]
        model_path = job.get("model_path")
        if not model_path or not Path(model_path).exists():
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            exported = model.export(format=fmt)
            return str(exported)
        except Exception as e:
            job["logs"].append(f"Export to {fmt} failed: {e}")
            return None

    def get_model_path(self, training_id: str) -> Optional[str]:
        if training_id not in self.training_jobs:
            return None
        return self.training_jobs[training_id].get("model_path")

    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """Return a summary of all training jobs, ordered newest-first."""
        jobs = []
        for job in self.training_jobs.values():
            job_id = job["id"]
            jobs.append({
                "id":            job_id,
                "name":          job.get("name", f"Run {job_id}"),
                "status":        job["status"],
                "progress":      job["progress"],
                "current_epoch": job.get("current_epoch", 0),
                "total_epochs":  job.get("total_epochs", 0),
                "model_type":    job["model_type"],
                "started_at":    job["started_at"],
                "model_path":    job.get("model_path"),
                "error":         job.get("error"),
                "last_checkpoint": job.get("last_checkpoint"),
            })
        # newest first
        jobs.sort(key=lambda j: j["started_at"], reverse=True)
        return jobs
