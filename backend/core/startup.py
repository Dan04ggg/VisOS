"""Startup helpers: package auto-install, CUDA check, temp cleanup, and batch-job persistence."""

import json
import threading
from pathlib import Path

from config import settings


def _ensure_packages(packages: list) -> None:
    """Auto-install missing Python packages. packages = [(import_name, pip_name), ...]"""
    import importlib, subprocess, sys
    missing = [pip for imp, pip in packages if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[startup] Auto-installing missing packages: {missing}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=False,
        )


def _ensure_cuda_torch() -> None:
    """
    If NVIDIA GPU hardware is present but the installed PyTorch lacks CUDA support,
    automatically install the CUDA-enabled wheel and restart the backend process.
    Runs in a background daemon thread so server startup is never blocked.
    """
    import subprocess, sys, os, pathlib, time

    # 1. Detect GPU hardware via nvidia-smi (doesn't need torch)
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0 or "GPU" not in r.stdout:
            settings._gpu_status["state"] = "no_gpu"
            return
        gpu_line = r.stdout.strip().splitlines()[0]
        settings._gpu_status["gpu_name"] = gpu_line
    except Exception:
        settings._gpu_status["state"] = "no_gpu"
        return

    # 2. Check if current torch already has CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            msg = f"CUDA ready: {name} (torch {torch.__version__}, CUDA {torch.version.cuda})"
            print(f"[GPU] {msg}")
            settings._gpu_status.update({"state": "ready", "message": msg, "gpu_name": name})
            return
        torch_ver = torch.__version__
        print(f"[GPU] GPU found ({gpu_line}) but torch {torch_ver} has no CUDA support.")
    except ImportError:
        torch_ver = "not installed"
        print(f"[GPU] GPU found ({gpu_line}) but torch is not installed yet.")

    # 3. Auto-install CUDA-enabled PyTorch
    cuda_ver = "cu124"
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        drv = float(nv.stdout.strip().split(".")[0]) if nv.returncode == 0 else 0
        cuda_ver = "cu124" if drv >= 545 else "cu121"
    except Exception:
        pass

    wheel_url = f"https://download.pytorch.org/whl/{cuda_ver}"
    settings._gpu_status.update({
        "state": "installing",
        "message": f"Installing CUDA PyTorch ({cuda_ver})… backend will restart when done. "
                   f"Inference uses CPU until then.",
        "gpu_name": gpu_line,
    })
    print(f"[GPU] Auto-installing CUDA PyTorch ({cuda_ver}) from {wheel_url} ...")

    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", wheel_url,
            "--upgrade", "--quiet",
        ],
        timeout=900,
        check=False,
    )

    if result.returncode == 0:
        print("[GPU] CUDA PyTorch installed. Triggering backend restart via WatchFiles...")
        settings._gpu_status["message"] = "Install done — restarting backend now…"
        try:
            pathlib.Path(__file__).touch()
        except Exception:
            pass
        time.sleep(3)
        os._exit(0)
    else:
        install_error = (result.stderr or result.stdout or "unknown error")[-400:]
        print(f"[GPU] CUDA PyTorch install failed: {install_error}")
        settings._gpu_status.update({
            "state": "failed",
            "message": f"Auto-install failed. Run manually:\n"
                       f"pip install torch torchvision --index-url {wheel_url}\n"
                       f"then restart the backend.",
        })


def _cleanup_temp() -> None:
    """Delete all subdirectories in workspace/temp to reclaim disk space on startup."""
    import shutil
    try:
        removed, freed = 0, 0
        for entry in settings.TEMP_DIR.iterdir():
            if entry.is_dir():
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                shutil.rmtree(entry, ignore_errors=True)
                freed += size
                removed += 1
        if removed:
            print(f"[startup] Cleaned temp dir: removed {removed} folder(s), freed {freed // (1024 * 1024)} MB.")
    except Exception as e:
        print(f"[startup] Temp cleanup error: {e}")


def _persist_jobs() -> None:
    """Persist batch job state to disk so it survives server restarts."""
    try:
        with open(settings.JOBS_FILE, "w") as f:
            json.dump(settings._text_annotate_jobs, f, indent=2, default=str)
    except Exception:
        pass


def _restore_jobs() -> None:
    """On startup, restore persisted batch jobs from disk.
    Any jobs that were running or paused are marked 'interrupted'."""
    if not settings.JOBS_FILE.exists():
        return
    try:
        with open(settings.JOBS_FILE) as f:
            jobs = json.load(f)
        count = 0
        for job_id, job in jobs.items():
            if job.get("status") in ("running", "paused"):
                job["status"] = "interrupted"
                job["paused"] = False
            settings._text_annotate_jobs[job_id] = job
            count += 1
        if count:
            print(f"[startup] Restored {count} batch job(s) from disk.")
    except Exception:
        pass
