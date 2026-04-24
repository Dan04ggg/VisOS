from fastapi import APIRouter

from config import settings

router = APIRouter()


@router.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0"}


@router.get("/api/device-info")
async def device_info():
    info: dict = {
        "device": settings.model_manager._get_device(),
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
        "gpu_status": settings._gpu_status,
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch_version"] = "not installed"
    return info


@router.get("/api/system")
async def system_stats():
    import asyncio, importlib

    stats: dict = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "gpu_percent": None,
        "gpu_memory_percent": None,
        "gpu_name": None,
    }

    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil
        loop = asyncio.get_event_loop()
        cpu_pct = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=0.5))
        stats["cpu_percent"] = cpu_pct
        vm = psutil.virtual_memory()
        stats["ram_percent"] = vm.percent

    pynvml_spec = importlib.util.find_spec("pynvml")
    if pynvml_spec is not None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            stats["gpu_percent"] = float(util.gpu)
            stats["gpu_memory_percent"] = round(mem_info.used / mem_info.total * 100, 1)
            stats["gpu_name"] = name if isinstance(name, str) else name.decode()
        except Exception:
            pass

    return stats


@router.get("/api/hardware")
async def hardware_info():
    import importlib, platform

    info: dict = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
        "cpu_cores": None,
        "cpu_model": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
    }

    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 1)

    try:
        if platform.system() == "Windows":
            import subprocess
            r = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=3,
            )
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    info["cpu_model"] = line.split("=", 1)[1].strip()
                    break
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            info["cpu_model"] = r.stdout.strip()
    except Exception:
        pass

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                for i in range(info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory // (1024 ** 2)
                    try:
                        free, _ = torch.cuda.mem_get_info(i)
                        mem_free = free // (1024 ** 2)
                    except Exception:
                        mem_free = None
                    info["gpus"].append({
                        "index": i,
                        "name": props.name,
                        "memory_total_mb": mem_total,
                        "memory_free_mb": mem_free,
                    })
        except Exception:
            pass

    if not info["gpus"] and importlib.util.find_spec("pynvml") is not None:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = count
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpus"].append({
                    "index": i,
                    "name": name if isinstance(name, str) else name.decode(),
                    "memory_total_mb": mem.total // (1024 ** 2),
                    "memory_free_mb": mem.free // (1024 ** 2),
                })
            if info["gpus"]:
                info["cuda_available"] = True
        except Exception:
            pass

    if not info["gpus"]:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                        })
                if info["gpus"]:
                    info["gpu_count"] = len(info["gpus"])
                    info["cuda_available"] = True
                    hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                    for line in hdr.stdout.splitlines():
                        if "CUDA Version:" in line:
                            import re
                            m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                            if m:
                                info["cuda_version"] = m.group(1)
                            break
        except Exception:
            pass

    return info
