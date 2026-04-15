#!/usr/bin/env python3
"""
CV Dataset Manager - Process Manager

Commands
--------
  python run.py start          Start both servers (default)
  python run.py stop           Kill both servers cleanly
  python run.py restart        Stop then start
  python run.py restart-back   Restart only the backend
  python run.py restart-front  Restart only the frontend
  python run.py status         Show what is running
  python run.py logs           Tail live logs from both processes
"""

import os
import sys
import signal
import subprocess
import time
import threading
import json
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.absolute()
BACKEND_DIR = ROOT / "backend"
PID_FILE    = ROOT / ".pids.json"       # tracks { "backend": PID, "frontend": PID }
LOG_DIR     = ROOT / ".logs"
BACK_LOG    = LOG_DIR / "backend.log"
FRONT_LOG   = LOG_DIR / "frontend.log"
FIRST_RUN_MARKER = ROOT / ".first_run_complete"

LOG_DIR.mkdir(exist_ok=True)

IS_WIN = sys.platform == "win32"


# ── Colour helpers ─────────────────────────────────────────────────────────────
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
C  = "\033[96m"
B  = "\033[94m"
W  = "\033[0m"
BOLD = "\033[1m"

def ok(msg):    print(f"  {G}✓{W}  {msg}")
def info(msg):  print(f"  {C}→{W}  {msg}")
def warn(msg):  print(f"  {Y}!{W}  {msg}")
def err(msg):   print(f"  {R}✗{W}  {msg}")
def head(msg):  print(f"\n{BOLD}{C}{msg}{W}")


# ── Live spinner ───────────────────────────────────────────────────────────────
class _Spinner:
    """Animated terminal spinner with a live-updating status line.

    Usage::

        spinner = _Spinner("Downloading packages…").start()
        spinner.update("Installing torch…")
        spinner.stop("Packages installed.")   # clears the line, then prints ✓
    """

    _FRAMES = ("|", "/", "-", "\\")   # ASCII — safe on every platform/encoding

    def __init__(self, initial_msg: str = ""):
        self._msg = initial_msg
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"\r  {C}{frame}{W}  {self._msg}   "
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(0.1)
            i += 1

    def update(self, msg: str):
        """Replace the status text shown beside the spinner."""
        self._msg = msg

    def start(self) -> "_Spinner":
        self._thread.start()
        return self

    def stop(self, final_msg: str = ""):
        """Stop the spinner and optionally print a success line."""
        self._stop_event.set()
        self._thread.join(timeout=0.5)
        # Erase the spinner line completely before printing the final status
        try:
            sys.stdout.write(f"\r{' ' * 72}\r")
            sys.stdout.flush()
        except Exception:
            pass
        if final_msg:
            ok(final_msg)


# ── PID file helpers ───────────────────────────────────────────────────────────
def read_pids() -> dict:
    try:
        return json.loads(PID_FILE.read_text())
    except Exception:
        return {}


def write_pids(pids: dict):
    PID_FILE.write_text(json.dumps(pids, indent=2))


def clear_pids():
    PID_FILE.unlink(missing_ok=True)


def is_alive(pid: int) -> bool:
    """Cross-platform: return True if a process with this PID is still running."""
    if pid <= 0:
        return False
    try:
        if IS_WIN:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)   # signal 0 = just check, don't kill
            return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ── Kill helpers ───────────────────────────────────────────────────────────────
def kill_pid(pid: int, name: str = "process"):
    """Gracefully terminate then force-kill if needed."""
    if not is_alive(pid):
        return
    info(f"Stopping {name} (PID {pid})…")
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
            # Give it 5 s to exit gracefully
            for _ in range(50):
                time.sleep(0.1)
                if not is_alive(pid):
                    break
            else:
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        warn(f"Could not kill {name}: {e}")


def kill_port(port: int):
    """Kill whatever is listening on a port — catches orphaned processes."""
    try:
        if IS_WIN:
            r = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            for line in r.stdout.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    subprocess.run(["taskkill", "/F", "/PID", parts[-1]],
                                   capture_output=True)
        else:
            subprocess.run(f"lsof -ti:{port} | xargs -r kill -9",
                           shell=True, capture_output=True)
    except Exception:
        pass


# ── Venv / dependency helpers ─────────────────────────────────────────────────
def _venv_paths():
    venv = BACKEND_DIR / "venv"
    if IS_WIN:
        return venv, venv / "Scripts" / "python.exe", venv / "Scripts" / "pip.exe"
    return venv, venv / "bin" / "python", venv / "bin" / "pip"


def _find_python() -> str:
    """Return a working Python 3 executable on the current machine."""
    import shutil
    for candidate in ("python3", "python", sys.executable):
        exe = shutil.which(candidate) or candidate
        try:
            out = subprocess.check_output(
                [exe, "-c", "import sys; print(sys.version_info[0])"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if out == "3":
                return exe
        except Exception:
            continue
    return None


def _venv_healthy(python_bin: Path) -> bool:
    """Return True if the venv Python binary exists AND actually runs."""
    if not python_bin.exists():
        return False
    try:
        result = subprocess.run(
            [str(python_bin), "-c", "import sys; print(sys.version_info[0])"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "3"
    except Exception:
        return False


def ensure_backend_deps():
    """Create venv if missing or broken, then install requirements with a live progress spinner."""
    venv, python_bin, pip_bin = _venv_paths()

    if not _venv_healthy(python_bin):
        if venv.exists():
            info("Existing virtual environment appears broken or machine-specific — recreating…")
            import shutil as _shutil
            _shutil.rmtree(venv, ignore_errors=True)

        system_python = _find_python()
        if not system_python:
            err(
                "Python 3 is not installed or not on PATH.\n"
                "  Please install Python 3.9+ from https://python.org"
            )
            sys.exit(1)

        spinner = _Spinner("Creating Python virtual environment…").start()
        result = subprocess.run(
            [system_python, "-m", "venv", str(venv)],
            capture_output=True,
        )
        spinner.stop()
        if result.returncode != 0:
            err("Failed to create virtual environment.")
            sys.stderr.buffer.write(result.stderr)
            sys.exit(1)
        ok("Virtual environment created.")

    spinner = _Spinner("Checking Python packages…").start()

    proc = subprocess.Popen(
        [
            str(pip_bin), "install", "-r", "requirements.txt",
            "--disable-pip-version-check", "--progress-bar", "off",
        ],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = []
    for raw in iter(proc.stdout.readline, ""):
        line = raw.strip()
        if not line:
            continue
        output_lines.append(line)
        # Surface the most meaningful pip progress lines as status updates
        if line.startswith("Collecting "):
            pkg = line[len("Collecting "):].split()[0]
            spinner.update(f"Collecting {pkg}…")
        elif line.startswith("Downloading "):
            parts = line[len("Downloading "):].split()
            pkg = parts[0] if parts else "package"
            spinner.update(f"Downloading {pkg}…")
        elif line.startswith("Installing collected packages:"):
            pkgs = line[len("Installing collected packages:"):].strip()
            # Truncate long lists so the line stays readable
            display = pkgs if len(pkgs) <= 48 else pkgs[:45] + "…"
            spinner.update(f"Installing {display}…")
        elif line.startswith("Successfully installed"):
            spinner.update("Finalizing…")

    proc.wait()
    spinner.stop()

    if proc.returncode != 0:
        err("Failed to install Python dependencies. Last output:")
        for line in output_lines[-20:]:
            print(f"    {line}")
        sys.exit(1)

    ok("Python dependencies ready.")
    return python_bin


def ensure_frontend_deps():
    """Install Node.js dependencies if node_modules is missing, with a live spinner."""
    if (ROOT / "node_modules").exists():
        return

    pm = "pnpm" if _has("pnpm") else "npm"
    spinner = _Spinner(f"Installing Node.js packages ({pm} install)…").start()

    proc = subprocess.Popen(
        f"{pm} install",
        cwd=ROOT,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = []
    for raw in iter(proc.stdout.readline, ""):
        line = raw.strip()
        if not line:
            continue
        output_lines.append(line)
        # pnpm/npm both emit lines like "added 123 packages" or package names
        if any(kw in line.lower() for kw in ("added", "packages", "resolved", "fetching")):
            display = line if len(line) <= 55 else line[:52] + "…"
            spinner.update(display)

    proc.wait()
    spinner.stop()

    if proc.returncode != 0:
        err(f"Failed to install Node.js dependencies ({pm} install). Last output:")
        for line in output_lines[-20:]:
            print(f"    {line}")
        sys.exit(1)

    ok("Node.js dependencies installed.")


def _has(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# ── Tee helper: forward a stream to both terminal and log file ────────────────
def _tee_stream(stream, log_file, prefix: str, colour: str, stop_event: threading.Event):
    """Read lines from *stream* and write them to both stdout and *log_file*."""
    for raw in iter(stream.readline, b""):
        if stop_event.is_set():
            break
        try:
            line = raw.decode("utf-8", errors="replace")
            log_file.write(line)
            log_file.flush()
        except Exception:
            pass
        try:
            # Encode for the Windows console (cp1252 etc.) — replace unencodable chars
            text = f"{colour}[{prefix}]{W} {line}"
            sys.stdout.buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            pass


# ── Process launchers ──────────────────────────────────────────────────────────
def start_backend(python_bin: Path):
    """Start uvicorn; returns (proc, tee_thread, stop_event)."""
    log = open(BACK_LOG, "a")
    separator = f"\n\n{'='*60}\n  BACKEND START  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n"
    log.write(separator)
    log.flush()
    sys.stdout.write(f"{C}[BACK ]{W} {separator}")

    # Use backend/_start.py instead of -m uvicorn.  The shim applies two fixes:
    #   • pathlib.glob compatibility for Python 3.14+ (no crash on startup)
    #   • a custom watchfiles filter so venv / __pycache__ / runs / workspace
    #     never trigger a hot-reload, regardless of uvicorn version
    # No --reload-exclude flags needed: the filter inside the shim handles it.
    start_script = BACKEND_DIR / "_start.py"

    cmd = [
        str(python_bin), str(start_script), "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
    ]
    kwargs = dict(cwd=BACKEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if IS_WIN:
        proc = subprocess.Popen(cmd, **kwargs,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, **kwargs, start_new_session=True)

    stop_event = threading.Event()
    tee = threading.Thread(
        target=_tee_stream,
        args=(proc.stdout, log, "BACK ", C, stop_event),
        daemon=True,
    )
    tee.start()
    return proc, tee, stop_event


def start_frontend():
    """Start the Next.js dev server; returns (proc, tee_thread, stop_event)."""
    log = open(FRONT_LOG, "a")
    separator = f"\n\n{'='*60}\n  FRONTEND START  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n"
    log.write(separator)
    log.flush()
    sys.stdout.write(f"{B}[FRONT]{W} {separator}")

    pm = "pnpm" if _has("pnpm") else "npm"
    cmd = f"{pm} run dev"
    kwargs = dict(cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    if IS_WIN:
        proc = subprocess.Popen(cmd, **kwargs,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, **kwargs, start_new_session=True)

    stop_event = threading.Event()
    tee = threading.Thread(
        target=_tee_stream,
        args=(proc.stdout, log, "FRONT", B, stop_event),
        daemon=True,
    )
    tee.start()
    return proc, tee, stop_event


def wait_for_backend(timeout=120) -> bool:
    """Poll /api/health until it responds or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8000/api/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def wait_for_frontend(timeout=120) -> bool:
    """Poll localhost:3000 until it responds or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:3000", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── Commands ───────────────────────────────────────────────────────────────────
def cmd_status():
    head("Process Status")
    pids = read_pids()

    back_pid  = pids.get("backend", 0)
    front_pid = pids.get("frontend", 0)

    if is_alive(back_pid):
        ok(f"Backend  running  (PID {back_pid})  →  http://localhost:8000")
    else:
        err(f"Backend  stopped  (last PID {back_pid or '—'})")

    if is_alive(front_pid):
        ok(f"Frontend running  (PID {front_pid})  →  http://localhost:3000")
    else:
        err(f"Frontend stopped  (last PID {front_pid or '—'})")

    print()
    info(f"Backend log:   {BACK_LOG}")
    info(f"Frontend log:  {FRONT_LOG}")
    print()


def cmd_stop(verbose=True):
    if verbose:
        head("Stopping Servers")
    pids = read_pids()

    kill_pid(pids.get("backend",  0), "backend")
    kill_pid(pids.get("frontend", 0), "frontend")

    # Belt-and-braces: also kill by port in case PID drifted
    kill_port(8000)
    kill_port(3000)

    # Remove Next.js dev-server lock so a restart can start cleanly
    for lock_name in ("lock", "server.lock"):
        next_lock = ROOT / ".next" / "dev" / lock_name
        if next_lock.exists():
            try:
                next_lock.unlink()
            except Exception:
                pass

    clear_pids()
    if verbose:
        ok("All servers stopped.")
        print()


def cmd_start(open_browser_flag=True):
    head("Starting VisOS")
    if not FIRST_RUN_MARKER.exists():
        print(f"  {Y}⚑{W}  {BOLD}First run:{W} packages need to install — this may take a few minutes.")
        print(f"     Future starts will be much faster.\n")

    # If any previously-tracked process is still alive (e.g. terminal was closed
    # while servers kept running in the background), stop them before re-launching.
    pids = read_pids()
    if is_alive(pids.get("backend", 0)) or is_alive(pids.get("frontend", 0)):
        info("Existing server processes detected — stopping them first…")
        cmd_stop(verbose=False)
        time.sleep(1)

    # Kill any orphaned processes on the ports before launching
    kill_port(8000)
    kill_port(3000)
    # Remove Next.js dev-server lock so a fresh instance can start cleanly
    for lock_name in ("lock", "server.lock"):
        next_lock = ROOT / ".next" / "dev" / lock_name
        if next_lock.exists():
            try:
                next_lock.unlink()
            except Exception:
                pass
    time.sleep(0.5)

    # Deps (each function shows its own spinner + final ok/err line)
    python_bin = ensure_backend_deps()
    ensure_frontend_deps()

    if not FIRST_RUN_MARKER.exists():
        FIRST_RUN_MARKER.touch()

    # Launch
    info("Launching backend…")
    back_proc,  _back_tee,  back_stop  = start_backend(python_bin)
    info("Launching frontend…")
    front_proc, _front_tee, front_stop = start_frontend()

    # Persist PIDs immediately so Ctrl+C can clean up
    write_pids({"backend": back_proc.pid, "frontend": front_proc.pid})

    # Health-check — both servers have up to 120 s to become ready
    spinner = _Spinner("Waiting for backend to become healthy…").start()
    backend_up = wait_for_backend(120)
    spinner.stop()
    if backend_up:
        ok("Backend is up   →  http://localhost:8000")
    else:
        warn("Backend did not respond within 120 s — check .logs/backend.log")

    spinner = _Spinner("Waiting for frontend to become ready…").start()
    frontend_up = wait_for_frontend(120)
    spinner.stop()
    if frontend_up:
        ok("Frontend is up  →  http://localhost:3000")
    else:
        warn("Frontend did not respond within 120 s — check .logs/frontend.log")

    print(f"""
{G}{BOLD}
╔══════════════════════════════════════════════════════╗
║               ALL SYSTEMS RUNNING                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Frontend UI:  http://localhost:3000                ║
║   Backend API:  http://localhost:8000                ║
║   API Docs:     http://localhost:8000/docs           ║
║                                                      ║
║   Logs:  .logs/backend.log  /  .logs/frontend.log    ║
║                                                      ║
║   Controls:                                          ║
║     python run.py stop            — stop everything  ║
║     python run.py restart         — full restart     ║
║     python run.py restart-back    — backend only     ║
║     python run.py restart-front   — frontend only    ║
║     python run.py status          — check PIDs       ║
║     python run.py logs            — tail live logs   ║
║                                                      ║
║   Press Ctrl+C here to stop both servers cleanly.    ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
{W}""")

    if open_browser_flag:
        threading.Thread(target=lambda: (time.sleep(1),
                                         webbrowser.open("http://localhost:3000")),
                         daemon=True).start()

    # Block and forward Ctrl+C cleanly
    try:
        while True:
            # If a child dies unexpectedly, tell the user
            if back_proc.poll() is not None:
                warn("Backend process exited unexpectedly.")
                warn("Run  python run.py restart-back  to bring it back.")
            if front_proc.poll() is not None:
                warn("Frontend process exited unexpectedly.")
                warn("Run  python run.py restart-front  to bring it back.")
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n{Y}Ctrl+C received — shutting down…{W}")
        back_stop.set()
        front_stop.set()
        cmd_stop(verbose=False)
        ok("Goodbye!")


def cmd_restart():
    head("Restarting All Servers")
    cmd_stop(verbose=False)
    time.sleep(1)
    cmd_start()


def cmd_restart_backend():
    head("Restarting Backend")
    pids = read_pids()
    kill_pid(pids.get("backend", 0), "backend")
    kill_port(8000)
    time.sleep(1)

    python_bin = ensure_backend_deps()
    proc, _tee, _stop = start_backend(python_bin)
    pids["backend"] = proc.pid
    write_pids(pids)

    spinner = _Spinner("Waiting for backend to become healthy…").start()
    backend_up = wait_for_backend(120)
    spinner.stop()
    if backend_up:
        ok(f"Backend restarted (PID {proc.pid})  →  http://localhost:8000")
    else:
        warn("Backend did not respond within 120 s — check .logs/backend.log")
    print()


def cmd_restart_frontend():
    head("Restarting Frontend")
    pids = read_pids()
    kill_pid(pids.get("frontend", 0), "frontend")
    kill_port(3000)
    time.sleep(1)

    ensure_frontend_deps()
    proc, _tee, _stop = start_frontend()
    pids["frontend"] = proc.pid
    write_pids(pids)

    spinner = _Spinner("Waiting for frontend to become ready…").start()
    frontend_up = wait_for_frontend(120)
    spinner.stop()
    if frontend_up:
        ok(f"Frontend restarted (PID {proc.pid})  →  http://localhost:3000")
    else:
        warn("Frontend did not respond within 120 s — check .logs/frontend.log")
    print()


def cmd_logs():
    """Tail both log files in parallel threads."""
    head("Live Logs  (Ctrl+C to stop)")
    print(f"  Backend  → {BACK_LOG}")
    print(f"  Frontend → {FRONT_LOG}\n")

    stop_event = threading.Event()

    def tail(path: Path, prefix: str, colour: str):
        BACK_LOG.touch(exist_ok=True)
        FRONT_LOG.touch(exist_ok=True)
        with open(path, "r", errors="replace") as f:
            f.seek(0, 2)   # seek to end
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    print(f"{colour}[{prefix}]{W} {line}", end="", flush=True)
                else:
                    time.sleep(0.1)

    threads = [
        threading.Thread(target=tail, args=(BACK_LOG,  "BACK ", C), daemon=True),
        threading.Thread(target=tail, args=(FRONT_LOG, "FRONT", B), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n{Y}Stopped log tail.{W}")


# ── Entry point ────────────────────────────────────────────────────────────────
USAGE = f"""
{BOLD}CV Dataset Manager — Process Manager{W}

  {C}python run.py{W}                  Start both servers (default)
  {C}python run.py start{W}            Start both servers
  {C}python run.py stop{W}             Stop both servers cleanly
  {C}python run.py restart{W}          Full stop → start cycle
  {C}python run.py restart-back{W}     Restart backend only   (after Python changes)
  {C}python run.py restart-front{W}    Restart frontend only  (rarely needed — HMR handles it)
  {C}python run.py status{W}           Show running PIDs and ports
  {C}python run.py logs{W}             Tail live output from both servers
"""

COMMANDS = {
    "start":         cmd_start,
    "stop":          cmd_stop,
    "restart":       cmd_restart,
    "restart-back":  cmd_restart_backend,
    "restart-front": cmd_restart_frontend,
    "status":        cmd_status,
    "logs":          cmd_logs,
    "--help":        lambda: print(USAGE),
    "-h":            lambda: print(USAGE),
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "start"
    fn  = COMMANDS.get(arg)
    if fn is None:
        err(f"Unknown command: {arg}")
        print(USAGE)
        sys.exit(1)
    fn()