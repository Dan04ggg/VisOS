#!/usr/bin/env python3
"""
Cross-platform uvicorn launcher with hot-reload that works on all Python versions.

Called by ../run.py instead of ``python -m uvicorn`` so we can apply two fixes
without touching uvicorn's source:

  1. Python ≥ 3.14 pathlib compatibility
     Python 3.14 made pathlib.Path.glob() reject non-relative (absolute) patterns.
     Uvicorn's resolve_reload_patterns() calls cwd.glob(pattern) where pattern
     can be an absolute path, crashing at startup.  We restore the old behaviour:
     if the pattern is absolute, just yield the path itself when it exists.

  2. Reliable venv / build-directory exclusion from hot-reload
     Uvicorn's FileFilter (the watch_filter used by WatchFilesReload) compares
     excluded directories against path.parents.  That comparison only works when
     both sides are absolute paths — a relative Path('venv') never matches an
     absolute Path('/backend/venv') even when it is a direct parent.

     The fix: before uvicorn parses sys.argv, we inject ``--reload-exclude``
     flags with *absolute* paths for each generated directory.  FileFilter then
     resolves them via p.is_dir() → exclude_dirs, and the parents check works.

IMPORTANT — multiprocessing / Windows note
==========================================
When uvicorn's --reload supervisor spawns a worker process on Windows (which uses
the "spawn" start method instead of "fork"), Python re-imports this script as the
module named __mp_main__.  Any top-level code that starts a new process would
therefore recurse infinitely and hit the
  RuntimeError: _check_not_importing_main()
guard inside multiprocessing.

The rule: code that starts uvicorn (or anything involving multiprocessing) MUST
live inside ``if __name__ == '__main__':``.

The pathlib shim (fix 1) is safe at module level because it is idempotent and
does not spawn anything — it must run in the worker process too so that any
path-related code there also gets the compat fix.
"""
from __future__ import annotations

import sys
import pathlib
from pathlib import Path


# ── 1. Python 3.14 pathlib.glob compatibility shim ────────────────────────────
# Applied at module level so it is active in both the supervisor process and
# any worker processes that re-import this script.
if sys.version_info >= (3, 14):
    _orig_glob = pathlib.Path.glob

    def _glob_compat(self, pattern, **kw):          # type: ignore[override]
        try:
            yield from _orig_glob(self, pattern, **kw)
        except NotImplementedError:
            # Absolute pattern: just yield the resolved path if it exists.
            p = pathlib.Path(pattern)
            if p.exists():
                yield p

    pathlib.Path.glob = _glob_compat  # type: ignore[method-assign]


# ── 2. Inject --reload-exclude flags + uvicorn entry point ───────────────────
# Everything below starts processes, so it must be guarded by __name__ == '__main__'.
# Worker processes re-import this file as '__mp_main__' and must NOT re-enter here.
if __name__ == '__main__':

    # Inject absolute --reload-exclude paths before uvicorn parses sys.argv.
    #
    # Why absolute?  FileFilter stores these as Path objects and later checks:
    #   exclude_dir in path.parents
    # That comparison only succeeds when both sides are absolute — a relative
    # Path('venv') never equals Path('C:/backend/venv') in a parents list.
    #
    # We only inject dirs that currently exist so they are treated as directory
    # excludes (p.is_dir() → True → exclude_dirs) rather than glob patterns.
    # Dirs that don't exist yet (e.g. 'runs' on first launch) are harmless to
    # skip; they contain no source files to watch anyway.
    _BACKEND_DIR = Path(__file__).parent.resolve()
    _EXCLUDE_DIR_NAMES = ("venv", "__pycache__", "runs", "workspace", ".git")
    for _name in _EXCLUDE_DIR_NAMES:
        _excl = _BACKEND_DIR / _name
        if _excl.is_dir():
            sys.argv.extend(["--reload-exclude", str(_excl)])

    from uvicorn.main import main as _uvicorn_main  # noqa: E402
    sys.exit(_uvicorn_main())
