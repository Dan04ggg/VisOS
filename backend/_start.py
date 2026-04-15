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
     Uvicorn resolves --reload-exclude directories into a separate
     reload_dirs_excludes list, but the watchfiles DefaultFilter it installs only
     reads from reload_excludes (string patterns).  Directory-based excludes are
     therefore silently ignored, causing spurious reloads whenever pip writes into
     the venv.  We monkey-patch WatchFilesReload to install a custom filter that
     rejects any path whose components include an excluded directory name.

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


# ── 2 & 3. Watchfiles patch + uvicorn entry point — supervisor process only ───
# Everything below starts processes, so it must be guarded by __name__ == '__main__'.
# Worker processes re-import this file as '__mp_main__' and must NOT re-enter here.
if __name__ == '__main__':

    _EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
        {"venv", "__pycache__", "runs", "workspace", ".git", "node_modules"}
    )

    # Patch WatchFilesReload before uvicorn instantiates the supervisor so
    # our directory exclusions are active without any CLI flags.
    #
    # The installed uvicorn uses FileFilter (not watchfiles.DefaultFilter) with
    # the signature: __call__(self, path: Path) -> bool.  We must match that
    # interface — wrapping the original FileFilter so uvicorn's include/exclude
    # patterns still work, while we bolt on directory-name exclusions.
    #
    # Import order matters: importing uvicorn.supervisors.watchfilesreload also
    # triggers uvicorn/supervisors/__init__.py, which binds:
    #   ChangeReload = WatchFilesReload   (a direct reference)
    # uvicorn/main.py then does `from uvicorn.supervisors import ChangeReload`,
    # capturing that reference by value.  Patching only the module attribute
    # (_wfr.WatchFilesReload) is therefore not enough — we must also update the
    # ChangeReload alias in uvicorn.supervisors BEFORE uvicorn.main is imported.
    try:
        import uvicorn.supervisors.watchfilesreload as _wfr   # type: ignore
        import uvicorn.supervisors as _supervisors_pkg         # type: ignore

        _OrigReload = _wfr.WatchFilesReload

        class _PatchedReload(_OrigReload):  # type: ignore[misc]
            def __init__(self, config, target, sockets):
                super().__init__(config, target, sockets)
                # self.watch_filter is now a FileFilter instance set by super().
                # Wrap it so excluded directory names are rejected first.
                _orig_filter = self.watch_filter

                class _WrappedFilter:
                    def __call__(self_, path: Path) -> bool:  # noqa: N805
                        if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
                            return False
                        return _orig_filter(path)

                self.watch_filter = _WrappedFilter()

        # Patch both locations so every reference resolves to _PatchedReload.
        _wfr.WatchFilesReload = _PatchedReload          # type: ignore[attr-defined]
        _supervisors_pkg.ChangeReload = _PatchedReload  # type: ignore[attr-defined]

    except (ImportError, AttributeError):
        pass  # watchfiles inactive or different uvicorn internal layout

    from uvicorn.main import main as _uvicorn_main  # noqa: E402
    sys.exit(_uvicorn_main())
