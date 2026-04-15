#!/bin/bash
# CV Dataset Manager — delegates to run.py which manages both processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Prefer python3, fall back to python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

exec "$PYTHON" run.py "${@:-start}"
