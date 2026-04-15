#!/bin/bash

# CV Dataset Manager - Backend Startup Script
# NOTE: prefer using `python run.py` from the project root — it handles
# both backend and frontend in one command.  This script is a lower-level
# backend-only fallback.

set -e

echo "Starting CV Dataset Manager Backend..."

# Navigate to backend directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../backend"

# Locate a working Python 3
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Create (or validate) virtual environment
VENV_PYTHON="venv/bin/python"
if [ ! -f "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import sys; assert sys.version_info[0]==3" 2>/dev/null; then
    echo "Creating/refreshing virtual environment..."
    rm -rf venv
    "$PYTHON" -m venv venv
fi

# Install/update dependencies using the venv Python directly (no activation needed)
echo "Installing dependencies..."
venv/bin/pip install -r requirements.txt --quiet --disable-pip-version-check

# Start the server using the venv Python
echo "Starting FastAPI server on http://localhost:8000"
exec venv/bin/python -m uvicorn main:app --reload \
    --reload-exclude "$PWD/venv" \
    --reload-exclude "$PWD/__pycache__" \
    --reload-exclude "$PWD/runs" \
    --reload-exclude "$PWD/workspace" \
    --host 0.0.0.0 --port 8000
