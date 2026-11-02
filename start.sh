#!/usr/bin/env bash
# Start the Data Capture Tool on Linux / Jetson.   Usage:  bash start.sh     (or: chmod +x start.sh && ./start.sh)
# Uses ./venv if it exists, otherwise python3. Override with:  PYTHON=/path/to/python bash start.sh
cd "$(dirname "$0")" || exit 1
if [ -n "$PYTHON" ]; then PY="$PYTHON"
elif [ -x venv/bin/python ]; then PY=venv/bin/python
else PY=python3; fi
echo "Using $PY"
exec "$PY" backend/app.py
