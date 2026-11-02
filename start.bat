@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" backend\app.py
) else (
    python backend\app.py
)
pause
