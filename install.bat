@echo off
setlocal enabledelayedexpansion
set "HERE=%~dp0"
cd /d "%HERE%"

echo ============================================
echo  O2noor LTX-2.5 Int4 Tile-Train - installer
echo ============================================

REM 1) Prefer a Python already on PATH (normal installs).
set "PYTHON="
where python >nul 2>nul && set "PYTHON=python"

REM 2) Fall back to ComfyUI's bundled Python (machines with only ComfyUI).
for %%P in (
  "%HERE%..\..\.venv\Scripts\python.exe"
  "%HERE%..\..\..\.venv\Scripts\python.exe"
  "%HERE%..\.venv\Scripts\python.exe"
  "%LOCALAPPDATA%\Programs\ComfyUI\.venv\Scripts\python.exe"
  "%LOCALAPPDATA%\Programs\ComfyUI\python_embeded\python.exe"
) do (
  if not defined PYTHON if exist "%%~P" set "PYTHON=%%~P"
)

if not defined PYTHON (
  echo.
  echo Could not auto-detect a Python.
  echo If ComfyUI is installed, run install.py with its bundled python:
  echo    "<path-to-ComfyUI>\.venv\Scripts\python.exe" install.py
  echo.
  echo Press any key to close.
  pause >nul
  exit /b 1
)

echo Using Python: %PYTHON%
echo.
"%PYTHON%" install.py

echo.
echo Done. Restart ComfyUI to load the O2noor LTX-2.5 nodes.
echo Press any key to close.
pause >nul
