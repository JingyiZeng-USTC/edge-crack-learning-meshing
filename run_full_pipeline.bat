@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if defined PIPELINE_PYTHON goto run
if defined CONDA_PREFIX (
  set "PIPELINE_PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PIPELINE_PYTHON=D:\itl\eg\pinn_gpu\python.exe"
)
:run
if not exist "%PIPELINE_PYTHON%" (
  echo Python not found. Activate conda or set PIPELINE_PYTHON.
  exit /b 1
)
"%PIPELINE_PYTHON%" -B -u "run_full_pipeline.py" %*
if errorlevel 1 exit /b 1
endlocal
