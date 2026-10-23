@echo off
setlocal
cd /d "%~dp0"
"D:\itl\eg\pinn_gpu\python.exe" -B -u "22_benchmark_mesh_speed.py" %*
if errorlevel 1 exit /b 1
endlocal
