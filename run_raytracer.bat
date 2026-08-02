@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe kerr_raytracer.py %*
pause
