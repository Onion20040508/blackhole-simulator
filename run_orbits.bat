@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe geodesic_orbits.py %*
pause
