@echo off
setlocal

REM Path to venv python.exe
set VENV_PY= "%cd%\venv\Scripts\python.exe"

if not exist %VENV_PY% (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)



echo Running app.py...
%VENV_PY% app.py

echo.
echo Done!
pause
