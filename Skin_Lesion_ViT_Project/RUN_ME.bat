@echo off
title Skin Lesion AI - Training Pipeline
color 0B
cd /d "%~dp0"

echo ======================================================================
echo    SKIN LESION RECOGNITION - TRAINING ^& BENCHMARK PIPELINE
echo    MobileNetV2 vs ViT with Masked Self-Supervised Pretraining
echo ======================================================================
echo.

:: 1. Check Python
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python (3.9 - 3.12) from https://www.python.org/
    pause
    exit /b 1
)
python --version

:: 2. Check Virtual Environment
if exist .venv goto VENV_EXISTS
echo [2/3] Creating virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
goto VENV_READY

:VENV_EXISTS
echo [2/3] Virtual environment (.venv) found.

:VENV_READY
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 3. Run Pipeline
echo.
echo [3/3] Running training, evaluation, figures, and report generation...
echo.
python run_all.py %*

echo.
echo ======================================================================
echo    TRAINING ^& REPORT GENERATION COMPLETED!
echo    Check the 'documents' and 'results' folders for outputs.
echo ======================================================================
echo.
pause
