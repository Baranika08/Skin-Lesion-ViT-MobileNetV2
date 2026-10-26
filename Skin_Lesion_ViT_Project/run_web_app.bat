@echo off
title Skin Lesion AI - Web Application
color 0A
cd /d "%~dp0"

echo ======================================================================
echo    SKIN LESION RECOGNITION SUITE - WEB APP LAUNCHER
echo    Vision Transformer (ViT + SimMIM) ^& MobileNetV2 Baseline
echo ======================================================================
echo.

:: 1. Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [ERROR] Python is not installed or not found in system PATH.
    echo.
    echo Please install Python 3.9, 3.10, 3.11, or 3.12 from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check the box "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)
python --version

:: 2. Check / Create Virtual Environment
if exist .venv goto VENV_EXISTS
echo [2/4] First-time setup: Creating virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Virtual environment created successfully.
goto VENV_READY

:VENV_EXISTS
echo [2/4] Virtual environment (.venv) found.

:VENV_READY
call .venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 3. Install Requirements
echo [3/4] Checking and installing required packages (PyTorch, Streamlit, etc.)...
echo       (If this is the first run, downloading packages may take 1-3 minutes)
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Package installation failed. Please check your internet connection.
    pause
    exit /b 1
)

:: 4. Free Port 8501 if previously occupied
powershell -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

:: 5. Bypass Streamlit first-time email prompt
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

:: 6. Launch App (Streamlit will open the browser automatically when ready)
echo.
echo [4/4] Starting Streamlit server on http://127.0.0.1:8501 ...
echo       Please wait a few seconds for model weights to initialize.
echo       Your browser will open automatically once the server is ready!
echo.
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
pause
