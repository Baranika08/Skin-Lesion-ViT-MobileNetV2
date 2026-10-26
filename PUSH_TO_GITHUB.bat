@echo off
title Push Project to GitHub
color 0B
cd /d "%~dp0"

echo ======================================================================
echo             PUSH PROJECT TO GITHUB REPOSITORY
echo ======================================================================
echo.

:: Check git
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

echo Repository is initialized and committed on branch 'main'.
echo.
git remote -v
echo.

set /p REPO_URL="Enter the GitHub Repository URL (e.g. https://github.com/username/repo.git): "
if "%REPO_URL%"=="" (
    echo [ERROR] No URL provided.
    pause
    exit /b 1
)

:: Check if origin already exists
git remote get-url origin >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Updating remote origin to: %REPO_URL%
    git remote set-url origin %REPO_URL%
) else (
    echo Adding remote origin: %REPO_URL%
    git remote add origin %REPO_URL%
)

echo.
echo Pushing branch 'main' to GitHub...
echo (If prompted, log in via your browser or paste a GitHub Personal Access Token)
echo.
git push -u origin main

if %ERRORLEVEL% equ 0 (
    color 0A
    echo.
    echo ======================================================================
    echo    SUCCESSFULLY PUSHED TO GITHUB!
    echo    View repository at: %REPO_URL%
    echo ======================================================================
) else (
    color 0C
    echo.
    echo [ERROR] Push failed.
    echo Common reasons:
    echo 1. You don't have write permissions to your friend's repository.
    echo    Ask your friend to invite your GitHub account as a Collaborator
    echo    (Repo Settings -> Collaborators -> Add people).
    echo 2. The remote repository already has commits.
    echo    To force push, run: git push -u origin main --force
)
echo.
pause
