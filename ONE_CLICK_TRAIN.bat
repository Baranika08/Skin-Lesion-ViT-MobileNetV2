@echo off
title Skin Lesion AI - Training Pipeline
color 0B
cd /d "%~dp0\Skin_Lesion_ViT_Project"

echo ======================================================================
echo    SKIN LESION RECOGNITION - FULL TRAINING ^& BENCHMARK PIPELINE
echo    Compares MobileNetV2 vs ViT across Label Budgets (10% - 100%)
echo ======================================================================
echo.

call RUN_ME.bat %*
pause
