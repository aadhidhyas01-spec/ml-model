@echo off
title YOLOv8 Extreme Accuracy Training (Cups & Bottle Caps)
cd /d "%~dp0"
echo =========================================================================
echo   YOLOv8 Extreme Training Engine (Cups & Bottle Caps Optimization)
echo   GPU: NVIDIA GeForce RTX 4060 (8GB VRAM)
echo   Resolution: 800x800 ^| Close-Mosaic: 15 ^| Epochs: 150
echo =========================================================================
echo.
python train_extreme_cup_cap.py %*
pause
