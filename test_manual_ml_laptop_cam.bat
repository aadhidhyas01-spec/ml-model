@echo off
title Manual ML Model Tester (Laptop Webcam)
cd /d "%~dp0"
echo =========================================================================
echo   Manual YOLOv8 Trash Detection Tester (Laptop Webcam)
echo   GPU: NVIDIA GeForce RTX 4060
echo =========================================================================
echo.
python -s test_manual_ml_cam.py --source 0
pause
