@echo off
title Manual ML Model Tester (ESP32-CAM)
cd /d "%~dp0"
echo =========================================================================
echo   Manual YOLOv8 Trash Detection Tester (ESP32-CAM Feed)
echo   GPU: NVIDIA GeForce RTX 4060
echo =========================================================================
echo.
python test_manual_ml_cam.py %*
pause
