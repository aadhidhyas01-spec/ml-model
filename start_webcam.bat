@echo off
title YOLOv8 Trash Detection - Real-time Webcam
cd /d "%~dp0"
echo ========================================================
echo  Starting YOLOv8 Real-time Webcam Trash Detection
echo  GPU: NVIDIA GeForce RTX 4060
echo ========================================================
echo.
echo Controls inside webcam window:
echo   - Press 'q' or ESC to exit
echo   - Press '+' or '-' to adjust confidence threshold
echo   - Press 's' to save a snapshot photo
echo.

python -s main.py --mode webcam --weights weights/best_trash_yolov8m.pt --conf 0.50

if errorlevel 1 (
    echo.
    echo Testing with secondary camera ID 1...
    python -s main.py --mode webcam --weights weights/best_trash_yolov8m.pt --conf 0.50 --camera_id 1
)

pause
