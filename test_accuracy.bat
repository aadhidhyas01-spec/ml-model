@echo off
title YOLOv8 Overall Accuracy Tester
cd /d "%~dp0"

echo ======================================================================
echo    YOLOv8 OVERALL ACCURACY & BENCHMARK TEST
echo ======================================================================
echo.

python -s test_accuracy.py --weights weights/best_trash_yolov8m.pt --imgsz 640 --conf 0.25

echo.
echo ======================================================================
echo  Accuracy benchmark finished!
echo ======================================================================
pause
