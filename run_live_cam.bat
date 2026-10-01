@echo off
title ESP32-CAM Live Stream Viewer
cd /d "%~dp0"
echo ========================================================
echo        Opening ESP32-CAM Live Video Stream
echo        Stream URL: http://192.168.137.202:81/stream
echo ========================================================
echo.
python view_cam.py
pause
