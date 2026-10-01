@echo off
title Smart Waste Management Robot - AI Vision & Autonomous Controller
cd /d "%~dp0"
echo =========================================================================
echo   Smart Waste Management Robot: AI Vision + Robotic Manipulation
echo   Camera Feed : http://192.168.137.202:81/stream
echo   Robot DevKit: 192.168.137.227:8888
echo   AI Model    : YOLOv8m (NVIDIA RTX 4060 GPU)
echo =========================================================================
echo.
echo Controls:
echo   - Press 'q' or ESC in video window to exit safely
echo.
python robot_wireless_controller.py %*
pause
