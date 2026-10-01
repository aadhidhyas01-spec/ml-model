@echo off
title YOLOv8 Iterative Train and Test Engine
cd /d "%~dp0"

echo ======================================================================
echo    YOLOv8 ITERATIVE TRAIN & TEST ENGINE (TARGET: ^>90%% ACCURACY)
echo ======================================================================
echo  This script will train the model in rounds, testing after each round,
echo  tracking metrics and saving the highest accuracy weights.
echo ======================================================================
echo.

python auto_train_test_loop.py --rounds 5 --epochs-per-round 25 --batch 16 --imgsz 640 --target 0.90 --metric precision

echo.
echo ======================================================================
echo  Training and testing loop completed!
echo ======================================================================
pause
