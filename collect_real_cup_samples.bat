@echo off
title Real Cup & Bottle Cap Live Collector
cd /d "%~dp0"
echo =========================================================================
echo   Real-World Cup & Bottle Cap Sample Collector
echo   Controls:
echo     [SPACE] / [C] - Save CUP (Auto-detected)
echo     [B]           - Save BOTTLE CAP (Auto-detected)
echo     [Click & Drag]- Draw manual bounding box
echo     [Q] / [ESC]   - Finish
echo =========================================================================
echo.
echo Launching camera collector...
python collect_real_cup_samples.py %*
pause
