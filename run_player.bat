@echo off
REM CSCI 576 Multimedia Project - Video Player Launcher
REM Quick launcher for the video segmentation player

echo ============================================
echo CSCI 576 Multimodal Video Segmentation Player
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Run the demo
if "%1"=="" (
    echo Launching player with first available video...
    python run_demo.py
) else (
    echo Opening video: %1
    python main.py "%1"
)

pause
