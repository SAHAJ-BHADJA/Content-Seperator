@echo off
REM CSCI 576 Multimedia Project - Batch Analysis
REM Analyzes all videos using ground truth segmentation

echo ============================================
echo CSCI 576 Batch Video Analysis
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Analyzing all test videos with ground truth...
echo.

python main.py --batch videos_with_ads --ground-truth --output segmentation_results

echo.
echo Analysis complete!
pause
