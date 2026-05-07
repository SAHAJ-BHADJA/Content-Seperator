@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Major improvement to ad detection - scene pair analysis and lower thresholds"
git push origin main
del "%~f0"
