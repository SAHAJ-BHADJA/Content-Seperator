@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Improve ad detection with visual anomaly detection and scene splitting"
git push origin main
del "%~f0"
