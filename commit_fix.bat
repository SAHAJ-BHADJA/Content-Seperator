@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Fix ground truth file detection for all video locations"
git push origin main
del "%~f0"
