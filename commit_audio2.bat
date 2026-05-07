@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Improve audio debugging and fix initialization"
git push origin main
del "%~f0"
