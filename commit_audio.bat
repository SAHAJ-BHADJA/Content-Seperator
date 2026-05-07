@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Fix audio playback - use play() instead of unpause()"
git push origin main
del "%~f0"
