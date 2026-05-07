@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Tune classifier for better accuracy - reduce false positives"
git push origin main
del "%~f0"
