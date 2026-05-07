@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Use WAV format for audio extraction instead of MP3"
git push origin main
del "%~f0"
