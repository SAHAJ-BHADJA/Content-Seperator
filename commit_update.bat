@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Add synchronized audio playback and improve player UI"
git push --force origin main
del "%~f0"
