@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Add speech analysis, export features, and full player functionality"
git push --force origin main
del "%~f0"
