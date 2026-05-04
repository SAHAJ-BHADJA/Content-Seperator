@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "CSCI 576 Multimedia Project: Multimodal Video Segmentation System"
git remote add origin https://github.com/SAHAJ-BHADJA/Content-Seperator.git
git push --force origin main
del "%~f0"
