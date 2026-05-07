@echo off
cd /d "D:\USC\Subjects\Multimedia System Design\Project"
git add .
git commit -m "Add Evaluate Accuracy button to player UI"
git push origin main
del "%~f0"
