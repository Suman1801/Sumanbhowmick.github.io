@echo off
set /p commitMsg=Enter commit message: 
git add .
git commit -m "%commitMsg%"
git push origin main
quarto publish gh-pages
pause
