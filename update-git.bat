@echo off
title 🚀 Git Commit + Quarto Publish

echo ===========================
echo 🔄 Committing Site Changes
echo ===========================
set /p commitMsg=Enter commit message: 

git add .
git commit -m "%commitMsg%"
git push origin main

echo ===========================
echo 🛠️ Rendering Site with Quarto
echo ===========================
quarto render

echo ===========================
echo 🚀 Publishing to gh-pages
echo ===========================
quarto publish gh-pages

echo ✅ Done!
pause
