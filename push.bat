@echo off
echo.
echo ========================================
echo  Sen's AI Detector - Git Push
echo ========================================
echo.

cd /d C:\ai_detector2

echo Staging all changes...
git add .

echo.
git status

echo.
set /p msg="Enter commit message (or press Enter for 'Update'): "
if "%msg%"=="" set msg=Update

git commit -m "%msg%"

echo.
echo Pushing to GitHub...
git push

echo.
echo ========================================
echo  Done! Check Render in 2-3 minutes.
echo  https://tinyurl.com/AI-Detector-by-Sen
echo ========================================
echo.
pause
