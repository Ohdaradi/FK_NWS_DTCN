@echo off
REM Deployment script for Fake News Detection app

echo.
echo ========================================
echo  Fake News Detection Deployment Script
echo ========================================
echo.

REM Check if git is configured
git config --global user.email >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Configuring Git...
    set /p EMAIL="Enter your GitHub email: "
    set /p USERNAME="Enter your GitHub username: "
    git config --global user.email "%EMAIL%"
    git config --global user.name "%USERNAME%"
    echo [OK] Git configured!
)

REM Check if remote is set
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo [REQUIRED] GitHub repository not linked!
    echo Please enter your GitHub repository URL:
    set /p REPO_URL="GitHub repo URL (e.g., https://github.com/username/FK_NWS_DTCN.git): "
    git remote add origin %REPO_URL%
    echo [OK] Remote added!
)

echo.
echo [STEP 1] Pulling latest changes...
git pull origin main 2>nul

echo [STEP 2] Staging changes...
git add .

echo [STEP 3] Committing...
set /p COMMIT_MSG="Commit message [default: 'Update app']: "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update app
git commit -m "%COMMIT_MSG%"

echo [STEP 4] Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo  Deployment Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Click "New app"
echo 3. Select your GitHub repo
echo 4. App will be live in 1-2 minutes!
echo.
pause
