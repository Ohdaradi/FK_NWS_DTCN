#!/bin/bash

# Deployment script for Fake News Detection app

echo ""
echo "========================================"
echo "  Fake News Detection Deployment Script"
echo "========================================"
echo ""

# Check if git is configured
if ! git config --global user.email &> /dev/null; then
    echo "[SETUP] Configuring Git..."
    read -p "Enter your GitHub email: " EMAIL
    read -p "Enter your GitHub username: " USERNAME
    git config --global user.email "$EMAIL"
    git config --global user.name "$USERNAME"
    echo "[OK] Git configured!"
fi

# Check if remote is set
if ! git remote get-url origin &> /dev/null; then
    echo ""
    echo "[REQUIRED] GitHub repository not linked!"
    read -p "Enter your GitHub repo URL (e.g., https://github.com/username/FK_NWS_DTCN.git): " REPO_URL
    git remote add origin "$REPO_URL"
    echo "[OK] Remote added!"
fi

echo ""
echo "[STEP 1] Pulling latest changes..."
git pull origin main 2>/dev/null || true

echo "[STEP 2] Staging changes..."
git add .

echo "[STEP 3] Committing..."
read -p "Commit message [default: 'Update app']: " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"Update app"}
git commit -m "$COMMIT_MSG"

echo "[STEP 4] Pushing to GitHub..."
git push -u origin main

echo ""
echo "========================================"
echo "  Deployment Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Click 'New app'"
echo "3. Select your GitHub repo"
echo "4. App will be live in 1-2 minutes!"
echo ""
