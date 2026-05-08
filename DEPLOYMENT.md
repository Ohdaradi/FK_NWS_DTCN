# 🚀 Deployment Guide - Streamlit Community Cloud

## ⚡ Quick Start (3 Steps)

### Step 1: Create GitHub Repository
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `FK_NWS_DTCN`
3. Description: `Fake News Detection Using NLP - SECTION-2I1 Group 10`
4. Choose **Public** (required for free Streamlit Cloud)
5. **DO NOT** initialize with README (we have one)
6. Click "Create repository"

### Step 2: Push Code to GitHub
Run this in PowerShell (from project directory):

```powershell
cd d:\FK_NWS_DTCN

# Configure Git (one time only)
git config --global user.email "your.email@example.com"
git config --global user.name "Your GitHub Username"

# Add GitHub remote (replace USERNAME/REPO with your details)
git remote add origin https://github.com/YOUR_USERNAME/FK_NWS_DTCN.git
git branch -M main
git push -u origin main
```

**Or use the automated script:**
```powershell
.\deploy.bat
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub (authorize Streamlit)
3. Click "New app"
4. **GitHub repo:** `YOUR_USERNAME/FK_NWS_DTCN`
5. **Branch:** `main`
6. **Main file path:** `app.py`
7. Click "Deploy"

✅ **Your app will be live in 1-2 minutes!**

---

## 📋 What's Included in This Setup

- ✅ `requirements.txt` - All dependencies
- ✅ `README.md` - Project documentation
- ✅ `.gitignore` - Excludes unnecessary files
- ✅ `.streamlit/config.toml` - Theme configuration
- ✅ `deploy.bat` / `deploy.sh` - Automated deployment scripts
- ✅ `.git` - Git repository initialized

---

## 🔄 Updating After Deployment

After any changes, just run:

```powershell
# Windows
.\deploy.bat

# macOS/Linux
./deploy.sh
```

Or manually:
```powershell
git add .
git commit -m "Your message"
git push origin main
```

Streamlit Cloud auto-deploys on push! 🎉

---

## ❓ Troubleshooting

**Q: "Repository not found"**
- Verify you created a PUBLIC repo on GitHub
- Check the URL is correct: `https://github.com/USERNAME/REPO.git`

**Q: "ModuleNotFoundError" on deploy**
- Make sure all imports in `app.py` are in `requirements.txt`
- Check `requirements.txt` was pushed to GitHub

**Q: "Permission denied (publickey)"**
- You may need to set up SSH keys: [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

**Q: Build fails on Streamlit Cloud**
- Check the "Manage app" → "Logs" tab
- Ensure all `.pkl` files are in the repo
- Verify all imports are available

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud/get-started)
- [GitHub Docs](https://docs.github.com)

---

**Created:** May 9, 2026  
**Project:** Fake News Detection Using NLP  
**Group:** SECTION-2I1 Group 10
